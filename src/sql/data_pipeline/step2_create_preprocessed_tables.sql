-- Copyright 2024 Google LLC.
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     https://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

-- SQL query to build machine learning training & prediction dataset to predict product refund amount at transaction level.
-- Step 2: Preprocess GA4 data.

-- @param ga4_project_id STRING Project id on Google Cloud Platform for the GA4 dataset (source data).
-- @param ga4_dataset_id STRING Dataset id of the GA4 dataset (source data).
-- @param project_id STRING Project id on Google Cloud Platform for intermediate tables and final output tables.
-- @param dataset_id STRING Dataset id of intermediate tables and final output tables.
-- @param return_policy_window_in_days INT The number of days after the transaction date that allows for product return/refund.
-- @param data_pipeline_type STRING Whether to process data for model training or for prediction.
-- @param end_date DATE Transaction data that is older than X days (return policy window) ago to make sure the refund is all set.

-- Set data_pipeline_type to determine whether to process data for training or for prediction
DECLARE data_pipeline_type STRING DEFAULT '{data_pipeline_type}';

-- Declare end_date to make sure to use transaction data that is older than X days (return policy window) ago to make sure the refund is all set.
DECLARE end_date DATE DEFAULT (CASE data_pipeline_type WHEN 'TRAINING' THEN DATE_ADD(CURRENT_DATE(), INTERVAL -{return_policy_window_in_days}-1 DAY) ELSE CURRENT_DATE() END);

-- Identifies invalid string values in the GA4 data and sets them to NULL.
CREATE TEMP FUNCTION NormalizeString(input STRING)
RETURNS STRING
AS (
  IF((input = '(not set)' OR input = ''), NULL, input)
);

-- Creates Channel Grouping from GA4 data based on traffic source, medium and name.
CREATE TEMP FUNCTION GetChannelGrouping(
  source STRING,
  medium STRING,
  name STRING)
RETURNS STRING
AS (
  CASE
    WHEN
      source = '(direct)'
      AND (medium IN ('(not set)', '(none)'))
      THEN 'Direct'
    WHEN regexp_contains(name, 'cross-network') THEN 'Cross-network'
    WHEN
      (
        regexp_contains(
          source,
          'alibaba|amazon|google shopping|shopify|etsy|ebay|stripe|walmart')
        OR regexp_contains(name, '^(.*(([^a-df-z]|^)shop|shopping).*)$'))
      AND regexp_contains(medium, '^(.*cp.*|ppc|paid.*)$')
      THEN 'Paid Shopping'
    WHEN
      regexp_contains(
        source, 'baidu|bing|duckduckgo|ecosia|google|yahoo|yandex')
      AND regexp_contains(medium, '^(.*cp.*|ppc|paid.*)$')
      THEN 'Paid Search'
    WHEN
      regexp_contains(
        source,
        'badoo|facebook|fb|instagram|linkedin|pinterest|tiktok|twitter|whatsapp')
      AND regexp_contains(medium, '^(.*cp.*|ppc|paid.*)$')
      THEN 'Paid Social'
    WHEN
      regexp_contains(
        source,
        'dailymotion|disneyplus|netflix|youtube|vimeo|twitch|vimeo|youtube')
      AND regexp_contains(medium, '^(.*cp.*|ppc|paid.*)$')
      THEN 'Paid Video'
    WHEN
      medium IN ('display', 'banner', 'expandable', 'interstitial', 'cpm')
      THEN 'Display'
    WHEN
      regexp_contains(
        source,
        'alibaba|amazon|google shopping|shopify|etsy|ebay|stripe|walmart')
      OR regexp_contains(name, '^(.*(([^a-df-z]|^)shop|shopping).*)$')
      THEN 'Organic Shopping'
    WHEN
      regexp_contains(
        source,
        'badoo|facebook|fb|instagram|linkedin|pinterest|tiktok|twitter|whatsapp')
      OR medium IN (
        'social', 'social-network', 'social-media', 'sm', 'social network', 'social media')
      THEN 'Organic Social'
    WHEN
      regexp_contains(
        source,
        'dailymotion|disneyplus|netflix|youtube|vimeo|twitch|vimeo|youtube')
      OR regexp_contains(medium, '^(.*video.*)$')
      THEN 'Organic Video'
    WHEN
      regexp_contains(
        source, 'baidu|bing|duckduckgo|ecosia|google|yahoo|yandex')
      OR medium = 'organic'
      THEN 'Organic Search'
    WHEN
      regexp_contains(source, 'email|e-mail|e_mail|e mail')
      OR regexp_contains(medium, 'email|e-mail|e_mail|e mail')
      THEN 'Email'
    WHEN medium = 'affiliate' THEN 'Affiliates'
    WHEN medium = 'referral' THEN 'Referral'
    WHEN medium = 'audio' THEN 'Audio'
    WHEN medium = 'sms' THEN 'SMS'
    WHEN
      medium LIKE '%push'
      OR regexp_contains(medium, 'mobile|notification')
      THEN 'Mobile Push Notifications'
    ELSE 'Unassigned'
    END
);

-- Transaction id to user_id mappig
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_id_to_user_id_mapping`
AS (
  SELECT DISTINCT
    Ga4Table.ecommerce.transaction_id AS transaction_id,
    PARSE_DATE('%Y%m%d', Ga4Table.event_date) AS transaction_date,
    Ga4Table.user_id AS user_id
  FROM `{ga4_project_id}.{ga4_dataset_id}.events_*` AS Ga4Table
  INNER JOIN `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_lookup` AS TransLookup
    ON
      TransLookup.transaction_id = Ga4Table.ecommerce.transaction_id
      AND TransLookup.transaction_date = PARSE_DATE('%Y%m%d', Ga4Table.event_date)
  INNER JOIN `{project_id}.{dataset_id}.{data_pipeline_type}_user_id_lookup` AS UserLookup
    ON UserLookup.user_id = Ga4Table.user_id
);

-- Transaction id to pseudo_user_id Mappig
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_id_to_pseudo_user_id_mapping`
AS (
  SELECT DISTINCT
    Ga4Table.ecommerce.transaction_id AS transaction_id,
    PARSE_DATE('%Y%m%d', Ga4Table.event_date) AS transaction_date,
    Ga4Table.user_pseudo_id
  FROM `{ga4_project_id}.{ga4_dataset_id}.events_*` AS Ga4Table
  INNER JOIN `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_lookup` AS TransLookup
    ON
      TransLookup.transaction_id = Ga4Table.ecommerce.transaction_id
      AND TransLookup.transaction_date = PARSE_DATE('%Y%m%d', event_date)
  INNER JOIN `{project_id}.{dataset_id}.{data_pipeline_type}_user_pseudo_id_lookup` AS PseudoUserLookup
    ON PseudoUserLookup.user_pseudo_id = Ga4Table.user_pseudo_id
);

-- Transaction id to pseudo user id and item mapping
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_id_to_pseudo_user_id_item_mapping`
AS (
  SELECT DISTINCT
    Ga4Table.ecommerce.transaction_id AS transaction_id,
    PARSE_DATE('%Y%m%d', Ga4Table.event_date) AS transaction_date,
    Items.item_name AS item_name,
    Ga4Table.user_pseudo_id
  FROM `{ga4_project_id}.{ga4_dataset_id}.events_*` AS Ga4Table
  CROSS JOIN UNNEST(items) AS Items
  INNER JOIN `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_lookup` AS TransLookup
    ON
      TransLookup.transaction_id = Ga4Table.ecommerce.transaction_id
      AND TransLookup.transaction_date = PARSE_DATE('%Y%m%d', event_date)
  INNER JOIN `{project_id}.{dataset_id}.{data_pipeline_type}_item_lookup` AS ItemsLookup
    ON ItemsLookup.item_name = CAST(Items.item_name AS STRING)
  INNER JOIN `{project_id}.{dataset_id}.{data_pipeline_type}_user_pseudo_id_lookup` AS PseudoIdLookup
    ON PseudoIdLookup.user_pseudo_id = Ga4Table.user_pseudo_id
);

-- Transaction id to user id and item mapping
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_id_to_user_id_item_mapping`
AS (
  SELECT DISTINCT
    Ga4Table.ecommerce.transaction_id AS transaction_id,
    PARSE_DATE('%Y%m%d', Ga4Table.event_date) AS transaction_date,
    CAST(Items.item_name AS STRING) AS item_name,
    Ga4Table.user_id AS user_id
  FROM `{ga4_project_id}.{ga4_dataset_id}.events_*` AS Ga4Table
  CROSS JOIN UNNEST(items) AS items
  INNER JOIN `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_lookup` AS TransLookup
    ON
      TransLookup.transaction_id = Ga4Table.ecommerce.transaction_id
      AND TransLookup.transaction_date = PARSE_DATE('%Y%m%d', Ga4Table.event_date)
  INNER JOIN `{project_id}.{dataset_id}.{data_pipeline_type}_item_lookup` AS ItemsLookup
    ON ItemsLookup.item_name = CAST(Items.item_name AS STRING)
  INNER JOIN `{project_id}.{dataset_id}.{data_pipeline_type}_user_id_lookup` AS UserIdLookup
    ON UserIdLookup.user_id = Ga4Table.user_id
);

-- Transaction id to session id mapping
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_id_to_session_id_mapping`
AS (
  SELECT DISTINCT
    Ga4Table.ecommerce.transaction_id AS transaction_id,
    PARSE_DATE('%Y%m%d', Ga4Table.event_date) AS transaction_date,
    SessionIdLookup.session_id,
  FROM `{ga4_project_id}.{ga4_dataset_id}.events_*` AS Ga4Table
  INNER JOIN `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_lookup` AS TransLookup
    ON
      TransLookup.transaction_id = Ga4Table.ecommerce.transaction_id
      AND TransLookup.transaction_date = PARSE_DATE('%Y%m%d', event_date)
  INNER JOIN `{project_id}.{dataset_id}.{data_pipeline_type}_session_id_lookup` AS SessionIdLookup
    ON
      SessionIdLookup.session_id = CONCAT(
        CAST(Ga4Table.user_pseudo_id AS STRING),
        '_',
        (
          SELECT CAST(value.int_value AS STRING)
          FROM UNNEST(event_params)
          WHERE key = 'ga_session_id'
        ))
);

-- Transaction Level (transaction id & transaction date) Purchase & Refund Stats
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_transaction_data`
AS (
  SELECT
    Ga4Table.ecommerce.transaction_id AS transaction_id,
    PARSE_DATE('%Y%m%d', Ga4Table.event_date) AS transaction_date,
    SUM(Ga4Table.ecommerce.purchase_revenue_in_usd) AS purchase_revenue_in_usd,
    SUM(Ga4Table.ecommerce.purchase_revenue) AS purchase_revenue,
    SUM(Ga4Table.ecommerce.refund_value_in_usd) AS refund_value_in_usd,
    SUM(Ga4Table.ecommerce.refund_value) AS refund_value,
    SUM(Ga4Table.ecommerce.unique_items) AS unique_items,
    SUM(Ga4Table.ecommerce.shipping_value) AS shipping_value,
    SUM(Ga4Table.ecommerce.shipping_value_in_usd) AS shipping_value_in_usd
  FROM `{ga4_project_id}.{ga4_dataset_id}.events_*` AS Ga4Table
  INNER JOIN `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_lookup` AS TransLookup
    ON
      TransLookup.transaction_id = Ga4Table.ecommerce.transaction_id
      AND TransLookup.transaction_date = PARSE_DATE('%Y%m%d', Ga4Table.event_date)
  GROUP BY 1, 2
);

-- Daily level transaction & refund stats
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_daily_level_transaction_refund_data`
AS (
  SELECT
    PARSE_DATE('%Y%m%d', Ga4Table.event_date) AS transaction_date,
    SUM(Ga4Table.ecommerce.purchase_revenue_in_usd) AS purchase_revenue_in_usd,
    SUM(Ga4Table.ecommerce.purchase_revenue) AS purchase_revenue,
    SUM(Ga4Table.ecommerce.refund_value_in_usd) AS refund_value_in_usd,
    SUM(Ga4Table.ecommerce.refund_value) AS refund_value
  FROM `{project_id}.{dataset_id}.FilterTransactionData`(end_date) AS Ga4Table
  GROUP BY 1
);

-- Daily level past transaction & refund stats
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_daily_level_past_transaction_refund_data`
AS (
  SELECT
    TransLookup.transaction_date AS transaction_date,
    SUM(Ga4Table.ecommerce.purchase_revenue_in_usd) AS past_purchase_revenue_in_usd,
    SUM(Ga4Table.ecommerce.purchase_revenue) AS past_purchase_revenue,
    SUM(Ga4Table.ecommerce.refund_value_in_usd) AS past_refund_value_in_usd,
    SUM(Ga4Table.ecommerce.refund_value) AS past_refund_value,
    COUNTIF(IFNULL(Ga4Table.ecommerce.refund_value, 0) > 0) AS past_refund_count,
    COUNTIF(IFNULL(Ga4Table.ecommerce.purchase_revenue, 0) > 0) AS past_trans_count,
    SAFE_DIVIDE(SUM(Ga4Table.ecommerce.refund_value), SUM(Ga4Table.ecommerce.purchase_revenue))
      AS past_grand_avg_refund_amt_proportion,
    SAFE_DIVIDE(
      COUNTIF(IFNULL(Ga4Table.ecommerce.refund_value, 0) > 0),
      COUNTIF(IFNULL(Ga4Table.ecommerce.purchase_revenue, 0) > 0)) AS past_grand_avg_refund_rate
  FROM `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_date_lookup` AS TransLookup
  LEFT JOIN `{project_id}.{dataset_id}.FilterTransactionData`(end_date) AS Ga4Table
    ON TransLookup.transaction_date > PARSE_DATE('%Y%m%d', Ga4Table.event_date)
  GROUP BY 1
);

-- Transaction id & item Level Stats
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_transaction_id_and_item_level_stats`
AS (
  SELECT
    PARSE_DATE('%Y%m%d', Ga4Table.event_date) AS transaction_date,
    Ga4Table.ecommerce.transaction_id AS transaction_id,
    CAST(Items.item_id AS STRING) AS item_id,
    IFNULL(ANY_VALUE(NormalizeString(Items.item_name)), 'unknown') AS item_name,
    IFNULL(ANY_VALUE(NormalizeString(Items.item_brand)), 'unknown') AS item_brand,
    IFNULL(ANY_VALUE(NormalizeString(Items.item_variant)), 'unknown') AS item_variant,
    IFNULL(ANY_VALUE(NormalizeString(Items.item_category)), 'unknown') AS item_category,
    IFNULL(ANY_VALUE(NormalizeString(Items.promotion_id)), 'unknown') <> 'unknown'
      OR IFNULL(ANY_VALUE(NormalizeString(Items.promotion_name)), 'unknown') <> 'unknown'
      OR IFNULL(ANY_VALUE(NormalizeString(Items.coupon)), 'unknown') <> 'unknown'
      AS discounted_product_flag,
    AVG(IFNULL(Items.price, 0)) AS price,
    SUM(Items.quantity) AS quantity,
    SUM(Items.item_revenue_in_usd) AS item_revenue_in_usd,
    SUM(Items.item_revenue) AS item_revenue,
    SUM(Items.item_refund_in_usd) AS item_refund_in_usd,
    SUM(Items.item_refund) AS item_refund,
  FROM `{ga4_project_id}.{ga4_dataset_id}.events_*` AS Ga4Table
  CROSS JOIN UNNEST(items) AS Items
  INNER JOIN `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_lookup` AS TransLookup
    ON
      TransLookup.transaction_id = Ga4Table.ecommerce.transaction_id
      AND TransLookup.transaction_date = PARSE_DATE('%Y%m%d', event_date)
  INNER JOIN `{project_id}.{dataset_id}.{data_pipeline_type}_item_lookup` AS ItemsLookup
    ON ItemsLookup.item_name = CAST(Items.item_name AS STRING)
  GROUP BY 1, 2, 3
);

-- Product level transaction & refund stats
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_product_level_transaction_refund_data`
AS (
  SELECT
    PARSE_DATE('%Y%m%d', event_date) AS transaction_date,
    CAST(Items.item_name AS STRING) AS item_name,
    SUM(Items.item_revenue_in_usd) AS item_revenue_in_usd,
    SUM(Items.item_revenue) AS item_revenue,
    SUM(Items.item_refund_in_usd) AS item_refund_in_usd,
    SUM(Items.item_refund) AS item_refund,
  FROM `{ga4_project_id}.{ga4_dataset_id}.events_*`
  CROSS JOIN UNNEST(items) AS Items
  INNER JOIN `{project_id}.{dataset_id}.{data_pipeline_type}_item_lookup` AS ItemsLookup
    ON ItemsLookup.item_name = CAST(Items.item_name AS STRING)
  GROUP BY 1, 2
);

-- Demographic attributes level transaction & refund stats
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_country_level_transaction_refund_data`
AS (
  SELECT
    TransLookup.transaction_date AS transaction_date,
    NormalizeString(geo.country) AS country,
    SUM(ecommerce.refund_value) AS past_refund_value,
    SUM(ecommerce.purchase_revenue) AS past_purchase_revenue,
    COUNTIF(IFNULL(ecommerce.refund_value, 0) > 0) AS past_refund_count,
    COUNTIF(IFNULL(ecommerce.purchase_revenue, 0) > 0) AS past_trans_count
  FROM `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_date_lookup` AS TransLookup
  LEFT JOIN `{project_id}.{dataset_id}.FilterTransactionData`(end_date) AS Ga4Table
    ON TransLookup.transaction_date > PARSE_DATE('%Y%m%d', Ga4Table.event_date)
  GROUP BY ALL
);

CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_city_level_transaction_refund_data`
AS (
  SELECT
    TransLookup.transaction_date AS transaction_date,
    NormalizeString(geo.city) AS city,
    SUM(ecommerce.refund_value) AS past_refund_value,
    SUM(ecommerce.purchase_revenue) AS past_purchase_revenue,
    COUNTIF(ecommerce.refund_value > 0) AS past_refund_count,
    COUNTIF(ecommerce.purchase_revenue > 0) AS past_trans_count
  FROM `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_date_lookup` AS TransLookup
  LEFT JOIN `{project_id}.{dataset_id}.FilterTransactionData`(end_date) AS Ga4Table
    ON TransLookup.transaction_date > PARSE_DATE('%Y%m%d', Ga4Table.event_date)
  GROUP BY ALL
);

CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_device_category_level_transaction_refund_data`
AS (
  SELECT
    TransLookup.transaction_date AS transaction_date,
    NormalizeString(device.category) AS device_category,
    SUM(ecommerce.refund_value) AS past_refund_value,
    SUM(ecommerce.purchase_revenue) AS past_purchase_revenue,
    COUNTIF(ecommerce.refund_value > 0) AS past_refund_count,
    COUNTIF(ecommerce.purchase_revenue > 0) AS past_trans_count
  FROM `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_date_lookup` AS TransLookup
  LEFT JOIN `{project_id}.{dataset_id}.FilterTransactionData`(end_date) AS Ga4Table
    ON TransLookup.transaction_date > PARSE_DATE('%Y%m%d', Ga4Table.event_date)
  GROUP BY ALL
);

CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_device_operating_system_level_transaction_refund_data`
AS (
  SELECT
    TransLookup.transaction_date AS transaction_date,
    NormalizeString(device.operating_system) AS operating_system,
    SUM(ecommerce.refund_value) AS past_refund_value,
    SUM(ecommerce.purchase_revenue) AS past_purchase_revenue,
    COUNTIF(ecommerce.refund_value > 0) AS past_refund_count,
    COUNTIF(ecommerce.purchase_revenue > 0) AS past_trans_count
  FROM `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_date_lookup` AS TransLookup
  LEFT JOIN `{project_id}.{dataset_id}.FilterTransactionData`(end_date) AS Ga4Table
    ON TransLookup.transaction_date > PARSE_DATE('%Y%m%d', Ga4Table.event_date)
  GROUP BY ALL
);

CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_device_language_level_transaction_refund_data`
AS (
  SELECT
    TransLookup.transaction_date AS transaction_date,
    NormalizeString(Ga4Table.device.language) AS language,
    SUM(Ga4Table.ecommerce.refund_value) AS past_refund_value,
    SUM(Ga4Table.ecommerce.purchase_revenue) AS past_purchase_revenue,
    COUNTIF(Ga4Table.ecommerce.refund_value > 0) AS past_refund_count,
    COUNTIF(Ga4Table.ecommerce.purchase_revenue > 0) AS past_trans_count
  FROM `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_date_lookup` AS TransLookup
  LEFT JOIN `{project_id}.{dataset_id}.FilterTransactionData`(end_date) AS Ga4Table
    ON TransLookup.transaction_date > PARSE_DATE('%Y%m%d', Ga4Table.event_date)
  GROUP BY ALL
);

-- Session id level web traffic data
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_session_id_level_web_traffic_info`
AS (
  SELECT
    SessionIdLookup.session_id,
    TIMESTAMP_MICROS(MAX(Ga4Table.event_timestamp)) AS session_end,
    TIMESTAMP_MICROS(MIN(Ga4Table.event_timestamp)) AS session_start,
    (MAX(Ga4Table.event_timestamp) - MIN(Ga4Table.event_timestamp)) / 1000000
      AS session_length_in_seconds,
    COUNTIF(
      (
        SELECT value.string_value
        FROM UNNEST(Ga4Table.event_params)
        WHERE key = 'session_engaged'
      )
      = '1') AS engaged_sessions,
    SUM(
      IFNULL(
        (
          SELECT value.int_value
          FROM UNNEST(Ga4Table.event_params)
          WHERE key = 'engagement_time_msec'
        ),
        0)
      / 1000) AS engagement_time_seconds,
    -- Traffic Source information
    ARRAY_AGG(
      (SELECT value.string_value FROM UNNEST(Ga4Table.event_params) WHERE key = 'campaign')
        IGNORE NULLS
      ORDER BY event_timestamp)[safe_offset(0)] AS campaign,
    ARRAY_AGG(
      (SELECT value.string_value FROM UNNEST(Ga4Table.event_params) WHERE key = 'medium')
        IGNORE NULLS
      ORDER BY event_timestamp)[safe_offset(0)] AS medium,
    ARRAY_AGG(
      (SELECT value.string_value FROM UNNEST(Ga4Table.event_params) WHERE key = 'source')
        IGNORE NULLS
      ORDER BY event_timestamp)[safe_offset(0)] AS source,
    ARRAY_AGG(
      GetChannelGrouping(
        Ga4Table.traffic_source.source,
        Ga4Table.traffic_source.medium,
        Ga4Table.traffic_source.name))[safe_offset(0)] AS channel_grouping_user
  FROM `{ga4_project_id}.{ga4_dataset_id}.events_*` AS Ga4Table
  INNER JOIN `{project_id}.{dataset_id}.{data_pipeline_type}_session_id_lookup` AS SessionIdLookup
    ON
      SessionIdLookup.session_id = CONCAT(
        CAST(user_pseudo_id AS STRING),
        '_',
        (
          SELECT CAST(value.int_value AS STRING)
          FROM UNNEST(event_params)
          WHERE key = 'ga_session_id'
        ))
  GROUP BY 1
);

-- Session id level web event data
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_session_id_level_web_event_info`
AS (
  SELECT
    SessionIdLookup.session_id,
    Ga4Table.event_name,
    (MAX(Ga4Table.event_timestamp) - MIN(Ga4Table.event_timestamp)) / 1000000
      AS event_length_in_seconds,
    COUNT(*) AS event_count
  FROM `{ga4_project_id}.{ga4_dataset_id}.events_*` AS Ga4Table
  INNER JOIN `{project_id}.{dataset_id}.{data_pipeline_type}_session_id_lookup` AS SessionIdLookup
    ON
      SessionIdLookup.session_id = CONCAT(
        CAST(user_pseudo_id AS STRING),
        '_',
        (
          SELECT CAST(value.int_value AS STRING)
          FROM UNNEST(event_params)
          WHERE key = 'ga_session_id'
        ))
  GROUP BY 1, 2
);

-- Customer purchase/refund history data
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_customer_purchase_refund_history_data_user_id_level`
AS (
  SELECT
    PARSE_DATE('%Y%m%d', Ga4Table.event_date) AS transaction_date,
    Ga4Table.user_id AS user_id,
    SUM(Ga4Table.ecommerce.purchase_revenue_in_usd) AS purchase_revenue_in_usd,
    SUM(Ga4Table.ecommerce.purchase_revenue) AS purchase_revenue,
    SUM(Ga4Table.ecommerce.refund_value_in_usd) AS refund_value_in_usd,
    SUM(Ga4Table.ecommerce.refund_value) AS refund_value,
    SUM(Ga4Table.ecommerce.unique_items) AS unique_items
  FROM `{ga4_project_id}.{ga4_dataset_id}.events_*` AS Ga4Table
  INNER JOIN `{project_id}.{dataset_id}.{data_pipeline_type}_user_id_lookup` AS UserIdLookup
    ON UserIdLookup.user_id = Ga4Table.user_id
  GROUP BY 1, 2
);

-- Customer purchase/refund history data
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_customer_purchase_refund_history_data_user_pseudo_id_level`
AS (
  SELECT
    PARSE_DATE('%Y%m%d', Ga4Table.event_date) AS transaction_date,
    Ga4Table.user_pseudo_id AS user_pseudo_id,
    SUM(Ga4Table.ecommerce.purchase_revenue_in_usd) AS purchase_revenue_in_usd,
    SUM(Ga4Table.ecommerce.purchase_revenue) AS purchase_revenue,
    SUM(Ga4Table.ecommerce.refund_value_in_usd) AS refund_value_in_usd,
    SUM(Ga4Table.ecommerce.refund_value) AS refund_value,
    SUM(Ga4Table.ecommerce.unique_items) AS unique_items
  FROM `{ga4_project_id}.{ga4_dataset_id}.events_*` AS Ga4Table
  INNER JOIN `{project_id}.{dataset_id}.{data_pipeline_type}_user_pseudo_id_lookup` AS PseudoUserLookup
    ON PseudoUserLookup.user_pseudo_id = Ga4Table.user_pseudo_id
  GROUP BY 1, 2
);

-- Customer demographic attributes data
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_customer_demographic_attributes_data_pseudo_user_id_level`
AS (
  SELECT
    Ga4Table.user_pseudo_id AS user_pseudo_id,
    ANY_VALUE(NormalizeString(Ga4Table.geo.country)) AS country,
    ANY_VALUE(NormalizeString(Ga4Table.geo.city)) AS city,
    ANY_VALUE(NormalizeString(Ga4Table.device.category)) AS device_category,
    ANY_VALUE(NormalizeString(Ga4Table.device.operating_system)) AS operating_system,
    ANY_VALUE(NormalizeString(Ga4Table.device.language)) AS language
  FROM `{ga4_project_id}.{ga4_dataset_id}.events_*` AS Ga4Table
  INNER JOIN `{project_id}.{dataset_id}.{data_pipeline_type}_user_pseudo_id_lookup` AS PseudoUserLookup
    ON PseudoUserLookup.user_pseudo_id = Ga4Table.user_pseudo_id
  GROUP BY ALL
);

-- Customer demographic attributes data
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_customer_demographic_attributes_data_user_id_level`
AS (
  SELECT
    Ga4Table.user_id AS user_id,
    ANY_VALUE(NormalizeString(Ga4Table.geo.country)) AS country,
    ANY_VALUE(NormalizeString(Ga4Table.geo.city)) AS city,
    ANY_VALUE(NormalizeString(Ga4Table.device.category)) AS device_category,
    ANY_VALUE(NormalizeString(Ga4Table.device.operating_system)) AS operating_system,
    ANY_VALUE(NormalizeString(Ga4Table.device.language)) AS language
  FROM `{ga4_project_id}.{ga4_dataset_id}.events_*` AS Ga4Table
  INNER JOIN `{project_id}.{dataset_id}.{data_pipeline_type}_user_id_lookup` AS UserIdLookup
    ON UserIdLookup.user_id = Ga4Table.user_id
  GROUP BY ALL
);

-- Customer & Product past purchase/refund interaction history
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_customer_product_purchase_refund_interaction_history_data_user_id_level`
AS (
  SELECT
    PARSE_DATE('%Y%m%d', Ga4Table.event_date) AS transaction_date,
    Ga4Table.user_id AS user_id,
    CAST(Items.item_name AS STRING) AS item_name,
    SUM(Items.item_revenue_in_usd) AS item_revenue_in_usd,
    SUM(Items.item_revenue) AS item_revenue,
    SUM(Items.item_refund_in_usd) AS item_refund_in_usd,
    SUM(Items.item_refund) AS item_refund
  FROM `{ga4_project_id}.{ga4_dataset_id}.events_*` AS Ga4Table
  CROSS JOIN UNNEST(items) AS items
  INNER JOIN `{project_id}.{dataset_id}.{data_pipeline_type}_user_id_lookup` AS UserIdLookup
    ON UserIdLookup.user_id = Ga4Table.user_id
  INNER JOIN `{project_id}.{dataset_id}.{data_pipeline_type}_item_lookup` AS ItemsLookup
    ON ItemsLookup.item_name = CAST(Items.item_name AS STRING)
  GROUP BY 1, 2, 3
);

CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_customer_product_purchase_refund_interaction_history_data_user_pseudo_id_level`
AS (
  SELECT
    PARSE_DATE('%Y%m%d', Ga4Table.event_date) AS transaction_date,
    Ga4Table.user_pseudo_id AS user_pseudo_id,
    CAST(Items.item_name AS STRING) AS item_name,
    SUM(Items.item_revenue_in_usd) AS item_revenue_in_usd,
    SUM(Items.item_revenue) AS item_revenue,
    SUM(Items.item_refund_in_usd) AS item_refund_in_usd,
    SUM(Items.item_refund) AS item_refund
  FROM `{ga4_project_id}.{ga4_dataset_id}.events_*` AS Ga4Table
  CROSS JOIN UNNEST(items) AS items
  INNER JOIN `{project_id}.{dataset_id}.{data_pipeline_type}_item_lookup` AS ItemsLookup
    ON ItemsLookup.item_name = CAST(Items.item_name AS STRING)
  INNER JOIN `{project_id}.{dataset_id}.{data_pipeline_type}_user_pseudo_id_lookup` AS PseudoUserLookup
    ON PseudoUserLookup.user_pseudo_id = Ga4Table.user_pseudo_id
  GROUP BY 1, 2, 3
);

-- Item brand level transaction & refund stats
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_brand_level_transaction_refund_data`
AS (
  SELECT
    TransLookup.transaction_date,
    IFNULL(NormalizeString(Items.item_brand), 'unknown') AS item_brand,
    SUM(Items.item_revenue) AS past_item_revenue,
    SUM(Items.item_refund) AS past_item_refund,
    COUNTIF(IFNULL(Items.item_revenue, 0) > 0) AS past_transaction_count,
    COUNTIF(IFNULL(Items.item_refund, 0) > 0) AS past_refund_count
  FROM `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_date_lookup` AS TransLookup
  LEFT JOIN `{ga4_project_id}.{ga4_dataset_id}.events_*` AS Ga4Table
    ON TransLookup.transaction_date > PARSE_DATE('%Y%m%d', Ga4Table.event_date)
  CROSS JOIN UNNEST(items) AS Items
  GROUP BY 1, 2
);

-- Item category level transaction & refund stats
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_item_category_level_transaction_refund_data`
AS (
  SELECT
    TransLookup.transaction_date,
    IFNULL(NormalizeString(Items.item_category), 'unknown') AS item_category,
    SUM(Items.item_revenue) AS past_item_revenue,
    SUM(Items.item_refund) AS past_item_refund,
    COUNTIF(IFNULL(Items.item_revenue, 0) > 0) AS past_transaction_count,
    COUNTIF(IFNULL(Items.item_refund, 0) > 0) AS past_refund_count
  FROM `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_date_lookup` AS TransLookup
  LEFT JOIN `{ga4_project_id}.{ga4_dataset_id}.events_*` AS Ga4Table
    ON TransLookup.transaction_date > PARSE_DATE('%Y%m%d', Ga4Table.event_date)
  CROSS JOIN UNNEST(items) AS Items
  GROUP BY 1, 2
);

-- Traffic source level past transaction & refund stats
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_traffic_source_level_past_transaction_refund_data`
AS (
  SELECT
    TransLookup.transaction_date AS transaction_date,
    (SELECT value.string_value FROM UNNEST(Ga4Table.event_params) WHERE key = 'source') AS source,
    IFNULL(SUM(Ga4Table.ecommerce.purchase_revenue), 0) AS past_purchase_revenue,
    IFNULL(SUM(Ga4Table.ecommerce.refund_value), 0) AS past_refund_value,
    COUNTIF(IFNULL(Ga4Table.ecommerce.purchase_revenue, 0) > 0) AS past_trans_count,
    COUNTIF(IFNULL(Ga4Table.ecommerce.refund_value, 0) > 0) AS past_refund_count
  FROM `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_date_lookup` AS TransLookup
  LEFT JOIN `{ga4_project_id}.{ga4_dataset_id}.events_*` AS Ga4Table
    ON TransLookup.transaction_date > PARSE_DATE('%Y%m%d', Ga4Table.event_date)
  WHERE
    (SELECT value.string_value FROM UNNEST(Ga4Table.event_params) WHERE key = 'source') IS NOT NULL
  GROUP BY 1, 2
);

-- Traffic medium level past transaction & refund stats
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_traffic_medium_level_past_transaction_refund_data`
AS (
  SELECT
    TransLookup.transaction_date AS transaction_date,
    (SELECT value.string_value FROM UNNEST(Ga4Table.event_params) WHERE key = 'medium') AS medium,
    IFNULL(SUM(Ga4Table.ecommerce.purchase_revenue), 0) AS past_purchase_revenue,
    IFNULL(SUM(Ga4Table.ecommerce.refund_value), 0) AS past_refund_value,
    COUNTIF(IFNULL(Ga4Table.ecommerce.purchase_revenue, 0) > 0) AS past_trans_count,
    COUNTIF(IFNULL(Ga4Table.ecommerce.refund_value, 0) > 0) AS past_refund_count
  FROM `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_date_lookup` AS TransLookup
  LEFT JOIN `{ga4_project_id}.{ga4_dataset_id}.events_*` AS Ga4Table
    ON TransLookup.transaction_date > PARSE_DATE('%Y%m%d', Ga4Table.event_date)
  WHERE
    (SELECT value.string_value FROM UNNEST(Ga4Table.event_params) WHERE key = 'medium') IS NOT NULL
  GROUP BY 1, 2
);

-- Traffic campaign level past transaction & refund stats
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_traffic_campaign_level_past_transaction_refund_data`
AS (
  SELECT
    TransLookup.transaction_date AS transaction_date,
    (SELECT value.string_value FROM UNNEST(Ga4Table.event_params) WHERE key = 'campaign')
      AS campaign,
    IFNULL(SUM(Ga4Table.ecommerce.purchase_revenue), 0) AS past_purchase_revenue,
    IFNULL(SUM(Ga4Table.ecommerce.refund_value), 0) AS past_refund_value,
    COUNTIF(IFNULL(Ga4Table.ecommerce.purchase_revenue, 0) > 0) AS past_trans_count,
    COUNTIF(IFNULL(Ga4Table.ecommerce.refund_value, 0) > 0) AS past_refund_count
  FROM `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_date_lookup` AS TransLookup
  LEFT JOIN `{ga4_project_id}.{ga4_dataset_id}.events_*` AS Ga4Table
    ON TransLookup.transaction_date > PARSE_DATE('%Y%m%d', Ga4Table.event_date)
  WHERE
    (SELECT value.string_value FROM UNNEST(Ga4Table.event_params) WHERE key = 'campaign')
    IS NOT NULL
  GROUP BY 1, 2
);
