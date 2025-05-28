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
-- Step 3: Create staging tables for target variables, in-transaction item descriptions, website traffic, website events, and past refund history.

-- @param project_id STRING Project id on Google Cloud Platform for intermediate tables and final output tables.
-- @param dataset_id STRING Dataset id of intermediate tables and final output tables.
-- @param data_pipeline_type STRING Whether to process data for training or for prediction.

-- Set data_pipeline_type to determine whether to process data for training or for prediction
DECLARE data_pipeline_type STRING DEFAULT '{data_pipeline_type}';

-- Returns 'unknown' if any of special values is set, the input otherwise.
CREATE TEMP FUNCTION CleanUpStringVariable(input STRING)
RETURNS STRING
AS (
  IF(input IN ('(none)', '(not set)', '(data deleted)', NULL), 'unknown', input)
);

-- Calculates refund rate with past refund history.
CREATE TEMP AGGREGATE FUNCTION RefundRate(
  refund_count FLOAT64,
  avg_refund_rate FLOAT64,
  transaction_count FLOAT64)
RETURNS FLOAT64
AS (
  SAFE_DIVIDE((SUM(refund_count) + AVG(avg_refund_rate) * 0.2), (SUM(transaction_count) + 0.2))
);

CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_item_description_staging_table`
AS (
  WITH
    SameItemInTransactionSummary AS (
      SELECT
        transaction_id,
        transaction_date,
        MAX(same_item_item_quantity) AS max_same_item_item_quantity,
        MIN(same_item_item_quantity) AS min_same_item_item_quantity,
        AVG(same_item_item_quantity) AS avg_same_item_item_quantity,
        IF(MAX(same_item_item_quantity) > 1, 1, 0) AS same_item_in_transaction_flag,
      FROM
        (
          SELECT
            transaction_id,
            transaction_date,
            item_name,
            SUM(quantity) AS same_item_item_quantity
          FROM `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_transaction_id_and_item_level_stats`
          GROUP BY 1, 2, 3
        ) AS TransItemLevelCount
      GROUP BY 1, 2
    ),
    SameItemVariantInTransactionSummary AS (
      SELECT
        transaction_id,
        transaction_date,
        MAX(same_item_variant_quantity) AS max_same_item_variant_quantity,
        MIN(same_item_variant_quantity) AS min_same_item_variant_quantity,
        AVG(same_item_variant_quantity) AS avg_same_item_variant_quantity,
        IF(MAX(same_item_variant_quantity) > 1, 1, 0) AS same_item_variant_in_transaction_flag
      FROM
        (
          SELECT
            transaction_id,
            transaction_date,
            item_name,
            item_variant,
            SUM(quantity) AS same_item_variant_quantity
          FROM `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_transaction_id_and_item_level_stats`
          GROUP BY ALL
        ) AS trans_item_item_variant_level_count
      GROUP BY 1, 2
    ),
    PastProductRefundHistory AS (
      SELECT
        item_name,
        transaction_date,
        COUNTIF(item_refund > 0) AS refund_count,
        COUNT(*) AS trans_count,
        SUM(item_refund) AS total_item_refund,
        SUM(item_revenue) AS total_item_revenue
      FROM `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_product_level_transaction_refund_data`
      WHERE item_revenue > 0
      GROUP BY 1, 2
    ),
    PastItemRefundHistoryWithinCurrentTransaction AS (
      SELECT
        TransItemData.transaction_id,
        TransItemData.transaction_date,
        TransItemData.item_id,
        SUM(PastProductRefundHistory.refund_count) AS past_product_refund_count,
        SUM(PastProductRefundHistory.trans_count) AS past_product_transaction_count,
        SUM(PastProductRefundHistory.total_item_refund) AS past_product_refund_amt,
        SUM(PastProductRefundHistory.total_item_revenue) AS past_product_revenue_amt
      FROM
        `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_transaction_id_and_item_level_stats`
          AS TransItemData
      LEFT JOIN PastProductRefundHistory
        ON
          TransItemData.item_name = PastProductRefundHistory.item_name
          AND TransItemData.transaction_date > PastProductRefundHistory.transaction_date
      GROUP BY 1, 2, 3
    )
  SELECT
    TransLevelItemLevelStats.transaction_id,
    TransLevelItemLevelStats.transaction_date,
    COUNT(*) AS item_count,
    COUNT(DISTINCT TransLevelItemLevelStats.item_id) AS unique_item_id_count,
    COUNT(DISTINCT item_name) AS unique_item_name_count,
    COUNT(DISTINCT item_category) AS unique_item_category_count,
    COUNT(DISTINCT item_brand) AS unique_item_brand_count,
    STRING_AGG(item_brand) AS item_brands,
    STRING_AGG(item_name) AS item_names,
    STRING_AGG(item_variant) AS item_variants,
    STRING_AGG(item_category) AS item_categories,
    MIN(price) AS min_price,
    MAX(price) AS max_price,
    AVG(price) AS avg_price,
    IFNULL(stddev_samp(price), 0) AS std_price,
    MIN(item_revenue) AS min_item_revenue,
    MAX(item_revenue) AS max_item_revenue,
    AVG(item_revenue) AS avg_item_revenue,
    IFNULL(stddev_samp(item_revenue), 0) AS std_item_revenue,
    MIN(quantity) AS min_quantity,
    MAX(quantity) AS max_quantity,
    AVG(quantity) AS avg_quantity,
    IFNULL(stddev_samp(quantity), 0) AS std_quantity,
    SUM(CAST(discounted_product_flag AS INT)) AS discounted_product_count,
    RefundRate(
      PastItemRefundHistoryWithinCurrentTransaction.past_product_refund_count,
      PastGeneralRefundHistory.past_grand_avg_refund_rate,
      PastItemRefundHistoryWithinCurrentTransaction.past_product_transaction_count)
      AS calculated_past_product_refund_rate,
    RefundRate(
      PastItemRefundHistoryWithinCurrentTransaction.past_product_refund_amt,
      PastGeneralRefundHistory.past_grand_avg_refund_amt_proportion,
      PastItemRefundHistoryWithinCurrentTransaction.past_product_revenue_amt)
      AS calculated_past_product_refund_amt_proportion,
    AVG(SameItemInTrans.max_same_item_item_quantity) AS max_same_item_item_quantity,
    AVG(SameItemInTrans.min_same_item_item_quantity) AS min_same_item_item_quantity,
    AVG(SameItemInTrans.avg_same_item_item_quantity) AS avg_same_item_item_quantity,
    AVG(SameItemInTrans.same_item_in_transaction_flag) AS same_item_in_transaction_flag,
    AVG(max_same_item_variant_quantity) AS max_same_item_variant_quantity,
    AVG(min_same_item_variant_quantity) AS min_same_item_variant_quantity,
    AVG(avg_same_item_variant_quantity) AS avg_same_item_variant_quantity,
    AVG(same_item_variant_in_transaction_flag) AS same_item_variant_in_transaction_flag
  FROM
    `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_transaction_id_and_item_level_stats`
      AS TransLevelItemLevelStats
  LEFT JOIN PastItemRefundHistoryWithinCurrentTransaction
    ON
      TransLevelItemLevelStats.transaction_id
        = PastItemRefundHistoryWithinCurrentTransaction.transaction_id
      AND TransLevelItemLevelStats.transaction_date
        = PastItemRefundHistoryWithinCurrentTransaction.transaction_date
      AND TransLevelItemLevelStats.item_id = PastItemRefundHistoryWithinCurrentTransaction.item_id
  LEFT JOIN
    `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_daily_level_past_transaction_refund_data`
      AS PastGeneralRefundHistory
    ON TransLevelItemLevelStats.transaction_date = PastGeneralRefundHistory.transaction_date
  LEFT JOIN SameItemInTransactionSummary AS SameItemInTrans
    ON
      TransLevelItemLevelStats.transaction_id = SameItemInTrans.transaction_id
      AND TransLevelItemLevelStats.transaction_date
        = SameItemInTrans.transaction_date
  LEFT JOIN SameItemVariantInTransactionSummary
    ON
      TransLevelItemLevelStats.transaction_id
        = SameItemVariantInTransactionSummary.transaction_id
      AND TransLevelItemLevelStats.transaction_date
        = SameItemVariantInTransactionSummary.transaction_date
  GROUP BY 1, 2
);

-- Create staging table for transaction target variable.
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_target_variable_staging_table`
AS (
  SELECT
    TransData.transaction_id,
    TransData.transaction_date,
    SUM(TransData.purchase_revenue) AS purchase_revenue,
    SUM(TransData.refund_value) AS refund_value,
    IF(SUM(TransData.refund_value) > 0, 1, 0) AS refund_flag,
    SAFE_DIVIDE(SUM(TransData.refund_value), SUM(TransData.purchase_revenue))
      AS refund_proportion,
    AVG(TransData.shipping_value) AS shipping_value,
  FROM `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_transaction_data` AS TransData
  GROUP BY 1, 2
);

-- Create staging table for website session information.
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_website_session_information_staging_table`
AS (
  SELECT
    TransIDToSessionId.transaction_date,
    TransIDToSessionId.transaction_id,
    SUM(IFNULL(WebTraffic.session_length_in_seconds, 0)) AS session_length_in_seconds,
    SUM(IFNULL(WebTraffic.engaged_sessions, 0)) AS engaged_sessions,
    SUM(IFNULL(WebTraffic.engagement_time_seconds, 0)) AS engagement_time_seconds,
    IFNULL(ANY_VALUE(CleanUpStringVariable(WebTraffic.campaign)), 'unknown') AS campaign,
    IFNULL(ANY_VALUE(CleanUpStringVariable(WebTraffic.medium)), 'unknown') AS medium,
    IFNULL(ANY_VALUE(CleanUpStringVariable(WebTraffic.source)), 'unknown') AS source,
    IFNULL(ANY_VALUE(WebTraffic.channel_grouping_user), 'unknown') AS channel_grouping_user
  FROM `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_id_to_session_id_mapping` AS TransIDToSessionId
  LEFT JOIN `{project_id}.{dataset_id}.{data_pipeline_type}_session_id_level_web_traffic_info` AS WebTraffic
    ON TransIDToSessionId.session_id = WebTraffic.session_id
  GROUP BY 1, 2
);

-- Create staging table for website event information.
IF data_pipeline_type = 'TRAINING' THEN
  CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_website_event_staging_table`
  AS (
    SELECT
      TransactionToSessionMapping.transaction_id,
      TransactionToSessionMapping.transaction_date,
      EventCountData.event_name,
      SUM(EventCountData.event_length_in_seconds) AS event_length_in_seconds,
      SUM(EventCountData.event_count) AS event_count
    FROM
      `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_id_to_session_id_mapping` AS TransactionToSessionMapping
    LEFT JOIN `{project_id}.{dataset_id}.{data_pipeline_type}_session_id_level_web_event_info` AS EventCountData
      ON TransactionToSessionMapping.session_id = EventCountData.session_id
    GROUP BY 1, 2, 3
  );
ELSE
  CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_website_event_staging_table`
  AS (
    WITH TestData
    AS (
      SELECT
        TransactionToSessionMapping.transaction_id,
        TransactionToSessionMapping.transaction_date,
        EventCountData.event_name,
        SUM(EventCountData.event_length_in_seconds) AS event_length_in_seconds,
        SUM(EventCountData.event_count) AS event_count
      FROM
        `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_id_to_session_id_mapping` AS TransactionToSessionMapping
      LEFT JOIN `{project_id}.{dataset_id}.{data_pipeline_type}_session_id_level_web_event_info` AS EventCountData
        ON TransactionToSessionMapping.session_id = EventCountData.session_id
      GROUP BY 1, 2, 3
    ),
    TransactionEventCombination AS (
      SELECT DISTINCT
        TestData.transaction_id,
        TestData.transaction_date,
        training_event_name.event_name
      FROM TestData
      CROSS JOIN
        (SELECT DISTINCT event_name
          FROM
          `{project_id}.{dataset_id}.TRAINING_website_event_staging_table`) AS training_event_name)
    SELECT
      TransactionEventCombination.transaction_id,
      TransactionEventCombination.transaction_date,
      TransactionEventCombination.event_name,
      IFNULL(TestData.event_length_in_seconds, 0) AS event_length_in_seconds,
      IFNULL(TestData.event_count, 0) AS event_count
      FROM TransactionEventCombination
      LEFT JOIN TestData
      ON TestData.transaction_id = TransactionEventCombination.transaction_id
      AND TestData.transaction_date = TransactionEventCombination.transaction_date
      AND TestData.event_name = TransactionEventCombination.event_name
      );
END IF;

-- Create staging table for past refund history at pseudo user level.
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_pseudo_user_past_refund_history_staging_table`
AS (
  WITH
    PastCustomerRefundHistory AS (
      SELECT
        TransToPseudoUserId.transaction_date,
        TransToPseudoUserId.user_pseudo_id,
        SUM(IFNULL(PurchaseRefundHistoryData.refund_value, 0)) AS past_user_refund_amt,
        SUM(IFNULL(PurchaseRefundHistoryData.purchase_revenue, 0)) AS past_user_purchase_amt,
        COUNTIF(IFNULL(PurchaseRefundHistoryData.refund_value, 0) > 0)
          AS past_user_refund_count,
        COUNT(*) AS past_user_transaction_count
      FROM
        `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_id_to_pseudo_user_id_mapping` AS TransToPseudoUserId
      LEFT JOIN
        `{project_id}.{dataset_id}.{data_pipeline_type}_customer_purchase_refund_history_data_user_pseudo_id_level`
          AS PurchaseRefundHistoryData
        ON
          TransToPseudoUserId.user_pseudo_id = PurchaseRefundHistoryData.user_pseudo_id
          AND TransToPseudoUserId.transaction_date > PurchaseRefundHistoryData.transaction_date
      GROUP BY 1, 2
    )
  SELECT
    TransToPseudoUserId.transaction_date,
    TransToPseudoUserId.transaction_id,
    SUM(PastCustomerRefundHistory.past_user_purchase_amt) AS past_user_purchase_revenue,
    SUM(PastCustomerRefundHistory.past_user_refund_amt) AS past_user_refund_value,
    SUM(PastCustomerRefundHistory.past_user_refund_count) AS past_user_refund_count,
    SUM(PastCustomerRefundHistory.past_user_transaction_count) AS past_user_transaction_count,
    RefundRate(
      PastCustomerRefundHistory.past_user_refund_count,
      PastGeneralRefundHistory.past_grand_avg_refund_rate,
      PastCustomerRefundHistory.past_user_transaction_count) AS calculated_past_user_refund_rate,
    RefundRate(
      PastCustomerRefundHistory.past_user_refund_amt,
      PastGeneralRefundHistory.past_grand_avg_refund_amt_proportion,
      PastCustomerRefundHistory.past_user_purchase_amt)
      AS calculated_past_user_refund_amt_proportion
  FROM `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_id_to_pseudo_user_id_mapping` AS TransToPseudoUserId
  LEFT JOIN PastCustomerRefundHistory
    ON
      TransToPseudoUserId.user_pseudo_id = PastCustomerRefundHistory.user_pseudo_id
      AND TransToPseudoUserId.transaction_date = PastCustomerRefundHistory.transaction_date
  LEFT JOIN
    `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_daily_level_past_transaction_refund_data`
      AS PastGeneralRefundHistory
    ON PastGeneralRefundHistory.transaction_date = TransToPseudoUserId.transaction_date
  GROUP BY 1, 2
);

-- Create staging table for past refund history at user level.
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_user_past_refund_history_staging_table`
AS (
  WITH
    PastCustomerRefundHistory AS (
      SELECT
        TransToUserId.transaction_date,
        TransToUserId.user_id,
        SUM(PurchaseRefundHistoryData.refund_value) AS past_user_refund_amt,
        SUM(IFNULL(PurchaseRefundHistoryData.purchase_revenue, 0)) AS past_user_purchase_amt,
        COUNTIF(IFNULL(PurchaseRefundHistoryData.refund_value, 0) > 0)
          AS past_user_refund_count,
        COUNT(*) AS past_user_transaction_count
      FROM `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_id_to_user_id_mapping` AS TransToUserId
      LEFT JOIN
        `{project_id}.{dataset_id}.{data_pipeline_type}_customer_purchase_refund_history_data_user_id_level`
          AS PurchaseRefundHistoryData
        ON
          TransToUserId.user_id = PurchaseRefundHistoryData.user_id
          AND TransToUserId.transaction_date > PurchaseRefundHistoryData.transaction_date
      GROUP BY 1, 2
    )
  SELECT
    TransToUserId.transaction_date,
    TransToUserId.transaction_id,
    SUM(PastCustomerRefundHistory.past_user_purchase_amt) AS past_user_purchase_revenue,
    SUM(PastCustomerRefundHistory.past_user_refund_amt) AS past_user_refund_value,
    SUM(PastCustomerRefundHistory.past_user_refund_count) AS past_user_refund_count,
    SUM(PastCustomerRefundHistory.past_user_transaction_count) AS past_user_transaction_count,
    RefundRate(
      PastCustomerRefundHistory.past_user_refund_count,
      PastGeneralRefundHistory.past_grand_avg_refund_rate,
      PastCustomerRefundHistory.past_user_transaction_count) AS calculated_past_user_refund_rate,
    RefundRate(
      PastCustomerRefundHistory.past_user_refund_amt,
      PastGeneralRefundHistory.past_grand_avg_refund_amt_proportion,
      PastCustomerRefundHistory.past_user_purchase_amt)
      AS calculated_past_user_refund_amt_proportion
  FROM `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_id_to_user_id_mapping` AS TransToUserId
  LEFT JOIN PastCustomerRefundHistory
    ON
      TransToUserId.user_id = PastCustomerRefundHistory.user_id
      AND TransToUserId.transaction_date = PastCustomerRefundHistory.transaction_date
  LEFT JOIN
    `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_daily_level_past_transaction_refund_data`
      AS PastGeneralRefundHistory
    ON PastGeneralRefundHistory.transaction_date = TransToUserId.transaction_date
  GROUP BY 1, 2
);

-- Create staging table for past refund history at pseudo user & item level.
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_pseudo_user_item_past_refund_history_staging_table`
AS (
  WITH
    PastCustomerItemRefundHistory AS (
      SELECT
        TransToPseudoUserIdItem.transaction_date,
        TransToPseudoUserIdItem.user_pseudo_id,
        TransToPseudoUserIdItem.item_name,
        SUM(PurchaseRefundHistoryData.item_refund) AS past_refund_amt,
        SUM(PurchaseRefundHistoryData.item_revenue) AS past_purchase_amt,
        COUNTIF(IFNULL(PurchaseRefundHistoryData.item_refund, 0) > 0)
          AS past_refund_count,
        COUNT(*) AS past_transaction_count
      FROM
        `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_id_to_pseudo_user_id_item_mapping`
          AS TransToPseudoUserIdItem
      LEFT JOIN
        `{project_id}.{dataset_id}.{data_pipeline_type}_customer_product_purchase_refund_interaction_history_data_user_pseudo_id_level`
          AS PurchaseRefundHistoryData
        ON
          TransToPseudoUserIdItem.user_pseudo_id = PurchaseRefundHistoryData.user_pseudo_id
          AND TransToPseudoUserIdItem.transaction_date > PurchaseRefundHistoryData.transaction_date
          AND TransToPseudoUserIdItem.item_name = PurchaseRefundHistoryData.item_name
      GROUP BY 1, 2, 3
    )
  SELECT
    TransToPseudoUserIdItem.transaction_date,
    TransToPseudoUserIdItem.transaction_id,
    SUM(PastCustomerItemRefundHistory.past_purchase_amt) AS past_purchase_amt,
    SUM(PastCustomerItemRefundHistory.past_refund_amt) AS past_refund_amt,
    SUM(PastCustomerItemRefundHistory.past_refund_count) AS past_refund_count,
    SUM(PastCustomerItemRefundHistory.past_transaction_count) AS past_transaction_count,
    RefundRate(
      PastCustomerItemRefundHistory.past_refund_count,
      PastGeneralRefundHistory.past_grand_avg_refund_rate,
      PastCustomerItemRefundHistory.past_transaction_count)
      AS calculated_past_user_item_refund_rate,
    RefundRate(
      PastCustomerItemRefundHistory.past_refund_amt,
      PastGeneralRefundHistory.past_grand_avg_refund_amt_proportion,
      PastCustomerItemRefundHistory.past_purchase_amt)
      AS calculated_past_user_item_refund_amt_proportion
  FROM
    `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_id_to_pseudo_user_id_item_mapping`
      AS TransToPseudoUserIdItem
  LEFT JOIN PastCustomerItemRefundHistory
    ON
      TransToPseudoUserIdItem.user_pseudo_id = PastCustomerItemRefundHistory.user_pseudo_id
      AND TransToPseudoUserIdItem.transaction_date = PastCustomerItemRefundHistory.transaction_date
      AND TransToPseudoUserIdItem.item_name = PastCustomerItemRefundHistory.item_name
  LEFT JOIN
    `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_daily_level_past_transaction_refund_data`
      AS PastGeneralRefundHistory
    ON PastGeneralRefundHistory.transaction_date = TransToPseudoUserIdItem.transaction_date
  GROUP BY 1, 2
);

-- Create staging table for past refund history at user & item level.
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_user_item_past_refund_history_staging_table`
AS (
  WITH
    PastCustomerItemRefundHistory AS (
      SELECT
        TransToUserIdItem.transaction_date,
        TransToUserIdItem.user_id,
        TransToUserIdItem.item_name,
        SUM(PurchaseRefundHistoryData.item_refund) AS past_refund_amt,
        SUM(PurchaseRefundHistoryData.item_revenue) AS past_purchase_amt,
        COUNTIF(IFNULL(PurchaseRefundHistoryData.item_refund, 0) > 0)
          AS past_refund_count,
        COUNT(*) AS past_transaction_count
      FROM `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_id_to_user_id_item_mapping` AS TransToUserIdItem
      LEFT JOIN
        `{project_id}.{dataset_id}.{data_pipeline_type}_customer_product_purchase_refund_interaction_history_data_user_id_level`
          AS PurchaseRefundHistoryData
        ON
          TransToUserIdItem.user_id = PurchaseRefundHistoryData.user_id
          AND TransToUserIdItem.transaction_date > PurchaseRefundHistoryData.transaction_date
          AND TransToUserIdItem.item_name = PurchaseRefundHistoryData.item_name
      GROUP BY 1, 2, 3
    )
  SELECT
    TransToUserIdItem.transaction_date,
    TransToUserIdItem.transaction_id,
    SUM(PastCustomerItemRefundHistory.past_purchase_amt) AS past_purchase_amt,
    SUM(PastCustomerItemRefundHistory.past_refund_amt) AS past_refund_amt,
    SUM(PastCustomerItemRefundHistory.past_refund_count) AS past_refund_count,
    SUM(PastCustomerItemRefundHistory.past_transaction_count) AS past_transaction_count,
    RefundRate(
      PastCustomerItemRefundHistory.past_refund_count,
      PastGeneralRefundHistory.past_grand_avg_refund_rate,
      PastCustomerItemRefundHistory.past_transaction_count)
      AS calculated_past_user_item_refund_rate,
    RefundRate(
      PastCustomerItemRefundHistory.past_refund_amt,
      PastGeneralRefundHistory.past_grand_avg_refund_amt_proportion,
      PastCustomerItemRefundHistory.past_purchase_amt)
      AS calculated_past_user_item_refund_amt_proportion
  FROM `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_id_to_user_id_item_mapping` AS TransToUserIdItem
  LEFT JOIN PastCustomerItemRefundHistory
    ON
      TransToUserIdItem.user_id = PastCustomerItemRefundHistory.user_id
      AND TransToUserIdItem.transaction_date = PastCustomerItemRefundHistory.transaction_date
      AND TransToUserIdItem.item_name = PastCustomerItemRefundHistory.item_name
  LEFT JOIN
    `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_daily_level_past_transaction_refund_data`
      AS PastGeneralRefundHistory
    ON PastGeneralRefundHistory.transaction_date = TransToUserIdItem.transaction_date
  GROUP BY 1, 2
);

-- Create recent transaction & refund data (used as seasonality features)
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_rolling_window_daily_refund_data`
AS (
  SELECT
    transaction_date,
    past_3_days_transaction_value,
    past_3_days_refund_value,
    IFNULL(SAFE_DIVIDE(past_3_days_refund_value, past_3_days_transaction_value), 0)
      AS past_3_days_refund_proportion,
    past_7_days_transaction_value,
    past_7_days_refund_value,
    IFNULL(SAFE_DIVIDE(past_7_days_refund_value, past_7_days_transaction_value), 0)
      AS past_7_days_refund_proportion,
    past_30_days_transaction_value,
    past_30_days_refund_value,
    IFNULL(SAFE_DIVIDE(past_30_days_refund_value, past_30_days_transaction_value), 0)
      AS past_30_days_refund_proportion
  FROM
    (
      SELECT
        transaction_date,
        SUM(purchase_revenue)
          OVER (ORDER BY transaction_date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING)
          AS past_3_days_transaction_value,
        SUM(refund_value)
          OVER (ORDER BY transaction_date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING)
          AS past_3_days_refund_value,
        SUM(purchase_revenue)
          OVER (ORDER BY transaction_date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING)
          AS past_7_days_transaction_value,
        SUM(refund_value)
          OVER (ORDER BY transaction_date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING)
          AS past_7_days_refund_value,
        SUM(purchase_revenue)
          OVER (ORDER BY transaction_date ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING)
          AS past_30_days_transaction_value,
        SUM(refund_value)
          OVER (ORDER BY transaction_date ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING)
          AS past_30_days_refund_value
      FROM `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_daily_level_transaction_refund_data`
      ORDER BY transaction_date
    ) AS RollingWindowData
);

-- Create staging table for customer demographic attributes at pseudo user id level.
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_customer_demographic_attributes_staging_data`
AS (
  WITH
    PseudoUserIdLevelDemoData AS (
      SELECT
        TransToPseudoUserId.transaction_date,
        TransToPseudoUserId.transaction_id,
        CustomerDemographicAttributesPseudoUserId.country,
        CustomerDemographicAttributesPseudoUserId.city,
        CustomerDemographicAttributesPseudoUserId.device_category,
        CustomerDemographicAttributesPseudoUserId.operating_system,
        CustomerDemographicAttributesPseudoUserId.language
      FROM
        `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_id_to_pseudo_user_id_mapping` AS TransToPseudoUserId
      LEFT JOIN
        `{project_id}.{dataset_id}.{data_pipeline_type}_customer_demographic_attributes_data_pseudo_user_id_level`
          AS CustomerDemographicAttributesPseudoUserId
        ON
          TransToPseudoUserId.user_pseudo_id
          = CustomerDemographicAttributesPseudoUserId.user_pseudo_id
    ),
    UserIdLevelDemoData AS (
      SELECT
        TransToUserId.transaction_date,
        TransToUserId.transaction_id,
        CustomerDemographicAttributesUserId.country,
        CustomerDemographicAttributesUserId.city,
        CustomerDemographicAttributesUserId.device_category,
        CustomerDemographicAttributesUserId.operating_system,
        CustomerDemographicAttributesUserId.language
      FROM `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_id_to_user_id_mapping` AS TransToUserId
      LEFT JOIN
        `{project_id}.{dataset_id}.{data_pipeline_type}_customer_demographic_attributes_data_user_id_level`
          AS CustomerDemographicAttributesUserId
        ON TransToUserId.user_id = CustomerDemographicAttributesUserId.user_id
    ),
    TransLevelDemoData AS (
      SELECT
        TransLookup.transaction_date,
        TransLookup.transaction_id,
        COALESCE(UserIdLevelDemoData.country, PseudoUserIdLevelDemoData.country, 'unknown')
          AS country,
        COALESCE(UserIdLevelDemoData.city, PseudoUserIdLevelDemoData.city, 'unknown') AS city,
        COALESCE(
          UserIdLevelDemoData.device_category, PseudoUserIdLevelDemoData.device_category, 'unknown')
          AS device_category,
        COALESCE(
          UserIdLevelDemoData.operating_system,
          PseudoUserIdLevelDemoData.operating_system,
          'unknown')
          AS operating_system,
        COALESCE(UserIdLevelDemoData.language, PseudoUserIdLevelDemoData.language, 'unknown')
          AS language
      FROM `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_lookup` AS TransLookup
      LEFT JOIN PseudoUserIdLevelDemoData
        ON
          TransLookup.transaction_date = PseudoUserIdLevelDemoData.transaction_date
          AND TransLookup.transaction_id = PseudoUserIdLevelDemoData.transaction_id
      LEFT JOIN UserIdLevelDemoData
        ON
          TransLookup.transaction_date = UserIdLevelDemoData.transaction_date
          AND TransLookup.transaction_id = UserIdLevelDemoData.transaction_id
    )
  SELECT
    TransLevelDemoData.transaction_date,
    TransLevelDemoData.transaction_id,
    IFNULL(
      RefundRate(
        PastCountryLevelTransRefundData.past_refund_count,
        PastGeneralRefundHistory.past_grand_avg_refund_rate,
        PastCountryLevelTransRefundData.past_trans_count),
      0) AS calculated_past_country_refund_rate,
    IFNULL(
      RefundRate(
        PastCountryLevelTransRefundData.past_refund_value,
        PastGeneralRefundHistory.past_grand_avg_refund_amt_proportion,
        PastCountryLevelTransRefundData.past_purchase_revenue),
      0)
      AS calculated_past_country_refund_amt_proportion,
    IFNULL(
      RefundRate(
        PastCityLevelTransRefundData.past_refund_count,
        PastGeneralRefundHistory.past_grand_avg_refund_rate,
        PastCityLevelTransRefundData.past_trans_count),
      0) AS calculated_past_city_refund_rate,
    IFNULL(
      RefundRate(
        PastCityLevelTransRefundData.past_refund_value,
        PastGeneralRefundHistory.past_grand_avg_refund_amt_proportion,
        PastCityLevelTransRefundData.past_purchase_revenue),
      0)
      AS calculated_past_city_refund_amt_proportion,
    IFNULL(
      RefundRate(
        PastDeviceCategoryLevelTransRefundData.past_refund_count,
        PastGeneralRefundHistory.past_grand_avg_refund_rate,
        PastDeviceCategoryLevelTransRefundData.past_trans_count),
      0)
      AS calculated_past_device_category_refund_rate,
    IFNULL(
      RefundRate(
        PastDeviceCategoryLevelTransRefundData.past_refund_value,
        PastGeneralRefundHistory.past_grand_avg_refund_amt_proportion,
        PastDeviceCategoryLevelTransRefundData.past_purchase_revenue),
      0)
      AS calculated_past_device_category_refund_amt_proportion,
    IFNULL(
      RefundRate(
        PastOperatingSystemLevelTransRefundData.past_refund_count,
        PastGeneralRefundHistory.past_grand_avg_refund_rate,
        PastOperatingSystemLevelTransRefundData.past_trans_count),
      0)
      AS calculated_past_device_operating_system_refund_rate,
    IFNULL(
      RefundRate(
        PastOperatingSystemLevelTransRefundData.past_refund_value,
        PastGeneralRefundHistory.past_grand_avg_refund_amt_proportion,
        PastOperatingSystemLevelTransRefundData.past_purchase_revenue),
      0)
      AS calculated_past_device_operating_system_refund_amt_proportion,
    IFNULL(
      RefundRate(
        PastLanguageLevelTransRefundData.past_refund_count,
        PastGeneralRefundHistory.past_grand_avg_refund_rate,
        PastLanguageLevelTransRefundData.past_trans_count),
      0)
      AS calculated_past_device_language_refund_rate,
    IFNULL(
      RefundRate(
        PastLanguageLevelTransRefundData.past_refund_value,
        PastGeneralRefundHistory.past_grand_avg_refund_amt_proportion,
        PastLanguageLevelTransRefundData.past_purchase_revenue),
      0)
      AS calculated_past_device_language_refund_amt_proportion
  FROM TransLevelDemoData
  LEFT JOIN
    `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_country_level_transaction_refund_data`
      AS PastCountryLevelTransRefundData
    ON
      PastCountryLevelTransRefundData.country = TransLevelDemoData.country
      AND PastCountryLevelTransRefundData.transaction_date = TransLevelDemoData.transaction_date
  LEFT JOIN
    `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_city_level_transaction_refund_data`
      AS PastCityLevelTransRefundData
    ON
      PastCityLevelTransRefundData.city = TransLevelDemoData.city
      AND PastCityLevelTransRefundData.transaction_date = TransLevelDemoData.transaction_date
  LEFT JOIN
    `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_device_category_level_transaction_refund_data`
      AS PastDeviceCategoryLevelTransRefundData
    ON
      PastDeviceCategoryLevelTransRefundData.device_category = TransLevelDemoData.device_category
      AND PastDeviceCategoryLevelTransRefundData.transaction_date
        = TransLevelDemoData.transaction_date
  LEFT JOIN
    `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_device_operating_system_level_transaction_refund_data`
      AS PastOperatingSystemLevelTransRefundData
    ON
      PastOperatingSystemLevelTransRefundData.operating_system = TransLevelDemoData.operating_system
      AND PastOperatingSystemLevelTransRefundData.transaction_date
        = TransLevelDemoData.transaction_date
  LEFT JOIN
    `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_device_language_level_transaction_refund_data`
      AS PastLanguageLevelTransRefundData
    ON
      PastLanguageLevelTransRefundData.language = TransLevelDemoData.language
      AND PastLanguageLevelTransRefundData.transaction_date = TransLevelDemoData.transaction_date
  LEFT JOIN
    `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_daily_level_past_transaction_refund_data`
      AS PastGeneralRefundHistory
    ON PastGeneralRefundHistory.transaction_date = TransLevelDemoData.transaction_date
  GROUP BY ALL
);

-- Create staging table for product brand & category attributes past refund history.
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_product_brand_category_past_refund_history_staging_table`
AS (
  SELECT
    TransItemLevelAttributes.transaction_date,
    TransItemLevelAttributes.transaction_id,
    IFNULL(
      RefundRate(
        PastBrandRefundHistory.past_item_refund,
        PastGeneralRefundHistory.past_grand_avg_refund_amt_proportion,
        PastBrandRefundHistory.past_item_revenue),
      0) AS calculated_past_brand_refund_amt_proportion,
    IFNULL(
      RefundRate(
        PastBrandRefundHistory.past_refund_count,
        PastGeneralRefundHistory.past_grand_avg_refund_rate,
        PastBrandRefundHistory.past_transaction_count),
      0) AS calculated_past_brand_refund_rate,
    IFNULL(
      RefundRate(
        PastCategoryRefundHistory.past_item_refund,
        PastGeneralRefundHistory.past_grand_avg_refund_amt_proportion,
        PastCategoryRefundHistory.past_item_revenue),
      0) AS calculated_past_category_refund_amt_proportion,
    IFNULL(
      RefundRate(
        PastCategoryRefundHistory.past_refund_count,
        PastGeneralRefundHistory.past_grand_avg_refund_rate,
        PastCategoryRefundHistory.past_transaction_count),
      0) AS calculated_past_category_refund_rate
  FROM
    `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_transaction_id_and_item_level_stats`
      AS TransItemLevelAttributes
  LEFT JOIN
    `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_brand_level_transaction_refund_data`
      AS PastBrandRefundHistory
    ON
      PastBrandRefundHistory.item_brand = TransItemLevelAttributes.item_brand
      AND PastBrandRefundHistory.transaction_date = TransItemLevelAttributes.transaction_date
  LEFT JOIN
    `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_item_category_level_transaction_refund_data`
      AS PastCategoryRefundHistory
    ON
      PastCategoryRefundHistory.item_category = TransItemLevelAttributes.item_category
      AND PastCategoryRefundHistory.transaction_date = TransItemLevelAttributes.transaction_date
  LEFT JOIN
    `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_daily_level_past_transaction_refund_data`
      AS PastGeneralRefundHistory
    ON PastGeneralRefundHistory.transaction_date = TransItemLevelAttributes.transaction_date
  GROUP BY ALL
);

-- Create staging table for traffic source, campaign and level past refund history.
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_traffic_source_campaign_past_refund_history_staging_table`
AS (
  WITH
    WebTrafficData AS (
      SELECT
        TransIDToSessionId.transaction_date,
        TransIDToSessionId.transaction_id,
        IFNULL(ANY_VALUE(CleanUpStringVariable(WebTraffic.campaign)), 'unknown') AS campaign,
        IFNULL(ANY_VALUE(CleanUpStringVariable(WebTraffic.medium)), 'unknown') AS medium,
        IFNULL(ANY_VALUE(CleanUpStringVariable(WebTraffic.source)), 'unknown') AS source,
      FROM `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_id_to_session_id_mapping` AS TransIDToSessionId
      LEFT JOIN `{project_id}.{dataset_id}.{data_pipeline_type}_session_id_level_web_traffic_info` AS WebTraffic
        ON TransIDToSessionId.session_id = WebTraffic.session_id
      GROUP BY 1, 2
    )
  SELECT
    WebTrafficData.transaction_date,
    WebTrafficData.transaction_id,
    IFNULL(
      RefundRate(
        PastTrafficSourceRefundHistory.past_refund_value,
        PastGeneralRefundHistory.past_grand_avg_refund_amt_proportion,
        PastTrafficSourceRefundHistory.past_purchase_revenue),
      0) AS calculated_past_traffic_source_refund_amt_proportion,
    IFNULL(
      RefundRate(
        PastTrafficSourceRefundHistory.past_refund_count,
        PastGeneralRefundHistory.past_grand_avg_refund_rate,
        PastTrafficSourceRefundHistory.past_trans_count),
      0) AS calculated_past_traffic_source_refund_rate,
    IFNULL(
      RefundRate(
        PastTrafficMediumRefundHistory.past_refund_value,
        PastGeneralRefundHistory.past_grand_avg_refund_amt_proportion,
        PastTrafficMediumRefundHistory.past_purchase_revenue),
      0) AS calculated_past_traffic_medium_refund_amt_proportion,
    IFNULL(
      RefundRate(
        PastTrafficMediumRefundHistory.past_refund_count,
        PastGeneralRefundHistory.past_grand_avg_refund_rate,
        PastTrafficMediumRefundHistory.past_trans_count),
      0) AS calculated_past_traffic_medium_refund_rate,
    IFNULL(
      RefundRate(
        PastTrafficCampaignRefundHistory.past_refund_value,
        PastGeneralRefundHistory.past_grand_avg_refund_amt_proportion,
        PastTrafficCampaignRefundHistory.past_purchase_revenue),
      0) AS calculated_past_traffic_campaign_refund_amt_proportion,
    IFNULL(
      RefundRate(
        PastTrafficCampaignRefundHistory.past_refund_count,
        PastGeneralRefundHistory.past_grand_avg_refund_rate,
        PastTrafficCampaignRefundHistory.past_trans_count),
      0) AS calculated_past_traffic_campaign_refund_rate
  FROM WebTrafficData
  LEFT JOIN
    `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_traffic_source_level_past_transaction_refund_data`
      AS PastTrafficSourceRefundHistory
    ON
      PastTrafficSourceRefundHistory.source = WebTrafficData.source
      AND PastTrafficSourceRefundHistory.transaction_date = WebTrafficData.transaction_date
  LEFT JOIN
    `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_traffic_medium_level_past_transaction_refund_data`
      AS PastTrafficMediumRefundHistory
    ON
      PastTrafficMediumRefundHistory.medium = WebTrafficData.medium
      AND PastTrafficMediumRefundHistory.transaction_date = WebTrafficData.transaction_date
  LEFT JOIN
    `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_traffic_campaign_level_past_transaction_refund_data`
      AS PastTrafficCampaignRefundHistory
    ON
      PastTrafficCampaignRefundHistory.campaign = WebTrafficData.campaign
      AND PastTrafficCampaignRefundHistory.transaction_date = WebTrafficData.transaction_date
  LEFT JOIN
    `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_daily_level_past_transaction_refund_data`
      AS PastGeneralRefundHistory
    ON PastGeneralRefundHistory.transaction_date = WebTrafficData.transaction_date
  GROUP BY ALL
);
