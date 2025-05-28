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
-- Step 6: Combine all features from staging tables into one table.

-- @param project_id STRING Project id on Google Cloud Platform for intermediate tables and final output tables.
-- @param dataset_id STRING Dataset id of intermediate tables and final output tables.
-- @param data_pipeline_type STRING Whether to process data for training or for prediction.

-- Join all features and target variables together from staging tables.
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_ml_ready_data`
AS (
  SELECT
    TargetVariable.transaction_id,
    TargetVariable.transaction_date,
    TargetVariable.purchase_revenue AS transaction_value,
    TargetVariable.refund_value AS refund_value,
    TargetVariable.refund_flag,
    TargetVariable.refund_proportion,
    TargetVariable.shipping_value,
    IFNULL(ItemDescription.item_count, 0) AS item_count,
    IFNULL(ItemDescription.unique_item_id_count, 0) AS unique_item_id_count,
    IFNULL(ItemDescription.unique_item_name_count, 0) AS unique_item_name_count,
    IFNULL(ItemDescription.unique_item_category_count, 0) AS unique_item_category_count,
    IFNULL(ItemDescription.unique_item_brand_count, 0) AS unique_item_brand_count,
    #IFNULL(ItemDescription.item_brands, 'unknown') AS item_brands,
    #IFNULL(ItemDescription.item_names, 'unknown') AS item_names,
    #IFNULL(ItemDescription.item_variants, 'unknown') AS item_variants,
    #IFNULL(ItemDescription.item_categories, 'unknown') AS item_categories,
    IFNULL(ItemDescription.min_price, 0) AS min_price,
    IFNULL(ItemDescription.max_price, 0) AS max_price,
    IFNULL(ItemDescription.avg_price, 0) AS avg_price,
    IFNULL(ItemDescription.std_price, 0) AS std_price,
    IFNULL(ItemDescription.min_item_revenue, 0) AS min_item_revenue,
    IFNULL(ItemDescription.max_item_revenue, 0) AS max_item_revenue,
    IFNULL(ItemDescription.avg_item_revenue, 0) AS avg_item_revenue,
    IFNULL(ItemDescription.std_item_revenue, 0) AS std_item_revenue,
    IFNULL(ItemDescription.min_quantity, 0) AS min_item_quantity,
    IFNULL(ItemDescription.max_quantity, 0) AS max_item_quantity,
    IFNULL(ItemDescription.avg_quantity, 0) AS avg_item_quantity,
    IFNULL(ItemDescription.std_quantity, 0) AS std_item_quantity,
    IFNULL(ItemDescription.discounted_product_count, 0) AS discounted_product_count,
    IFNULL(ItemDescription.calculated_past_product_refund_rate, 0)
      AS calculated_past_product_refund_rate,
    IFNULL(ItemDescription.calculated_past_product_refund_amt_proportion, 0)
      AS calculated_past_product_refund_amt_proportion,
    IFNULL(ItemDescription.max_same_item_item_quantity, 0) AS max_same_item_item_quantity,
    IFNULL(ItemDescription.min_same_item_item_quantity, 0) AS min_same_item_item_quantity,
    IFNULL(ItemDescription.avg_same_item_item_quantity, 0) AS avg_same_item_item_quantity,
    IFNULL(ItemDescription.same_item_in_transaction_flag, 0) AS same_item_in_transaction_flag,
    IFNULL(ItemDescription.max_same_item_variant_quantity, 0) AS max_same_item_variant_quantity,
    IFNULL(ItemDescription.min_same_item_variant_quantity, 0) AS min_same_item_variant_quantity,
    IFNULL(ItemDescription.avg_same_item_variant_quantity, 0) AS avg_same_item_variant_quantity,
    IFNULL(ItemDescription.same_item_variant_in_transaction_flag, 0)
      AS same_item_variant_in_transaction_flag,
    IFNULL(session_length_in_seconds, 0) AS session_length_in_seconds,
    IFNULL(engaged_sessions, 0) AS website_engaged_sessions,
    IFNULL(engagement_time_seconds, 0) AS web_traffic_engagement_time_seconds,
    #IFNULL(campaign, 'unknown') AS web_traffic_campaign,
    #IFNULL(medium, 'unknown') AS web_traffic_medium,
    #IFNULL(source, 'unknown') AS web_traffic_source,
    IFNULL(channel_grouping_user, 'unknown') AS web_traffic_channel_grouping_user,
    TrafficSourceCampaignPastRefundHistory.* EXCEPT (transaction_id, transaction_date),
    COALESCE(
      UserPastRefund.past_user_purchase_revenue,
      PseudoUserPastRefund.past_user_purchase_revenue,
      0) AS past_user_purchase_revenue,
    COALESCE(
      UserPastRefund.past_user_refund_value,
      PseudoUserPastRefund.past_user_refund_value,
      0) AS past_user_refund_value,
    COALESCE(
      UserPastRefund.past_user_refund_count,
      PseudoUserPastRefund.past_user_refund_count,
      0) AS past_user_refund_count,
    COALESCE(
      UserPastRefund.past_user_transaction_count,
      PseudoUserPastRefund.past_user_purchase_revenue,
      0) AS past_user_transaction_count,
    COALESCE(
      UserPastRefund.calculated_past_user_refund_rate,
      PseudoUserPastRefund.calculated_past_user_refund_rate,
      0) AS calculated_past_user_refund_rate,
    COALESCE(
      UserPastRefund.calculated_past_user_refund_amt_proportion,
      PseudoUserPastRefund.calculated_past_user_refund_amt_proportion,
      0) AS calculated_past_user_refund_amt_proportion,
    COALESCE(
      UserItemPastRefund.past_purchase_amt,
      PseudoUserItemPastRefund.past_purchase_amt,
      0) AS past_user_same_item_purchase_amt,
    COALESCE(UserItemPastRefund.past_refund_amt, PseudoUserItemPastRefund.past_refund_amt, 0)
      AS past_user_same_item_refund_amt,
    COALESCE(
      UserItemPastRefund.past_refund_count,
      PseudoUserItemPastRefund.past_refund_count,
      0) AS past_user_same_item_refund_count,
    COALESCE(
      UserItemPastRefund.past_transaction_count,
      PseudoUserItemPastRefund.past_transaction_count,
      0) AS past_user_same_item_transaction_count,
    COALESCE(
      UserItemPastRefund.calculated_past_user_item_refund_rate,
      PseudoUserItemPastRefund.calculated_past_user_item_refund_rate,
      0) AS calculated_past_user_item_refund_rate,
    COALESCE(
      UserItemPastRefund.calculated_past_user_item_refund_amt_proportion,
      PseudoUserItemPastRefund.calculated_past_user_item_refund_amt_proportion,
      0) AS calculated_past_user_item_refund_amt_proportion,
    ProductBrandCategoryPastRefundHistory.* EXCEPT (transaction_id, transaction_date),
    WebsiteEventLength.* EXCEPT (transaction_id, transaction_date),
    WebsiteEventCount.* EXCEPT (transaction_id, transaction_date),
    DemographicAttributes.* EXCEPT (transaction_id, transaction_date),
    RollingWindowRefund.* EXCEPT (transaction_date),
    IF(
      COALESCE(
        UserPastRefund.past_user_purchase_revenue,
        PseudoUserPastRefund.past_user_purchase_revenue,
        0)
        = 0,
      'first time purchase',
      'existing customer transaction') AS customer_type
  FROM `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_target_variable_staging_table` AS TargetVariable
  LEFT JOIN
    `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_item_description_staging_table` AS ItemDescription
    ON
      TargetVariable.transaction_id = ItemDescription.transaction_id
      AND TargetVariable.transaction_date = ItemDescription.transaction_date
  LEFT JOIN `{project_id}.{dataset_id}.{data_pipeline_type}_website_session_information_staging_table` AS WebInfo
    ON
      TargetVariable.transaction_id = WebInfo.transaction_id
      AND TargetVariable.transaction_date = WebInfo.transaction_date
  LEFT JOIN
    `{project_id}.{dataset_id}.{data_pipeline_type}_pseudo_user_past_refund_history_staging_table`
      AS PseudoUserPastRefund
    ON
      TargetVariable.transaction_id = PseudoUserPastRefund.transaction_id
      AND TargetVariable.transaction_date = PseudoUserPastRefund.transaction_date
  LEFT JOIN
    `{project_id}.{dataset_id}.{data_pipeline_type}_user_past_refund_history_staging_table` AS UserPastRefund
    ON
      TargetVariable.transaction_id = UserPastRefund.transaction_id
      AND TargetVariable.transaction_date = UserPastRefund.transaction_date
  LEFT JOIN
    `{project_id}.{dataset_id}.{data_pipeline_type}_user_item_past_refund_history_staging_table`
      AS UserItemPastRefund
    ON
      TargetVariable.transaction_id = UserItemPastRefund.transaction_id
      AND TargetVariable.transaction_date = UserItemPastRefund.transaction_date
  LEFT JOIN
    `{project_id}.{dataset_id}.{data_pipeline_type}_pseudo_user_item_past_refund_history_staging_table`
      AS PseudoUserItemPastRefund
    ON
      TargetVariable.transaction_id = PseudoUserItemPastRefund.transaction_id
      AND TargetVariable.transaction_date = PseudoUserItemPastRefund.transaction_date
  LEFT JOIN
    `{project_id}.{dataset_id}.{data_pipeline_type}_aggregated_website_event_count_by_event_name_staging_table`
      AS WebsiteEventCount
    ON
      TargetVariable.transaction_id = WebsiteEventCount.transaction_id
      AND TargetVariable.transaction_date = WebsiteEventCount.transaction_date
  LEFT JOIN
    `{project_id}.{dataset_id}.{data_pipeline_type}_aggregated_website_event_length_by_event_name_staging_table`
      AS WebsiteEventLength
    ON
      TargetVariable.transaction_id = WebsiteEventLength.transaction_id
      AND TargetVariable.transaction_date = WebsiteEventLength.transaction_date
  LEFT JOIN
    `{project_id}.{dataset_id}.{data_pipeline_type}_rolling_window_daily_refund_data` AS RollingWindowRefund
    ON TargetVariable.transaction_date = RollingWindowRefund.transaction_date
  LEFT JOIN
    `{project_id}.{dataset_id}.{data_pipeline_type}_customer_demographic_attributes_staging_data`
      AS DemographicAttributes
    ON
      TargetVariable.transaction_id = DemographicAttributes.transaction_id
      AND TargetVariable.transaction_date = DemographicAttributes.transaction_date
  LEFT JOIN `{project_id}.{dataset_id}.{data_pipeline_type}_product_brand_category_past_refund_history_staging_table`
    AS ProductBrandCategoryPastRefundHistory
    ON
      TargetVariable.transaction_id = ProductBrandCategoryPastRefundHistory.transaction_id
      AND TargetVariable.transaction_date = ProductBrandCategoryPastRefundHistory.transaction_date
  LEFT JOIN `{project_id}.{dataset_id}.{data_pipeline_type}_traffic_source_campaign_past_refund_history_staging_table` AS TrafficSourceCampaignPastRefundHistory
    ON
      TargetVariable.transaction_id = TrafficSourceCampaignPastRefundHistory.transaction_id
      AND TargetVariable.transaction_date = TrafficSourceCampaignPastRefundHistory.transaction_date
);
