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
-- Step 8: Drop intermediate tables.

-- @param project_id STRING Project id on Google Cloud Platform for intermediate tables and final output tables.
-- @param dataset_id STRING Dataset id of intermediate tables and final output tables.
-- @param data_pipeline_type STRING Whether to process data for training or for prediction.

-- Set data_pipeline_type to determine whether to process data for training or for prediction
DECLARE data_pipeline_type STRING DEFAULT '{data_pipeline_type}';

DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_id_to_user_id_mapping`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_id_to_pseudo_user_id_mapping`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_id_to_session_id_mapping`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_id_to_user_id_item_mapping`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_id_to_pseudo_user_id_item_mapping`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_date_lookup`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_transaction_data`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_product_level_transaction_refund_data`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_daily_level_past_transaction_refund_data`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_session_id_level_web_traffic_info`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_session_id_level_web_event_info`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_customer_purchase_refund_history_data_user_id_level`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_customer_purchase_refund_history_data_user_pseudo_id_level`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_transaction_id_and_item_level_stats`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_customer_product_purchase_refund_interaction_history_data_user_id_level`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_customer_product_purchase_refund_interaction_history_data_user_pseudo_id_level`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_item_description_staging_table`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_target_variable_staging_table`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_website_session_information_staging_table`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_website_event_count_by_event_name_staging_table`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_website_event_length_by_event_name_staging_table`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_pseudo_user_past_refund_history_staging_table`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_user_past_refund_history_staging_table`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_user_item_past_refund_history_staging_table`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_pseudo_user_item_past_refund_history_staging_table`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_user_id_lookup`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_user_pseudo_id_lookup`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_lookup`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_item_lookup`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_session_id_lookup`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_rolling_window_daily_refund_data`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_daily_level_transaction_refund_data`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_aggregated_website_event_length_by_event_name_staging_table`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_aggregated_website_event_count_by_event_name_staging_table`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_country_level_transaction_refund_data`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_city_level_transaction_refund_data`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_customer_demographic_attributes_staging_data`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_customer_demographic_attributes_data_user_id_level`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_customer_demographic_attributes_data_pseudo_user_id_level`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_device_category_level_transaction_refund_data`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_device_operating_system_level_transaction_refund_data`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_device_language_level_transaction_refund_data`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_product_brand_category_past_refund_history_staging_table`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_item_category_level_transaction_refund_data`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_brand_level_transaction_refund_data`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_traffic_source_campaign_past_refund_history_staging_table`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_traffic_source_level_past_transaction_refund_data`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_traffic_medium_level_past_transaction_refund_data`;
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_preprocessed_traffic_campaign_level_past_transaction_refund_data`;

-- Need to keep website_event_staging_table created during training phase for creating event metrics for predictin data later.
IF data_pipeline_type = 'PREDICTION' THEN
  DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{data_pipeline_type}_website_event_staging_table`;
END IF;


