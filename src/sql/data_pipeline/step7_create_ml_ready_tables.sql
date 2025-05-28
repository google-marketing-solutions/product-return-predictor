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
-- Step 7: Create ML ready tables for existing customers and first time purchase.

-- @param project_id STRING Project id on Google Cloud Platform for intermediate tables and final output tables.
-- @param dataset_id STRING Dataset id of intermediate tables and final output tables.
-- @param data_pipeline_type STRING Whether to process data for training or for prediction.

-- Create ML ready table for existing customers (with more relevant features).
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_ml_ready_data_for_existing_customers`
AS (
SELECT * EXCEPT (customer_type)
FROM `{project_id}.{dataset_id}.{data_pipeline_type}_ml_ready_data`
WHERE customer_type = 'existing customer transaction');

-- Create ML ready table for new customers (with fewer relevant features).
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_ml_ready_data_for_first_time_purchase`
AS (
SELECT
  * EXCEPT (
    customer_type,
    past_user_purchase_revenue,
    past_user_refund_value,
    past_user_refund_count,
    past_user_transaction_count,
    calculated_past_user_refund_rate,
    calculated_past_user_refund_amt_proportion,
    past_user_same_item_purchase_amt,
    past_user_same_item_refund_amt,
    past_user_same_item_refund_count,
    past_user_same_item_transaction_count,
    calculated_past_user_item_refund_rate,
    calculated_past_user_item_refund_amt_proportion)
FROM `{project_id}.{dataset_id}.{data_pipeline_type}_ml_ready_data`
WHERE customer_type = 'first time purchase');
