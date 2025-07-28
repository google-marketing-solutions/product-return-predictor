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
-- Step 1: Create lookup tables to limit the data for processing for creating ML training & prediction dataset.

-- @param ga4_project_id STRING Project id on Google Cloud Platform for the GA4 dataset (source data).
-- @param ga4_dataset_id STRING Dataset id of the GA4 dataset (source data).
-- @param project_id STRING Project id on Google Cloud Platform for intermediate tables and final output tables.
-- @param dataset_id STRING Dataset id of intermediate tables and final output tables.
-- @param return_policy_window_in_days INT The number of days after the transaction date that allows for product return/refund.
-- @param recency_of_data_in_days INT The number of days of data before the current date to use for creating datasets and training the model.
-- @param data_pipeline_type STRING Whether to process data for training or for prediction.
-- @param start_date DATE Start date for the data to use recent data only to reduce the processing time and make sure the data used in the model is relevant.
-- @param recency_of_transaction_for_prediction_in_days INT The number of days of data before the current date to use for creating dataset for model prediction.
-- @param transaction_start_date DATE Start date to limit the transaction ids to be processed for model training & prediction.
-- @param end_date DATE Transaction data that is older than X days (return policy window) ago to make sure the refund is all set.


-- Set data_pipeline_type to determine whether to process data for training or for testing
DECLARE data_pipeline_type STRING DEFAULT '{data_pipeline_type}';

-- Set start_date for the data to use recent data only to reduce the processing time and make sure the data used in the model is relevant.
DECLARE start_date DATE DEFAULT DATE_ADD(CURRENT_DATE(), INTERVAL -{recency_of_data_in_days} DAY);

-- Set transaction_start_date to limit the transaction ids to be processed for model training & prediction
-- For training pipeline, transaction_start_date would be the same as the start date above.
-- For prediction pipeline, transaction_start_date and end_date would be used to limit the transaction to be fed into the prediction pipeline.
DECLARE transaction_start_date DATE DEFAULT (CASE data_pipeline_type WHEN'TRAINING' THEN DATE_ADD(CURRENT_DATE(), INTERVAL -{recency_of_data_in_days} DAY) ELSE DATE_ADD(CURRENT_DATE(), INTERVAL -{recency_of_transaction_for_prediction_in_days} DAY) END);

-- Declare end_date to make sure to use transaction data that is older than X days (return policy window) ago to make sure the refund is all set.
DECLARE end_date DATE DEFAULT (CASE data_pipeline_type WHEN'TRAINING' THEN DATE_ADD(CURRENT_DATE(), INTERVAL -{return_policy_window_in_days}-1 DAY) ELSE CURRENT_DATE() END);

-- Create table value function (TVF) to return data with valid transaction revenue and transaction id from GA4.
CREATE OR REPLACE TABLE FUNCTION `{project_id}.{dataset_id}.FilterTransactionData`(
  end_date DATE)
AS (
  SELECT *
  FROM `{ga4_project_id}.{ga4_dataset_id}.events_*`
  WHERE
    ecommerce.purchase_revenue IS NOT NULL
    AND ecommerce.purchase_revenue > 0
    AND ecommerce.transaction_id IS NOT NULL
    AND ecommerce.transaction_id <> '(not set)'
    AND PARSE_DATE('%Y%m%d', event_date) <= end_date
);

-- Create lookup table for transaction_id.
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_lookup`
AS (
  SELECT DISTINCT
    PARSE_DATE('%Y%m%d', event_date) AS transaction_date,
    ecommerce.transaction_id
  FROM `{project_id}.{dataset_id}.FilterTransactionData`(end_date)
  WHERE PARSE_DATE('%Y%m%d', event_date) >= transaction_start_date
);

-- Create lookup table for transaction date
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_transaction_date_lookup`
AS (
  SELECT DISTINCT
    PARSE_DATE('%Y%m%d', event_date) AS transaction_date
  FROM `{project_id}.{dataset_id}.FilterTransactionData`(end_date)
  WHERE PARSE_DATE('%Y%m%d', event_date) >= transaction_start_date
);

-- Create lookup table for user_id.
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_user_id_lookup`
AS (
  SELECT DISTINCT CAST(user_id AS STRING) AS user_id
  FROM `{project_id}.{dataset_id}.FilterTransactionData`(end_date)
  WHERE user_id IS NOT NULL
);

-- Create lookup table for user_pseudo_id.
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_user_pseudo_id_lookup`
AS (
  SELECT DISTINCT CAST(user_pseudo_id AS STRING) AS user_pseudo_id
  FROM `{project_id}.{dataset_id}.FilterTransactionData`(end_date)
  WHERE user_pseudo_id IS NOT NULL
);

-- Create lookup table for session_id.
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_session_id_lookup`
AS (
  SELECT DISTINCT
    CONCAT(
      SAFE_CAST(user_pseudo_id AS STRING),
      '_',
      (
        SELECT SAFE_CAST(value.int_value AS STRING)
        FROM UNNEST(event_params)
        WHERE key = 'ga_session_id'
      ))
      AS session_id
  FROM `{project_id}.{dataset_id}.FilterTransactionData`(end_date)
  WHERE
    user_pseudo_id IS NOT NULL
    AND (
      SELECT SAFE_CAST(value.int_value AS STRING)
      FROM UNNEST(event_params)
      WHERE key = 'ga_session_id'
    ) IS NOT NULL
    AND PARSE_DATE('%Y%m%d', event_date) >= start_date
);

-- Create lookup table for item_name.
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_item_lookup`
AS (
  SELECT DISTINCT SAFE_CAST(items.item_name AS STRING) AS item_name
  FROM `{project_id}.{dataset_id}.FilterTransactionData`(end_date)
  CROSS JOIN UNNEST(items) AS items
  WHERE
    IFNULL(items.item_revenue, 0) > 0
    AND items.item_id IS NOT NULL
    AND items.item_id <> '(not set)'
    AND items.item_name IS NOT NULL
    AND items.item_name <> '(not set)'
);
