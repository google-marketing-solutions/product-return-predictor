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
-- Step 5: Aggregate event pivot tables at transaction level for website events activity.

-- @param project_id STRING Project id on Google Cloud Platform for intermediate tables and final output tables.
-- @param dataset_id STRING Dataset id of intermediate tables and final output tables.
-- @param data_pipeline_type STRING Whether to process data for training or for prediction.

-- Create pivot table for website event count by event name.
DECLARE dynamic_string_for_aggregating_event_length STRING;
DECLARE dynamic_string_for_aggregating_event_count STRING;

-- Aggregate event length data from pivot table.
SET dynamic_string_for_aggregating_event_length = (
  SELECT
    STRING_AGG(CONCAT(' SUM(IFNULL(', column_name, ',0)) AS ', column_name), ', ')
      AS column_operation
  FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.COLUMNS`
  WHERE
    table_name = '{data_pipeline_type}_website_event_length_by_event_name_staging_table'
    AND column_name NOT IN ('transaction_id', 'transaction_date')
);

EXECUTE
  IMMEDIATE
    FORMAT(
      '''
      CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_aggregated_website_event_length_by_event_name_staging_table`
      AS
      SELECT transaction_id, transaction_date, % s
      FROM `{project_id}.{dataset_id}.{data_pipeline_type}_website_event_length_by_event_name_staging_table`
      GROUP BY transaction_id, transaction_date
      ''',
      dynamic_string_for_aggregating_event_length);

-- Aggregate event count data from pivot table.
SET dynamic_string_for_aggregating_event_count = (
  SELECT
    STRING_AGG(CONCAT(' SUM(IFNULL(', column_name, ',0)) AS ', column_name), ', ')
      AS column_operation
  FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.COLUMNS`
  WHERE
    table_name = '{data_pipeline_type}_website_event_count_by_event_name_staging_table'
    AND column_name NOT IN ('transaction_id', 'transaction_date')
);

EXECUTE
  IMMEDIATE
    FORMAT(
      '''
      CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_aggregated_website_event_count_by_event_name_staging_table`
      AS
      SELECT transaction_id, transaction_date, % s
      FROM `{project_id}.{dataset_id}.{data_pipeline_type}_website_event_count_by_event_name_staging_table`
      GROUP BY transaction_id, transaction_date
      ''',
      dynamic_string_for_aggregating_event_count);
