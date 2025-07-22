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
-- Step 4: Create pivot tables for website events activity.

-- @param project_id STRING Project id on Google Cloud Platform for intermediate tables and final output tables.
-- @param dataset_id STRING Dataset id of intermediate tables and final output tables.
-- @param data_pipeline_type STRING Whether to process data for training or for prediction.

DECLARE dynamic_string_for_pivoting_by_event_name STRING;

-- Create pivot table for website event count by event name.
SET dynamic_string_for_pivoting_by_event_name = (
  SELECT CONCAT('(', STRING_AGG(DISTINCT CONCAT("'", event_name, "'"), ','), ')')
  FROM `{project_id}.{dataset_id}.{data_pipeline_type}_session_id_level_web_event_info`
);

EXECUTE
  IMMEDIATE
    FORMAT(
      '''
      CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_website_event_count_by_event_name_staging_table`
      AS
      SELECT * EXCEPT (event_length_in_seconds)
      FROM `{project_id}.{dataset_id}.{data_pipeline_type}_website_event_staging_table`
        PIVOT(SUM(event_count) AS event_count FOR event_name IN % s)
      ''',
      dynamic_string_for_pivoting_by_event_name);

-- Create pivot table for website event length by event name.
EXECUTE
  IMMEDIATE
    FORMAT(
      '''
      CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{data_pipeline_type}_website_event_length_by_event_name_staging_table`
      AS
      SELECT * EXCEPT (event_count)
      FROM `{project_id}.{dataset_id}.{data_pipeline_type}_website_event_staging_table`
          PIVOT(SUM(event_length_in_seconds) AS event_length_in_seconds FOR event_name IN % s);
      ''',
      dynamic_string_for_pivoting_by_event_name);
