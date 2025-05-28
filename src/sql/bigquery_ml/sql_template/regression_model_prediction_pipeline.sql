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

-- SQL query to predict with a 1-step regression model in BigQuery ML.

-- @param project_id STRING Project id on Google Cloud Platform for ml ready dataset.
-- @param dataset_id STRING Dataset id on Google Cloud Platform for ml ready dataset.
-- @param transaction_date_col STRING Column name for the transaction date in ml ready dataset.
-- @param transaction_id_col STRING Column name for the transaction id in ml ready dataset.
-- @param refund_value_col STRING Column name for the refund value in ml ready dataset.
-- @param regression_model_type STRING Model type name for regression model (e.g. Linear Regression).
-- @param preprocessed_table_name STRING Table name for preprocessed data used for model prediction.

-- Prediction
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.PREDICTION_{preprocessed_table_name}_with_target_variable_{refund_value_col}_regression_{regression_model_type}_predictions`
AS (
  SELECT
    {transaction_date_col},
    {transaction_id_col},
    predicted_{refund_value_col} AS prediction
  FROM
    ML.PREDICT(
      MODEL `{project_id}.{dataset_id}.{preprocessed_table_name}_with_target_variable_{refund_value_col}_regressor_{regression_model_type}`,
      (
        SELECT *
        FROM `{project_id}.{dataset_id}.PREDICTION_ml_data_{preprocessed_table_name}_with_target_variable_refund_value`
      ))
);