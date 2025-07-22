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

-- SQL query to predict with a trained 2-step model (classification first and then regression) in BigQuery ML.

-- @param project_id STRING Project id on Google Cloud Platform for ml ready dataset.
-- @param dataset_id STRING Dataset id on Google Cloud Platform for ml ready dataset.
-- @param transaction_date_col STRING Column name for the transaction date in ml ready dataset.
-- @param transaction_id_col STRING Column name for the transaction id in ml ready dataset.
-- @param refund_flag_col STRING Column name for the refund flag in ml ready dataset.
-- @param refund_value_col STRING Column name for the refund value in ml ready dataset.
-- @param binary_classifier_model_type STRING Model type name for binary classification model (e.g. Logistic Regression).
-- @param regression_model_type STRING Model type name for regression model (e.g. Linear Regression).
-- @param probability_threshold_for_prediction FLOAT64 The probability threshold to predict whether transaction has products refunded.
-- @param preprocessed_table_name STRING Table name for preprocessed data used for model prediction.

-- Prediction using classification model
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.PREDICTION_{preprocessed_table_name}_with_target_variable_{refund_flag_col}_binary_classifier_{binary_classifier_model_type}_predictions`
AS (
  SELECT
    Prediction.{transaction_date_col},
    Prediction.{transaction_id_col},
    Prob.label AS label,
    Prob.prob AS probability
  FROM
    (
      SELECT *
      FROM
        ML.PREDICT(
          MODEL
            `{project_id}.{dataset_id}.TRAINING_ml_data_{preprocessed_training_table_name}_with_target_variable_refund_flag_binary_classifier_{binary_classifier_model_type}`,
          (
            SELECT * FROM `{project_id}.{dataset_id}.PREDICTION_ml_data_{preprocessed_table_name}_with_target_variable_refund_flag`
          ))
    ) AS Prediction
  CROSS JOIN UNNEST(Prediction.predicted_{refund_flag_col}_probs) AS Prob
  WHERE Prob.prob > {probability_threshold_for_prediction}
);

-- Prediction using regression model
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.PREDICTION_{preprocessed_table_name}_with_target_variable_{refund_value_col}_regression_{regression_model_type}_predictions`
AS (
  SELECT
    {transaction_date_col},
    {transaction_id_col},
    predicted_{refund_value_col} AS prediction
 FROM
    ML.PREDICT(
      MODEL `{project_id}.{dataset_id}.TRAINING_ml_data_{preprocessed_training_table_name}_with_target_variable_refund_value_regressor_{regression_model_type}`,
      (
        SELECT *
        FROM `{project_id}.{dataset_id}.PREDICTION_ml_data_{preprocessed_table_name}_with_target_variable_refund_value`
      ))
);

-- Merge the predictions from the two models
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.PREDICTION_{preprocessed_table_name}_with_target_variable_{refund_flag_col}_{refund_value_col}_2_step_predictions`
AS (
  SELECT
    ClassifierPrediction.{transaction_id_col},
    ClassifierPrediction.{transaction_date_col},
    IF(ClassifierPrediction.probability < {probability_threshold_for_prediction}, 0, RegressionPrediction.prediction) AS prediction
  FROM
    `{project_id}.{dataset_id}.PREDICTION_{preprocessed_table_name}_with_target_variable_{refund_flag_col}_binary_classifier_{binary_classifier_model_type}_predictions`
      AS ClassifierPrediction
  LEFT JOIN
    `{project_id}.{dataset_id}.PREDICTION_{preprocessed_table_name}_with_target_variable_{refund_value_col}_regression_{regression_model_type}_predictions`
      AS RegressionPrediction
    ON
      ClassifierPrediction.{transaction_id_col} = RegressionPrediction.{transaction_id_col}
      AND ClassifierPrediction.{transaction_date_col}
        = RegressionPrediction.{transaction_date_col}
);
