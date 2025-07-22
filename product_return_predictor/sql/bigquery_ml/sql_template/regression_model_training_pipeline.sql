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

-- SQL query to create a 1-step regression model in BigQuery ML.

-- Create a regression model to predict the refund_value.
-- Note: Here the label is either the column for refund value or refund proportion (refund_value/transaction_value).

-- @param project_id STRING Project id on Google Cloud Platform for ml ready dataset.
-- @param dataset_id STRING Dataset id on Google Cloud Platform for ml ready dataset.
-- @param transaction_date_col STRING Column name for the transaction date in ml ready dataset.
-- @param transaction_id_col STRING Column name for the transaction id in ml ready dataset.
-- @param refund_value_col STRING Column name for the refund value in ml ready dataset.
-- @param regression_model_type STRING Model type name for regression model (e.g. Linear Regression).
-- @param regression_model_config STRING Configure for BQML regression model (e.g. Parameter Configuration, Hyperparameter Tuning).
-- @param num_tiers INT64 Number of tiers for calculating tier level average prediction.
-- @param preprocessed_table_name STRING Table name for preprocessed data used for model training.

-- Build and train regression model using hyperparameter tuning
CREATE OR REPLACE MODEL `{project_id}.{dataset_id}.TRAINING_ml_data_{preprocessed_table_name}_with_target_variable_refund_value_regressor_{regression_model_type}`
  OPTIONS (
    {regression_model_config},
    ENABLE_GLOBAL_EXPLAIN = TRUE,
    INPUT_LABEL_COLS = ['{refund_value_col}'])
AS
SELECT * EXCEPT ({transaction_date_col}, {transaction_id_col}, train_test)
FROM `{project_id}.{dataset_id}.TRAINING_ml_data_{preprocessed_table_name}_with_target_variable_refund_value`
WHERE
  train_test = 'train';

-- Get trial info from hyperparameter tuning
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.TRAINING_{preprocessed_table_name}_with_target_variable_{refund_value_col}_regression_{regression_model_type}_trial_info`
AS (
  SELECT *
  FROM
    ML.TRIAL_INFO(
      MODEL `{project_id}.{dataset_id}.TRAINING_ml_data_{preprocessed_table_name}_with_target_variable_refund_value_regressor_{regression_model_type}`)
);

-- Get trial_info from best-performing hyperparameters/trials
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.TRAINING_{preprocessed_table_name}_with_target_variable_{refund_value_col}_regression_{regression_model_type}_best_trial_info`
AS (
  SELECT
    *,
    '{preprocessed_table_name}_with_target_variable_{refund_value_col}_regressor_{regression_model_type}' AS model_name,
  FROM
    ML.TRIAL_INFO(
      MODEL `{project_id}.{dataset_id}.TRAINING_ml_data_{preprocessed_table_name}_with_target_variable_refund_value_regressor_{regression_model_type}`)
  WHERE
    trial_id IN (
      SELECT
        MIN(trial_id) AS min_trial_id
      FROM
        ML.TRIAL_INFO(
          MODEL `{project_id}.{dataset_id}.TRAINING_ml_data_{preprocessed_table_name}_with_target_variable_refund_value_regressor_{regression_model_type}`)
          AS MLTrialInfo
      WHERE
        MLTrialInfo.status = 'SUCCEEDED'
        AND MLTrialInfo.is_optimal = TRUE
    )
);

-- Model Evaluation on test set
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.TRAINING_{preprocessed_table_name}_with_target_variable_{refund_value_col}_regression_{regression_model_type}_test_set_evaluation_metrics`
AS (
  SELECT *
  FROM
    ML.EVALUATE(
      MODEL `{project_id}.{dataset_id}.TRAINING_ml_data_{preprocessed_table_name}_with_target_variable_refund_value_regressor_{regression_model_type}`,
      (
        SELECT * EXCEPT ({transaction_date_col}, {transaction_id_col}, train_test)
        FROM `{project_id}.{dataset_id}.TRAINING_ml_data_{preprocessed_table_name}_with_target_variable_refund_value`
        WHERE
          train_test = 'test'
      ))
);

-- Prediction on training and test sets
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.TRAINING_{preprocessed_table_name}_with_target_variable_{refund_value_col}_regression_{regression_model_type}_predictions`
AS (
  SELECT
    {transaction_date_col},
    {transaction_id_col},
    predicted_{refund_value_col} AS prediction,
    {refund_value_col} AS actual,
    train_test
  FROM
    ML.PREDICT(
      MODEL `{project_id}.{dataset_id}.TRAINING_ml_data_{preprocessed_table_name}_with_target_variable_refund_value_regressor_{regression_model_type}`,
      (
        SELECT *
        FROM `{project_id}.{dataset_id}.TRAINING_ml_data_{preprocessed_table_name}_with_target_variable_refund_value`
      ))
);

-- Create feature importance table for regression model
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.TRAINING_{preprocessed_table_name}_with_target_variable_{refund_value_col}_regression_{regression_model_type}_feature_importance`
AS (
  SELECT *
  FROM
    ML.GLOBAL_EXPLAIN(
      MODEL `{project_id}.{dataset_id}.TRAINING_ml_data_{preprocessed_table_name}_with_target_variable_refund_value_regressor_{regression_model_type}`)
);

-- Create performance metrics for the regression model using finalized predictions
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.TRAINING_{preprocessed_table_name}_with_target_variable_{refund_value_col}_regression_performance_metrics`
AS (
  WITH
    SpearmanCorrelationScore
    AS (
      SELECT
        train_test,
        CORR(actual_ranking, prediction_ranking) AS spearman_rank_correlation_score
      FROM
        (
          SELECT
            train_test,
            RANK() OVER (ORDER BY actual DESC) AS actual_ranking,
            RANK() OVER (ORDER BY prediction DESC) AS prediction_ranking
          FROM
            `{project_id}.{dataset_id}.TRAINING_{preprocessed_table_name}_with_target_variable_{refund_value_col}_regression_{regression_model_type}_predictions`
        ) AS Ranking
      GROUP BY train_test
    ),
    ModelAccuracy AS (
      SELECT
        train_test,
        SQRT(AVG(POWER(actual - prediction, 2))) AS rmse,
        AVG(ABS(actual - prediction)) AS mae
      FROM
        `{project_id}.{dataset_id}.TRAINING_{preprocessed_table_name}_with_target_variable_{refund_value_col}_regression_{regression_model_type}_predictions`
      GROUP BY train_test
    ),
    NumObservations AS (
      SELECT
        train_test,
        COUNT(*) AS n_observations
      FROM
        `{project_id}.{dataset_id}.TRAINING_{preprocessed_table_name}_with_target_variable_{refund_value_col}_regression_{regression_model_type}_predictions`
      GROUP BY train_test
    )
  SELECT
    NumObservations.train_test,
    NumObservations.n_observations,
    ModelAccuracy.rmse,
    ModelAccuracy.mae,
    SpearmanCorrelationScore.spearman_rank_correlation_score
  FROM NumObservations
  LEFT JOIN ModelAccuracy
    ON NumObservations.train_test = ModelAccuracy.train_test
  LEFT JOIN SpearmanCorrelationScore
    ON NumObservations.train_test = SpearmanCorrelationScore.train_test
);

-- Create tier-level average prediction comparison between prediction and actual
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.TRAINING_{preprocessed_table_name}_with_target_variable_{refund_value_col}_tier_level_avg_prediction_comparison`
AS (
  SELECT
    train_test,
    tier,
    AVG(actual) AS tier_level_avg_actual,
    AVG(prediction) AS tier_level_avg_prediction
  FROM
    (
      SELECT
        train_test,
        NTILE({num_tiers}) OVER (PARTITION BY train_test ORDER BY prediction DESC) AS tier,
        CAST(actual AS FLOAT64) AS actual,
        CAST(prediction AS FLOAT64) AS prediction
      FROM
        `{project_id}.{dataset_id}.TRAINING_{preprocessed_table_name}_with_target_variable_{refund_value_col}_regression_{regression_model_type}_predictions`
    ) AS Pred
  GROUP BY train_test, tier
  ORDER BY train_test, tier
);
