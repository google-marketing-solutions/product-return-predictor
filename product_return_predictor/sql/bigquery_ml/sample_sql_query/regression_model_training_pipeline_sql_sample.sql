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

-- SQL query to demonstrate how the BQML can be used to build a one-step regression model using `bigquery-public-data.ml_datasets.penguins` as the sample dataset.
-- For demonstration, we will use body_mass_g as the label for regression model.
-- penguin_id will be created as the id column to distinguish each row in the dataset and train_test column will be created to split the data into train and test sets.

-- @param project_id STRING Project id on Google Cloud Platform for ml ready dataset.
-- @param dataset_id STRING Dataset id on Google Cloud Platform for ml ready dataset.

-- Create sample dataset for demonstration
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.TRAINING_penguins`
AS (
  SELECT
    * EXCEPT (split_field, random),
    RANK() OVER (ORDER BY random) AS penguin_id,
    IF(split_field <= 0.8, 'train', 'test') AS train_test
  FROM
    (
      SELECT
        *,
        ROUND(ABS(RAND()), 1) AS split_field,
        RAND() AS random
      FROM `bigquery-public-data.ml_datasets.penguins`
      WHERE
        body_mass_g IS NOT NULL
    ) AS PenguinData
);

-- Build and train regression model using hyperparameter tuning
CREATE OR REPLACE MODEL `{project_id}.{dataset_id}.body_mass_g_regressor_linear_reg`
  OPTIONS (
    MODEL_TYPE = 'LINEAR_REG',
    ENABLE_GLOBAL_EXPLAIN = TRUE,
    L1_REG = HPARAM_RANGE(0, 5.0),
    NUM_TRIALS = 5,
    INPUT_LABEL_COLS = ['body_mass_g'])
AS
SELECT * EXCEPT (penguin_id, train_test)
FROM
  `{project_id}.{dataset_id}.TRAINING_penguins`
WHERE
  train_test = 'train';

-- Get trial info from hyperparameter tuning
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.TRAINING_body_mass_g_regression_linear_reg_trail_info`
AS (
  SELECT *
  FROM
    ML.TRIAL_INFO(MODEL `{project_id}.{dataset_id}.body_mass_g_regressor_linear_reg`)
);

-- Get trial_info from best-performing hyperparameters/trials
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.TRAINING_body_mass_g_regression_linear_reg_best_trial_info`
AS (
  SELECT
    *,
    'body_mass_g_regression_linear_reg' AS model_name,
  FROM
    ML.TRIAL_INFO(MODEL `{project_id}.{dataset_id}.body_mass_g_regressor_linear_reg`)
  WHERE
    trial_id IN (
      SELECT
        MIN(trial_id) AS min_trial_id
      FROM
        ML.TRIAL_INFO(
          MODEL `{project_id}.{dataset_id}.body_mass_g_regressor_linear_reg`)
          AS MLTrialInfo
      WHERE
        MLTrialInfo.status = 'SUCCEEDED'
        AND MLTrialInfo.is_optimal = TRUE
    )
);

-- Model Evaluation on test set
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.TRAINING_body_mass_g_regression_linear_reg_test_set_evaluation_metrics`
AS (
  SELECT
    *
  FROM
    ML.EVALUATE(
      MODEL `{project_id}.{dataset_id}.body_mass_g_regressor_linear_reg`,
      (
        SELECT * EXCEPT (penguin_id, train_test)
        FROM
          `{project_id}.{dataset_id}.TRAINING_penguins`
        WHERE
          train_test = 'test'
      ))
);

-- Prediction on training and test sets
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.TRAINING_body_mass_g_regression_linear_reg_predictions`
AS (
  SELECT
    penguin_id,
    predicted_body_mass_g AS prediction,
    train_test,
    body_mass_g AS actual
  FROM
    ML.PREDICT(
      MODEL `{project_id}.{dataset_id}.body_mass_g_regressor_linear_reg`,
      (
        SELECT *
        FROM
          `{project_id}.{dataset_id}.TRAINING_penguins`
      ))
);

-- Create feature importance table for regression model
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.TRAINING_body_mass_g_regression_linear_reg_feature_importance`
AS (
  SELECT *
  FROM
    ML.GLOBAL_EXPLAIN(
      MODEL `{project_id}.{dataset_id}.body_mass_g_regressor_linear_reg`)
);

-- Create performance metrics for the 2-step model using finalized predictions
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.TRAINING_body_mass_g_regression_performance_metrics`
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
          FROM `{project_id}.{dataset_id}.TRAINING_body_mass_g_regression_linear_reg_predictions`
        ) AS Ranking
      GROUP BY train_test
    ),
    ModelAccuracy AS (
      SELECT
        train_test,
        SQRT(AVG(POWER(actual - prediction, 2))) AS rmse,
        AVG(ABS(actual - prediction)) AS mae
      FROM `{project_id}.{dataset_id}.TRAINING_body_mass_g_regression_linear_reg_predictions`
      GROUP BY train_test
    ),
    NumObservations AS (
      SELECT
        train_test,
        COUNT(*) AS n_observations
      FROM `{project_id}.{dataset_id}.TRAINING_body_mass_g_regression_linear_reg_predictions`
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
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.TRAINING_body_mass_g_tier_level_avg_prediction_comparison`
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
        NTILE(10) OVER (PARTITION BY train_test ORDER BY prediction DESC) AS tier,
        CAST(actual AS FLOAT64) AS actual,
        CAST(prediction AS FLOAT64) AS prediction
      FROM `{project_id}.{dataset_id}.TRAINING_body_mass_g_regression_linear_reg_predictions`
    ) AS Pred
  GROUP BY train_test, tier
  ORDER BY train_test, tier
);
