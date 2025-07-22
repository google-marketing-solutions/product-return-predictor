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

-- SQL query to demonstrate how the BQML can be used to predict with a trained one-step regression model using `bigquery-public-data.ml_datasets.penguins` as the sample dataset.
-- For demonstration, body_mass_g is considered as the label for regression model.
-- penguin_id will be created as the id column to distinguish each row in the dataset. 

-- @param project_id STRING Project id on Google Cloud Platform for ml ready dataset.
-- @param dataset_id STRING Dataset id on Google Cloud Platform for ml ready dataset.

-- Create sample dataset for demonstration
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.PREDICTION_penguins`
AS (
  SELECT
    * EXCEPT (random),
    RANK() OVER (ORDER BY random) AS penguin_id,
  FROM
    (
      SELECT
        *,
        RAND() AS random
      FROM `bigquery-public-data.ml_datasets.penguins`
      WHERE
        body_mass_g IS NOT NULL
    ) AS PenguinData
);

-- Prediction
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.PREDICTION_body_mass_g_regression_linear_reg_predictions`
AS (
  SELECT
    penguin_id,
    predicted_body_mass_g AS prediction,
  FROM
    ML.PREDICT(
      MODEL `{project_id}.{dataset_id}.body_mass_g_regressor_linear_reg`,
      (
        SELECT *
        FROM
          `{project_id}.{dataset_id}.PREDICTION_penguins`
      ))
);
