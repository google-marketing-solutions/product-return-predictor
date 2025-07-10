# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for evaluating model performance and generating prediction."""

from collections.abc import Mapping

from google.cloud import bigquery
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from product_return_predictor.src.python import constant
from product_return_predictor.src.python import utils


def _validate_regression_model_type(
    regression_model_type: constant.SupportedModelTypes,
) -> None:
  """Check if regression model type is supported.

  Args:
    regression_model_type: Model type of regression model (e.g. Linear
      regression).

  Raises:
    ValueError: If regression model type is not supported.
  """
  if regression_model_type not in [
      constant.LinearBigQueryMLModelType.LINEAR_REGRESSION,
      constant.DNNBigQueryMLModelType.DNN_REGRESSOR,
      constant.BoostedTreeBigQueryMLModelType.BOOSTED_TREE_REGRESSOR,
  ]:
    raise ValueError(
        f'Regression model type {regression_model_type} is not supported.'
    )


def _validate_regression_and_classification_model_types(
    binary_classifier_model_type: constant.SupportedModelTypes = constant.LinearBigQueryMLModelType.LOGISTIC_REGRESSION,
    regression_model_type: constant.SupportedModelTypes = constant.LinearBigQueryMLModelType.LINEAR_REGRESSION,
):
  """Validate input model types for generating model performance metrics.

  Args:
    binary_classifier_model_type: Model type of binary classification (e.g.
      Logistic regression).
    regression_model_type: Model type of regression model (e.g. Linear
      regression).

  Raises:
    ValueError: If regression or classification model type is not supported.
  """
  if binary_classifier_model_type not in [
      constant.LinearBigQueryMLModelType.LOGISTIC_REGRESSION,
      constant.DNNBigQueryMLModelType.DNN_CLASSIFIER,
      constant.BoostedTreeBigQueryMLModelType.BOOSTED_TREE_CLASSIFIER,
  ]:
    raise ValueError(
        f'Binary classifier model type {binary_classifier_model_type} is not'
        ' supported.'
    )
  _validate_regression_model_type(regression_model_type)


def _regression_model_performance_metrics_data(
    project_id: str,
    bigquery_client: bigquery.Client,
    dataset_id: str,
    refund_value: str,
    regression_model_type: constant.SupportedModelTypes,
    preprocessed_table_name: str,
) -> pd.DataFrame:
  """Get regression model performance metrics.

  Args:
    project_id: Google Cloud Platform project id.
    bigquery_client: Google Cloud BigQuery client.
    dataset_id: Google Cloud Platform dataset id where model performance metrics
      are stored.
    refund_value: Name of refund value column in training data.
    regression_model_type: Model type of regression model (e.g. Linear
      regression).
    preprocessed_table_name: Name of preprocessed table after feature
      engineering.

  Returns:
    Regression model performance metrics dataframe.
  """
  regression_model_performance_metrics_table_name = (
      constant.MODEL_PERFORMANCE_METRICS_TABLE_NAMES['regression_model'].format(
          preprocessed_table_name=preprocessed_table_name,
          refund_value_col=refund_value,
          regression_model_type=regression_model_type.value,
      )
  )
  regression_model_performance_metrics_df = utils.read_bq_table_to_df(
      project_id=project_id,
      bigquery_client=bigquery_client,
      dataset_id=dataset_id,
      table_name=regression_model_performance_metrics_table_name,
  )
  return regression_model_performance_metrics_df


def _binary_classifier_model_performance_metrics_data(
    project_id: str,
    bigquery_client: bigquery.Client,
    dataset_id: str,
    refund_flag: str,
    binary_classifier_model_type: constant.SupportedModelTypes,
    preprocessed_table_name: str,
) -> pd.DataFrame:
  """Get binary classifier model performance metrics.

  Args:
    project_id: Google Cloud Platform project id.
    bigquery_client: Google Cloud BigQuery client.
    dataset_id: Google Cloud Platform dataset id where model performance metrics
      are stored.
    refund_flag: Name of refund flag column in training data.
    binary_classifier_model_type: Model type of binary classification (e.g.
      Logistic regression).
    preprocessed_table_name: Name of preprocessed table after feature
      engineering.

  Returns:
    Binary classifier model performance metrics dataframe.
  """
  classification_model_performance_metrics_table_name = (
      constant.MODEL_PERFORMANCE_METRICS_TABLE_NAMES[
          'classification_model'
      ].format(
          refund_flag_col=refund_flag,
          binary_classifier_model_type=binary_classifier_model_type.value,
          preprocessed_table_name=preprocessed_table_name,
      )
  )

  classification_model_performance_metrics_df = utils.read_bq_table_to_df(
      project_id=project_id,
      bigquery_client=bigquery_client,
      dataset_id=dataset_id,
      table_name=classification_model_performance_metrics_table_name,
  )
  return classification_model_performance_metrics_df


def _two_step_model_performance_metrics_data(
    project_id: str,
    bigquery_client: bigquery.Client,
    dataset_id: str,
    refund_value: str,
    refund_flag: str,
    preprocessed_table_name: str,
) -> pd.DataFrame:
  """Get two-step (classification & regression) model performance metrics.

  Args:
    project_id: Google Cloud Platform project id.
    bigquery_client: Google Cloud BigQuery client.
    dataset_id: Google Cloud Platform dataset id where model performance metrics
      are stored.
    refund_value: Name of refund value column in training data.
    refund_flag: Name of refund flag column in training data.
    preprocessed_table_name: Name of preprocessed table after feature
      engineering.

  Returns:
    two-step model performance metrics dataframe.
  """
  two_step_prediction_performance_metrics_table_name = (
      constant.MODEL_PERFORMANCE_METRICS_TABLE_NAMES['2_step_model'].format(
          refund_flag_col=refund_flag,
          refund_value_col=refund_value,
          preprocessed_table_name=preprocessed_table_name,
      )
  )

  two_step_prediction_performance_metrics_df = utils.read_bq_table_to_df(
      project_id=project_id,
      bigquery_client=bigquery_client,
      dataset_id=dataset_id,
      table_name=two_step_prediction_performance_metrics_table_name,
  )
  return two_step_prediction_performance_metrics_df


def model_performance_metrics(
    project_id: str,
    bigquery_client: bigquery.Client,
    dataset_id: str,
    preprocessed_table_name: str,
    refund_value: str,
    refund_flag: str,
    regression_model_type: constant.SupportedModelTypes = constant.LinearBigQueryMLModelType.LINEAR_REGRESSION,
    binary_classifier_model_type: constant.SupportedModelTypes = constant.LinearBigQueryMLModelType.LOGISTIC_REGRESSION,
    is_two_step_model: bool = False,
) -> Mapping[str, pd.DataFrame]:
  """Get model performance metrics from BigQuery tables after training.

  Args:
    project_id: Google Cloud Platform project id.
    bigquery_client: Google Cloud BigQuery client.
    dataset_id: Google Cloud Platform dataset id where model performance metrics
      are stored.
    preprocessed_table_name: Name of preprocessed table after feature
      engineering.
    refund_value: Name of refund value column in training data.
    refund_flag: Name of refund flag column in training data.
    regression_model_type: Model type of regression model (e.g. Linear
      regression).
    binary_classifier_model_type: Model type of binary classification (e.g.
      Logistic regression).
    is_two_step_model: Whether to train a two-step model (binary classificaiton
      with refund flag (0/1) as label and regression model with refund value as
      label) instead of training a regression model directly with refund value
      as label.

  Returns:
    Mapping of model performance metrics dataframes.
  """
  _validate_regression_and_classification_model_types(
      binary_classifier_model_type, regression_model_type
  )
  performance_metrics_dfs = dict()
  performance_metrics_dfs['regression_model'] = (
      _regression_model_performance_metrics_data(
          project_id=project_id,
          bigquery_client=bigquery_client,
          dataset_id=dataset_id,
          refund_value=refund_value,
          regression_model_type=regression_model_type,
          preprocessed_table_name=preprocessed_table_name,
      )
  )
  if not is_two_step_model:
    return performance_metrics_dfs
  performance_metrics_dfs['classification_model'] = (
      _binary_classifier_model_performance_metrics_data(
          project_id=project_id,
          bigquery_client=bigquery_client,
          dataset_id=dataset_id,
          refund_flag=refund_flag,
          binary_classifier_model_type=binary_classifier_model_type,
          preprocessed_table_name=preprocessed_table_name,
      )
  )
  performance_metrics_dfs['two_step_model'] = (
      _two_step_model_performance_metrics_data(
          project_id=project_id,
          bigquery_client=bigquery_client,
          dataset_id=dataset_id,
          refund_value=refund_value,
          refund_flag=refund_flag,
          preprocessed_table_name=preprocessed_table_name,
      )
  )
  return performance_metrics_dfs


def model_prediction(
    project_id: str,
    dataset_id: str,
    bigquery_client: bigquery.Client,
    preprocessed_table_name: str,
    refund_value: str,
    refund_flag: str | None = None,
    regression_model_type: constant.SupportedModelTypes = constant.LinearBigQueryMLModelType.LINEAR_REGRESSION,
    use_prediction_pipeline: bool = False,
    is_two_step_model: bool = False,
) -> pd.DataFrame:
  """Get model prediction from BigQuery table.

  Args:
    project_id: Google Cloud Platform project id.
    dataset_id: Google Cloud Platform dataset id where model prediction is
      stored.
    bigquery_client: Google Cloud BigQuery client.
    preprocessed_table_name: Name of preprocessed table after feature
      engineering.
    refund_value: Name of refund value column in training data.
    refund_flag: Name of refund flag column in training data.
    regression_model_type: Model type of regression model (e.g. Linear
      regression).
    use_prediction_pipeline: Whether to get prediction from prediction pipeline.
    is_two_step_model: Whether to get prediction from two-step model or one-step
      regression model.

  Returns:
    Dataframe of model prediction.
  """
  _validate_regression_model_type(regression_model_type)
  if is_two_step_model:
    if refund_flag is None:
      raise ValueError(
          'refund_flag should be provided for 2 step classification and'
          ' regression models.'
      )
    key2 = '2_step_model'
  else:
    key2 = 'regression_model'
  if use_prediction_pipeline:
    key1 = 'prediction'
  else:
    key1 = 'training'
  prediction_table_name = constant.MODEL_PREDICTION_TABLE_NAMES[key1][key2]
  if refund_flag is not None:
    prediction_table_name = prediction_table_name.format(
        refund_flag_col=refund_flag,
        refund_value_col=refund_value,
        preprocessed_table_name=preprocessed_table_name,
    )
  else:
    prediction_table_name = prediction_table_name.format(
        refund_value_col=refund_value,
        regression_model_type=regression_model_type.value,
        preprocessed_table_name=preprocessed_table_name,
    )
  return utils.read_bq_table_to_df(
      project_id=project_id,
      bigquery_client=bigquery_client,
      dataset_id=dataset_id,
      table_name=prediction_table_name,
  )


def plot_predictions_actuals_distribution(
    prediction_df: pd.DataFrame,
    use_prediction_pipeline: bool = False,
    **kwargs,
) -> Mapping[str, pd.DataFrame]:
  """Show distribution of predictions vs actual labels.

  Args:
    prediction_df: Dataframe of with model prediction and also actual label data
      when use_prediction_pipeline is False.
    use_prediction_pipeline: Whether to show distribution for prediction
      generated from prediction pipeline. If false, show distribution for
      prediction generated from training phase.
    **kwargs: Additional arguments to pass to matplotlib.pyplot to create plots.

  Returns:
    Mapping of predictions and actuals distribution description.
  """
  predictions_actuals_dist = dict()
  _, axes = plt.subplots(1, 2, **kwargs)
  sns.histplot(prediction_df['prediction'], ax=axes[0], kde=True)
  axes[0].set_title('predictions distribution')
  predictions_actuals_dist['prediction'] = prediction_df[
      'prediction'
  ].describe()
  if not use_prediction_pipeline:
    sns.histplot(prediction_df['actual'], ax=axes[1], kde=True)
    axes[1].set_title('actuals distribution')
    predictions_actuals_dist['actual'] = prediction_df['actual'].describe()
  plt.tight_layout()
  plt.show()

  return predictions_actuals_dist


def plot_tier_level_product_return_actual_vs_prediction_comparison(
    training_comparison_df: pd.DataFrame,
    testing_comparison_df: pd.DataFrame,
    **kwargs,
) -> matplotlib.figure.Figure:
  """Plot tier level product return actual vs prediction comparison in bar chart.

  Args:
    training_comparison_df: Dataframe of tier level product return average
      actual vs prediction comparison for training data.
    testing_comparison_df: Dataframe of tier level product return average actual
      vs prediction comparison for testing data.
    **kwargs: Additional arguments to pass to matplotlib.pyplot to create plots.
  """
  _, ax = plt.subplots(2, 1)
  training_comparison_df.plot(
      x='tier',
      y=['tier_level_avg_actual', 'tier_level_avg_prediction'],
      kind='bar',
      ax=ax[0],
      **kwargs,
  )
  ax[0].set_title('Training Tier Level Comparison', fontsize='x-large')

  testing_comparison_df.plot(
      x='tier',
      y=['tier_level_avg_actual', 'tier_level_avg_prediction'],
      kind='bar',
      ax=ax[1],
      **kwargs,
  )
  ax[1].set_title('Testing Tier Level Comparison', fontsize='x-large')
  ax[1].set_xlabel(
      'Product Return Prediction Transaction Tier', fontsize='x-large'
  )
  plt.show()


def compare_and_plot_tier_level_avg_prediction(
    project_id: str,
    dataset_id: str,
    bigquery_client: bigquery.Client,
    preprocessed_table_name: str,
    refund_flag: str,
    refund_value: str,
    **kwargs,
) -> None:
  """Compare tier level average prediction vs actual by plotting.

  Args:
    project_id: Google Cloud Platform project id.
    dataset_id: Google Cloud Platform dataset id where model prediction and
      actual data are stored.
    bigquery_client: Google Cloud BigQuery client.
    preprocessed_table_name: Name of preprocessed table after feature
      engineering.
    refund_flag: Name of refund flag column in training data.
    refund_value: Name of refund value column in training data.
    **kwargs: Additional arguments to pass to matplotlib.pyplot to create
      comparison plots.
  """
  tier_level_avg_prediction_comparison_table_name = (
      constant.TIER_LEVEL_AVG_PREDICTION_COMPARISON_TABLE_NAME.format(
          project_id=project_id,
          dataset_id=dataset_id,
          refund_flag_col=refund_flag,
          refund_value_col=refund_value,
          preprocessed_table_name=preprocessed_table_name,
      )
  )
  tier_level_avg_prediction_comparison_df = utils.read_bq_table_to_df(
      project_id=project_id,
      bigquery_client=bigquery_client,
      dataset_id=dataset_id,
      table_name=tier_level_avg_prediction_comparison_table_name,
  )

  train_tier_avg_comparison = tier_level_avg_prediction_comparison_df.loc[
      tier_level_avg_prediction_comparison_df['train_test'] == 'train'
  ]
  test_tier_avg_comparison = tier_level_avg_prediction_comparison_df.loc[
      tier_level_avg_prediction_comparison_df['train_test'] == 'test'
  ]

  plot_tier_level_product_return_actual_vs_prediction_comparison(
      training_comparison_df=train_tier_avg_comparison,
      testing_comparison_df=test_tier_avg_comparison,
      **kwargs,
  )


def plot_feature_importance(
    feature_importance_df: pd.DataFrame,
    feature_col_name: str,
    feature_attribution_col_name: str,
    title_name: str | None = None,
) -> None:
  """Plot feature importance in bar chart.

  Args:
    feature_importance_df: Dataframe of feature importance.
    feature_col_name: Name of feature column in feature importance dataframe
      (e.g. feature).
    feature_attribution_col_name: Name of feature attribution column in feature
      importance dataframe (e.g. attribution).
    title_name: Title of the plot.
  """
  _, ax = plt.subplots(figsize=(16, 10))
  ax.set_title(title_name)
  ax.set_xlabel(feature_attribution_col_name)
  ax.set_ylabel(feature_col_name)
  feature_importance_df.sort_values(
      by=feature_attribution_col_name, inplace=True
  )
  feature_names = feature_importance_df[feature_col_name]
  attribution = feature_importance_df[feature_attribution_col_name]

  ax.barh(feature_names, attribution, align='center', height=0.5)
  plt.show()


def training_feature_importance(
    project_id: str,
    dataset_id: str,
    bigquery_client: bigquery.Client,
    preprocessed_table_name: str,
    regression_model_type: constant.SupportedModelTypes,
    binary_classifier_model_type: constant.SupportedModelTypes,
    refund_value: str,
    refund_flag: str,
    is_two_step_model: bool = False,
) -> Mapping[str, pd.DataFrame]:
  """Get feature importance from BigQuery tables after training.

  Args:
    project_id: Google Cloud Platform project id.
    dataset_id: Google Cloud Platform dataset id where feature importance data
      is stored.
    bigquery_client: Google Cloud BigQuery client.
    preprocessed_table_name: Name of preprocessed table after feature
      engineering.
    regression_model_type: Model type of regression model (e.g. Linear
      regression).
    binary_classifier_model_type: Model type of binary classification (e.g.
      Logistic regression).
    refund_value: Name of refund value column in training data.
    refund_flag: Name of refund flag column in training data.
    is_two_step_model: Whether the model is a two-step model (classification +
      regression).

  Returns:
    Mapping of feature importance dataframes with model type as key.
  """
  feature_importance_dfs_by_model_type = dict()

  regression_feature_importance_table_name = (
      constant.REGRESSION_FEATURE_IMPORTANCE_TABLE_NAME.format(
          refund_value_col=refund_value,
          regression_model_type=regression_model_type.value,
          preprocessed_table_name=preprocessed_table_name,
      )
  )

  regression_feature_importance_df = utils.read_bq_table_to_df(
      project_id=project_id,
      bigquery_client=bigquery_client,
      dataset_id=dataset_id,
      table_name=regression_feature_importance_table_name,
  )

  feature_importance_dfs_by_model_type[regression_model_type.value] = (
      regression_feature_importance_df
  )

  plot_feature_importance(
      feature_importance_df=regression_feature_importance_df,
      feature_col_name='feature',
      feature_attribution_col_name='attribution',
      title_name='regression model feature importance',
  )

  if is_two_step_model:

    classification_feature_importance_table_name = (
        constant.CLASSIFICATION_FEATURE_IMPORTANCE_TABLE_NAME.format(
            refund_flag_col=refund_flag,
            binary_classifier_model_type=binary_classifier_model_type.value,
            preprocessed_table_name=preprocessed_table_name,
        )
    )

    classification_feature_importance_df = utils.read_bq_table_to_df(
        project_id=project_id,
        bigquery_client=bigquery_client,
        dataset_id=dataset_id,
        table_name=classification_feature_importance_table_name,
    )
    plot_feature_importance(
        feature_importance_df=classification_feature_importance_df,
        feature_col_name='feature',
        feature_attribution_col_name='attribution',
        title_name='classification model feature importance',
    )

    feature_importance_dfs_by_model_type[binary_classifier_model_type.value] = (
        classification_feature_importance_df
    )

  return feature_importance_dfs_by_model_type
