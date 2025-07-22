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

from unittest import mock

from google.cloud import bigquery
import matplotlib
import pandas as pd

from absl.testing import absltest
from product_return_predictor.product_return_predictor import constant
from product_return_predictor.product_return_predictor import model_prediction_evaluation
from product_return_predictor.product_return_predictor import utils


_PREDICTION_DF = pd.DataFrame({
    'prediction': [10.5, 20.0, 15.75, 0, 5.2],
    'actual': [12.0, 18.5, 16.0, 2.5, 4.8],
})


class ModelPredictionEvaluationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_bigquery_client = mock.Mock(spec=bigquery.Client)

  @mock.patch.object(matplotlib.pyplot, 'show', autospec=True)
  def test_reasonable_input_for_prediction_vs_actual_returns_well_formed_descriptions_for_training_pipeline(
      self, _
  ):
    result = model_prediction_evaluation.plot_predictions_actuals_distribution(
        _PREDICTION_DF, use_prediction_pipeline=False
    )
    self.assertIn('prediction', result)
    self.assertIn('actual', result)
    self.assertIsInstance(result['prediction'], pd.Series)
    self.assertIsInstance(result['actual'], pd.Series)

  @mock.patch.object(matplotlib.pyplot, 'show', autospec=True)
  def test_reasonable_input_for_prediction_vs_actual_returns_well_formed_descriptions_for_prediction_pipeline(
      self, _
  ):
    result = model_prediction_evaluation.plot_predictions_actuals_distribution(
        _PREDICTION_DF[['prediction']], use_prediction_pipeline=True
    )

    self.assertIn('prediction', result)
    self.assertNotIn('actual', result)
    self.assertIsInstance(result['prediction'], pd.Series)

  @mock.patch.object(bigquery, 'Client', autospec=True)
  @mock.patch.object(utils, 'read_bq_table_to_df', autospec=True)
  def test_model_prediction_calls_bigquery_client_and_returns_expected_prediction_df(
      self, mock_read_bq_table_to_df, mock_bigquery_client
  ):
    mock_prediction_df = pd.DataFrame(
        {'transaction_id': [1, 2, 3], 'prediction': [100.5, 200.25, 150.75]}
    )
    mock_read_bq_table_to_df.return_value = mock_prediction_df
    prediction_df = model_prediction_evaluation.model_prediction(
        project_id='test_project',
        dataset_id='test_dataset',
        bigquery_client=mock_bigquery_client,
        refund_value=constant.TargetVariable.REFUND_VALUE.value,
        is_two_step_model=False,
        preprocessed_table_name='ml_ready_table',
    )
    pd.testing.assert_frame_equal(prediction_df, mock_prediction_df)
    expected_table_name = constant.MODEL_PREDICTION_TABLE_NAMES['training'][
        'regression_model'
    ].format(
        refund_value_col=constant.TargetVariable.REFUND_VALUE.value,
        regression_model_type=constant.LinearBigQueryMLModelType.LINEAR_REGRESSION.value,
        preprocessed_table_name='ml_ready_table',
    )
    mock_read_bq_table_to_df.assert_called_once_with(
        project_id='test_project',
        bigquery_client=mock_bigquery_client,
        dataset_id='test_dataset',
        table_name=expected_table_name,
    )

  @mock.patch.object(utils, 'read_bq_table_to_df', autospec=True)
  def test_model_prediction_two_step_model_missing_refund_flag_raises_error(
      self, mock_read_bq_table
  ):
    mock_read_bq_table.return_value = pd.DataFrame({'some_col': [1, 2, 3]})
    with self.assertRaisesRegex(
        ValueError,
        'refund_flag should be provided for 2 step classification and'
        ' regression models.',
    ) as context:
      model_prediction_evaluation.model_prediction(
          project_id='test_project',
          dataset_id='test_dataset',
          bigquery_client=self.mock_bigquery_client,
          refund_value=constant.TargetVariable.REFUND_VALUE.value,
          refund_flag=None,
          is_two_step_model=True,
          preprocessed_table_name='ml_ready_table',
      )
    self.assertIn(
        'refund_flag should be provided for 2 step classification and'
        ' regression models.',
        str(context.exception),
    )

  @mock.patch.object(bigquery, 'Client', autospec=True)
  @mock.patch.object(utils, 'read_bq_table_to_df', autospec=True)
  def test_model_performance_metrics_calls_bigquery_client_and_returns_expected_performance_metrics_df(
      self, mock_read_bq_table_to_df, mock_bigquery_client
  ):
    mock_performance_metrics_df = pd.DataFrame({
        'metric_name': ['mean_absolute_error', 'mean_squared_error'],
        'metric_value': [10.5, 110.25],
    })
    mock_read_bq_table_to_df.return_value = mock_performance_metrics_df
    performance_metrics_dfs = (
        model_prediction_evaluation.model_performance_metrics(
            project_id='test_project',
            bigquery_client=mock_bigquery_client,
            dataset_id='test_dataset',
            refund_value=constant.TargetVariable.REFUND_VALUE.value,
            refund_flag=constant.TargetVariable.REFUND_FLAG.value,
            is_two_step_model=False,
            preprocessed_table_name='ml_ready_table',
        )
    )
    self.assertIsInstance(performance_metrics_dfs, dict)
    self.assertIn('regression_model', performance_metrics_dfs)
    pd.testing.assert_frame_equal(
        performance_metrics_dfs['regression_model'], mock_performance_metrics_df
    )
    expected_table_name = constant.MODEL_PERFORMANCE_METRICS_TABLE_NAMES[
        'regression_model'
    ].format(
        refund_value_col=constant.TargetVariable.REFUND_VALUE.value,
        regression_model_type=constant.LinearBigQueryMLModelType.LINEAR_REGRESSION.value,
        preprocessed_table_name='ml_ready_table',
    )
    mock_read_bq_table_to_df.assert_called_once_with(
        project_id='test_project',
        bigquery_client=mock_bigquery_client,
        dataset_id='test_dataset',
        table_name=expected_table_name,
    )

  @mock.patch.object(
      model_prediction_evaluation,
      'plot_tier_level_product_return_actual_vs_prediction_comparison',
      autospec=True,
  )
  @mock.patch.object(utils, 'read_bq_table_to_df', autospec=True)
  def test_compare_and_plot_tier_level_avg_prediction_calls_plot_comparison_function(
      self, mock_read_bq_table_to_df, mock_plot_tier_level
  ):
    mock_comparison_df = pd.DataFrame({
        'tier': [1, 2, 3],
        'tier_level_avg_actual': [10.0, 12.0, 15.0],
        'tier_level_avg_prediction': [10.5, 12.3, 15.1],
        'train_test': ['train', 'train', 'test'],
    })
    mock_read_bq_table_to_df.return_value = mock_comparison_df

    model_prediction_evaluation.compare_and_plot_tier_level_avg_prediction(
        project_id='test_project',
        dataset_id='test_dataset',
        bigquery_client=mock.MagicMock(),
        refund_flag=constant.TargetVariable.REFUND_FLAG.value,
        refund_value=constant.TargetVariable.REFUND_VALUE.value,
        preprocessed_table_name='ml_ready_table',
    )

    mock_plot_tier_level.assert_called_once()

  @mock.patch.object(
      model_prediction_evaluation, 'plot_feature_importance', autospec=True
  )
  @mock.patch.object(utils, 'read_bq_table_to_df', autospec=True)
  def test_training_feature_importance_calls_plot_feature_importance_with_expected_args(
      self, mock_read_bq_table_to_df, mock_plot_feature_importance
  ):
    mock_feature_importance_df = pd.DataFrame(
        {'feature': ['feature1', 'feature2'], 'attribution': [0.6, 0.4]}
    )
    mock_read_bq_table_to_df.return_value = mock_feature_importance_df
    model_prediction_evaluation.training_feature_importance(
        project_id='test_project',
        dataset_id='test_dataset',
        bigquery_client=mock.MagicMock(),
        regression_model_type=constant.LinearBigQueryMLModelType.LINEAR_REGRESSION,
        binary_classifier_model_type=constant.LinearBigQueryMLModelType.LOGISTIC_REGRESSION,
        refund_value=constant.TargetVariable.REFUND_VALUE.value,
        refund_flag=constant.TargetVariable.REFUND_FLAG.value,
        is_two_step_model=False,
        preprocessed_table_name='ml_ready_table',
    )
    mock_plot_feature_importance.assert_called_once_with(
        feature_importance_df=mock_feature_importance_df,
        feature_col_name='feature',
        feature_attribution_col_name='attribution',
        title_name='regression model feature importance',
    )


if __name__ == '__main__':
  absltest.main()
