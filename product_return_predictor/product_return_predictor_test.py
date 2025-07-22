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
from google.cloud import storage
import pandas as pd

from absl.testing import absltest
from product_return_predictor import constant
from product_return_predictor import data_cleaning_feature_selection
from product_return_predictor import model
from product_return_predictor import model_prediction_evaluation
from product_return_predictor import product_return_predictor
from product_return_predictor import utils


class ProductReturnPredictorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_bigquery_client = mock.create_autospec(bigquery.Client)
    self.mock_storage_client = mock.create_autospec(storage.Client)

    self.common_args = {
        "project_id": "test_project",
        "dataset_id": "test_dataset",
        "gcp_bq_client": self.mock_bigquery_client,
        "gcp_storage": self.mock_storage_client,
        "gcp_bucket_name": "test_bucket",
        "location": "US",
    }

  def test_post_init_ga4_data_true_success(self):
    predictor = product_return_predictor.ProductReturnPredictor(
        **self.common_args,
        use_ga4_data_for_feature_engineering=True,
        ga4_project_id="ga4_project",
        ga4_dataset_id="ga4_dataset",
    )
    self.assertEqual(
        predictor.refund_value_col, constant.TargetVariable.REFUND_VALUE.value
    )
    self.assertEqual(
        predictor.refund_flag_col, constant.TargetVariable.REFUND_FLAG.value
    )
    self.assertEqual(
        predictor.refund_proportion_col,
        constant.TargetVariable.REFUND_PROPORTION.value,
    )
    self.assertEqual(
        predictor.transaction_date_col,
        constant.IDColumnNames.TRANSACTION_DATE.value,
    )
    self.assertEqual(
        predictor.transaction_id_col,
        constant.IDColumnNames.TRANSACTION_ID.value,
    )
    self.assertIn(predictor.refund_value_col, predictor.numeric_labels)
    self.assertIn(predictor.refund_proportion_col, predictor.numeric_labels)
    self.assertIn(predictor.refund_flag_col, predictor.categorical_labels)
    self.assertIn(predictor.transaction_date_col, predictor.id_cols)
    self.assertIn(predictor.transaction_id_col, predictor.id_cols)

  def test_post_init_ga4_data_true_missing_ga4_project_id(self):
    with self.assertRaisesRegex(ValueError, "ga4_project_id must be specified"):
      product_return_predictor.ProductReturnPredictor(
          **self.common_args, use_ga4_data_for_feature_engineering=True
      )

  def test_post_init_ga4_data_true_missing_ga4_dataset_id(self):
    with self.assertRaisesRegex(ValueError, "ga4_dataset_id must be specified"):
      product_return_predictor.ProductReturnPredictor(
          **self.common_args,
          use_ga4_data_for_feature_engineering=True,
          ga4_project_id="ga4_project",
      )

  @mock.patch.object(utils, "check_bigquery_table_exists", autospec=True)
  def test_post_init_ga4_data_false_success(self, mock_check_table_exists):
    mock_check_table_exists.return_value = True
    predictor = product_return_predictor.ProductReturnPredictor(
        **self.common_args,
        use_ga4_data_for_feature_engineering=False,
        ml_training_table_name="training_table",
        ml_prediction_table_name="prediction_table",
        transaction_date_col="date_col",
        transaction_id_col="id_col",
        refund_value_col="value_col",
        refund_flag_col="flag_col",
        refund_proportion_col="prop_col",
    )
    self.assertEqual(predictor.ml_training_table_name, "training_table")
    self.assertEqual(predictor.transaction_date_col, "date_col")
    mock_check_table_exists.assert_called_once_with(
        self.mock_bigquery_client, "test_dataset", "training_table"
    )

  def test_post_init_ga4_data_false_raises_error_if_missing_transaction_date_col(
      self,
  ):
    with self.assertRaisesRegex(
        ValueError, "transaction_date_col must be specified"
    ):
      product_return_predictor.ProductReturnPredictor(
          **self.common_args, use_ga4_data_for_feature_engineering=False
      )

  def test_post_init_ga4_data_false_raises_error_if_missing_transaction_id_col(
      self,
  ):
    with self.assertRaisesRegex(
        ValueError, "transaction_id_col must be specified"
    ):
      product_return_predictor.ProductReturnPredictor(
          **self.common_args,
          use_ga4_data_for_feature_engineering=False,
          transaction_date_col="date_col",
      )

  def test_post_init_ga4_data_false_raises_error_if_missing_refund_value_col(
      self,
  ):
    with self.assertRaisesRegex(
        ValueError, "refund_value_col must be specified"
    ):
      product_return_predictor.ProductReturnPredictor(
          **self.common_args,
          use_ga4_data_for_feature_engineering=False,
          transaction_date_col="date_col",
          transaction_id_col="id_col",
      )

  def test_post_init_ga4_data_false_raises_error_if_missing_refund_flag_col(
      self,
  ):
    with self.assertRaisesRegex(
        ValueError, "refund_flag_col must be specified"
    ):
      product_return_predictor.ProductReturnPredictor(
          **self.common_args,
          use_ga4_data_for_feature_engineering=False,
          transaction_date_col="date_col",
          transaction_id_col="id_col",
          refund_value_col="value_col",
      )

  def test_post_init_ga4_data_false_raises_error_if_missing_refund_proportion_col(
      self,
  ):
    with self.assertRaisesRegex(
        ValueError, "refund_proportion_col must be specified"
    ):
      product_return_predictor.ProductReturnPredictor(
          **self.common_args,
          use_ga4_data_for_feature_engineering=False,
          transaction_date_col="date_col",
          transaction_id_col="id_col",
          refund_value_col="value_col",
          refund_flag_col="flag_col",
      )

  @mock.patch.object(utils, "check_bigquery_table_exists", autospec=True)
  def test_post_init_ga4_data_false_ml_training_raises_error_if_table_not_exist(
      self, mock_check_bigquery_table_exists
  ):
    mock_check_bigquery_table_exists.return_value = False
    with self.assertRaisesRegex(
        ValueError, "ml_training_table_name must exist"
    ):
      product_return_predictor.ProductReturnPredictor(
          **self.common_args,
          use_ga4_data_for_feature_engineering=False,
          ml_training_table_name="non_existent_table",
          transaction_date_col="date_col",
          transaction_id_col="id_col",
          refund_value_col="value_col",
          refund_flag_col="flag_col",
          refund_proportion_col="prop_col",
      )

  @mock.patch.object(utils, "read_file", autospec=True)
  @mock.patch.object(utils, "read_bq_table_to_df", autospec=True)
  @mock.patch.object(
      data_cleaning_feature_selection,
      "data_preprocessing_for_ml",
      autospec=True,
  )
  def test_data_processing_feature_engineering_ga4_data_with_correct_query_args(
      self,
      mock_data_preprocessing,
      mock_read_bq_table_to_df,
      mock_read_file,
  ):
    mock_read_bq_table_to_df.return_value = pd.DataFrame({"col1": [1, 2]})
    mock_read_file.return_value = (
        "SELECT * FROM {ga4_project_id}.{ga4_dataset_id}.table;"
    )
    predictor = product_return_predictor.ProductReturnPredictor(
        **self.common_args,
        use_ga4_data_for_feature_engineering=True,
        ga4_project_id="ga4_project",
        ga4_dataset_id="ga4_dataset",
    )
    predictor.data_processing_feature_engineering(
        data_pipeline_type=constant.DataPipelineType.TRAINING,
        recency_of_transaction_for_prediction_in_days=30,
        return_policy_window_in_days=90,
        recency_of_data_in_days=365,
    )

    self.assertLen(
        constant.GA4_DATA_PIPELINE_QUERY_TEMPLATES,
        mock_read_file.call_count,
    )
    self.mock_bigquery_client.query.assert_called()
    query_call_args = self.mock_bigquery_client.query.call_args[0][0]
    print(query_call_args)
    self.assertIn(
        "SELECT * FROM ga4_project.ga4_dataset.table", query_call_args
    )
    self.assertEqual(mock_read_bq_table_to_df.call_count, 2)
    mock_data_preprocessing.assert_called()
    self.assertEqual(mock_data_preprocessing.call_count, 2)

  @mock.patch.object(utils, "read_file", autospec=True)
  @mock.patch.object(utils, "read_bq_table_to_df", autospec=True)
  @mock.patch.object(
      data_cleaning_feature_selection,
      "data_preprocessing_for_ml",
      autospec=True,
  )
  def test_data_processing_feature_engineering_ga4_data_with_correct_args_for_data_preprocessing_for_ml(
      self,
      mock_data_preprocessing,
      mock_read_bq_table_to_df,
      mock_read_file,
  ):
    mock_read_bq_table_to_df.return_value = pd.DataFrame({"col1": [1, 2]})
    mock_read_file.return_value = (
        "SELECT * FROM {ga4_project_id}.{ga4_dataset_id}.table;"
    )
    predictor = product_return_predictor.ProductReturnPredictor(
        **self.common_args,
        use_ga4_data_for_feature_engineering=True,
        ga4_project_id="ga4_project",
        ga4_dataset_id="ga4_dataset",
    )
    predictor.data_processing_feature_engineering(
        data_pipeline_type=constant.DataPipelineType.TRAINING,
        recency_of_transaction_for_prediction_in_days=30,
        return_policy_window_in_days=90,
        recency_of_data_in_days=365,
    )
    call_args_existing = mock_data_preprocessing.call_args_list[0].kwargs
    self.assertFalse(call_args_existing["use_prediction_pipeline"])
    self.assertEqual(call_args_existing["df"].shape, (2, 1))
    self.assertEqual(
        call_args_existing["bigquery_client"], self.mock_bigquery_client
    )
    self.assertEqual(
        call_args_existing["gcp_storage_client"], self.mock_storage_client
    )
    self.assertEqual(call_args_existing["gcp_bucket_name"], "test_bucket")
    self.assertEqual(call_args_existing["dataset_id"], "test_dataset")
    self.assertEqual(
        call_args_existing["table_name"],
        "TRAINING_ml_ready_data_for_existing_customers",
    )
    self.assertListEqual(
        call_args_existing["id_cols"],
        [
            constant.IDColumnNames.TRANSACTION_DATE.value,
            constant.IDColumnNames.TRANSACTION_ID.value,
        ],
    )
    self.assertListEqual(
        call_args_existing["numeric_labels"],
        [
            constant.TargetVariable.REFUND_VALUE.value,
            constant.TargetVariable.REFUND_PROPORTION.value,
        ],
    )
    self.assertListEqual(
        call_args_existing["categorical_labels"],
        [constant.TargetVariable.REFUND_FLAG.value],
    )
    self.assertListEqual(
        call_args_existing["train_test_split_order_by_cols"],
        [constant.IDColumnNames.TRANSACTION_DATE.value],
    )
    self.assertEqual(call_args_existing["location"], "US")

    call_args_first_time = mock_data_preprocessing.call_args_list[1].kwargs
    self.assertEqual(
        call_args_first_time["table_name"],
        "TRAINING_ml_ready_data_for_first_time_purchase",
    )

  @mock.patch.object(model, "bigquery_ml_model_training", autospec=True)
  @mock.patch.object(
      model_prediction_evaluation, "model_performance_metrics", autospec=True
  )
  @mock.patch.object(
      model_prediction_evaluation, "model_prediction", autospec=True
  )
  @mock.patch.object(
      model_prediction_evaluation,
      "plot_predictions_actuals_distribution",
      autospec=True,
  )
  @mock.patch.object(
      model_prediction_evaluation,
      "compare_and_plot_tier_level_avg_prediction",
      autospec=True,
  )
  @mock.patch.object(
      model_prediction_evaluation, "training_feature_importance", autospec=True
  )
  def test_model_training_pipeline_evaluation_and_prediction_ga4_existing_customer_calls_correct_args_and_returns_expected_results(
      self,
      mock_training_feature_importance,
      mock_compare_and_plot,
      mock_plot_predictions,
      mock_model_prediction,
      mock_model_performance_metrics,
      mock_bigquery_ml_model_training,
  ):
    mock_training_feature_importance.return_value = {
        "feature_importance_df": pd.DataFrame()
    }
    mock_model_prediction.return_value = pd.DataFrame()
    mock_model_performance_metrics.return_value = {"metric_df": pd.DataFrame()}
    mock_plot_predictions.return_value = {"plot_data": pd.DataFrame()}

    predictor = product_return_predictor.ProductReturnPredictor(
        **self.common_args,
        use_ga4_data_for_feature_engineering=True,
        ga4_project_id="ga4_project",
        ga4_dataset_id="ga4_dataset",
    )
    (
        performance_metrics,
        model_prediction_df,
        predictions_actuals_distribution,
        feature_importance,
    ) = predictor.model_training_pipeline_evaluation_and_prediction(
        is_two_step_model=True,
        first_time_purchase=False,
        regression_model_type=constant.LinearBigQueryMLModelType.LINEAR_REGRESSION,
        binary_classifier_model_type=constant.LinearBigQueryMLModelType.LOGISTIC_REGRESSION,
    )

    expected_table_name = f"{constant.DataPipelineType.TRAINING.value}_ml_ready_data_for_existing_customers"

    mock_bigquery_ml_model_training.assert_called_once_with(
        preprocessed_table_name=expected_table_name,
        project_id=self.common_args["project_id"],
        dataset_id=self.common_args["dataset_id"],
        transaction_date_col=constant.IDColumnNames.TRANSACTION_DATE.value,
        transaction_id_col=constant.IDColumnNames.TRANSACTION_ID.value,
        num_tiers=10,
        bigquery_client=self.mock_bigquery_client,
        regression_model_type=constant.LinearBigQueryMLModelType.LINEAR_REGRESSION,
        binary_classifier_model_type=constant.LinearBigQueryMLModelType.LOGISTIC_REGRESSION,
        refund_value=constant.TargetVariable.REFUND_VALUE.value,
        refund_flag=constant.TargetVariable.REFUND_FLAG.value,
        is_two_step_model=True,
        probability_threshold_for_prediction=0.5,
        probability_threshold_for_model_evaluation=0.5,
        bqml_template_files_dir=constant.BQML_QUERY_TEMPLATE_FILES,
    )
    mock_model_performance_metrics.assert_called_once_with(
        preprocessed_table_name=expected_table_name,
        project_id=self.common_args["project_id"],
        bigquery_client=self.mock_bigquery_client,
        dataset_id=self.common_args["dataset_id"],
        is_two_step_model=True,
        refund_flag=constant.TargetVariable.REFUND_FLAG.value,
        refund_value=constant.TargetVariable.REFUND_VALUE.value,
        regression_model_type=constant.LinearBigQueryMLModelType.LINEAR_REGRESSION,
        binary_classifier_model_type=constant.LinearBigQueryMLModelType.LOGISTIC_REGRESSION,
    )
    mock_model_prediction.assert_called_once_with(
        project_id=self.common_args["project_id"],
        dataset_id=self.common_args["dataset_id"],
        bigquery_client=self.mock_bigquery_client,
        preprocessed_table_name=expected_table_name,
        refund_value=constant.TargetVariable.REFUND_VALUE.value,
        refund_flag=constant.TargetVariable.REFUND_FLAG.value,
        regression_model_type=constant.LinearBigQueryMLModelType.LINEAR_REGRESSION,
        use_prediction_pipeline=False,
        is_two_step_model=True,
    )
    mock_plot_predictions.assert_called_once()
    mock_compare_and_plot.assert_called_once_with(
        project_id=self.common_args["project_id"],
        dataset_id=self.common_args["dataset_id"],
        bigquery_client=self.mock_bigquery_client,
        preprocessed_table_name=expected_table_name,
        refund_flag=constant.TargetVariable.REFUND_FLAG.value,
        refund_value=constant.TargetVariable.REFUND_VALUE.value,
    )
    mock_training_feature_importance.assert_called_once_with(
        project_id=self.common_args["project_id"],
        dataset_id=self.common_args["dataset_id"],
        preprocessed_table_name=expected_table_name,
        bigquery_client=self.mock_bigquery_client,
        refund_value=constant.TargetVariable.REFUND_VALUE.value,
        refund_flag=constant.TargetVariable.REFUND_FLAG.value,
        regression_model_type=constant.LinearBigQueryMLModelType.LINEAR_REGRESSION,
        binary_classifier_model_type=constant.LinearBigQueryMLModelType.LOGISTIC_REGRESSION,
        is_two_step_model=True,
    )

    self.assertIsInstance(performance_metrics, dict)
    self.assertIsInstance(model_prediction_df, pd.DataFrame)
    self.assertIsInstance(predictions_actuals_distribution, dict)
    self.assertIsInstance(feature_importance, dict)

  @mock.patch.object(
      model_prediction_evaluation, "model_performance_metrics", autospec=True
  )
  @mock.patch.object(
      model_prediction_evaluation, "model_prediction", autospec=True
  )
  @mock.patch.object(
      model_prediction_evaluation,
      "plot_predictions_actuals_distribution",
      autospec=True,
  )
  @mock.patch.object(
      model_prediction_evaluation,
      "compare_and_plot_tier_level_avg_prediction",
      autospec=True,
  )
  @mock.patch.object(
      model_prediction_evaluation, "training_feature_importance", autospec=True
  )
  @mock.patch.object(model, "bigquery_ml_model_training", autospec=True)
  def test_model_training_pipeline_evaluation_and_prediction_ga4_first_time_purchase_calls_correct_args(
      self,
      mock_bigquery_ml_model_training,
      mock_compare_and_plot_tier_level_avg_prediction,
      mock_plot_predictions_actuals_distribution,
      mock_model_prediction,
      mock_model_performance_metrics,
      mock_training_feature_importance,
  ):
    mock_compare_and_plot_tier_level_avg_prediction.return_value = None
    mock_plot_predictions_actuals_distribution.return_value = None
    mock_training_feature_importance.return_value = None
    mock_model_prediction.return_value = None
    mock_model_performance_metrics.return_value = None
    predictor = product_return_predictor.ProductReturnPredictor(
        **self.common_args,
        use_ga4_data_for_feature_engineering=True,
        ga4_project_id="ga4_project",
        ga4_dataset_id="ga4_dataset",
    )
    predictor.model_training_pipeline_evaluation_and_prediction(
        is_two_step_model=True,
        first_time_purchase=True,
    )
    expected_table_name = f"{constant.DataPipelineType.TRAINING.value}_ml_ready_data_for_first_time_purchase"
    mock_bigquery_ml_model_training.assert_called_once_with(
        preprocessed_table_name=expected_table_name,
        project_id=self.common_args["project_id"],
        dataset_id=self.common_args["dataset_id"],
        transaction_date_col=constant.IDColumnNames.TRANSACTION_DATE.value,
        transaction_id_col=constant.IDColumnNames.TRANSACTION_ID.value,
        num_tiers=10,
        bigquery_client=self.mock_bigquery_client,
        regression_model_type=constant.LinearBigQueryMLModelType.LINEAR_REGRESSION,
        binary_classifier_model_type=constant.LinearBigQueryMLModelType.LOGISTIC_REGRESSION,
        refund_value=constant.TargetVariable.REFUND_VALUE.value,
        refund_flag=constant.TargetVariable.REFUND_FLAG.value,
        is_two_step_model=True,
        probability_threshold_for_prediction=0.5,
        probability_threshold_for_model_evaluation=0.5,
        bqml_template_files_dir=constant.BQML_QUERY_TEMPLATE_FILES,
    )

  @mock.patch.object(
      model_prediction_evaluation, "model_performance_metrics", autospec=True
  )
  @mock.patch.object(
      model_prediction_evaluation, "model_prediction", autospec=True
  )
  @mock.patch.object(
      model_prediction_evaluation,
      "plot_predictions_actuals_distribution",
      autospec=True,
  )
  @mock.patch.object(
      model_prediction_evaluation,
      "compare_and_plot_tier_level_avg_prediction",
      autospec=True,
  )
  @mock.patch.object(
      model_prediction_evaluation, "training_feature_importance", autospec=True
  )
  @mock.patch.object(model, "bigquery_ml_model_training", autospec=True)
  def test_model_training_pipeline_evaluation_and_prediction_non_ga4_calls_correct_args(
      self,
      mock_bigquery_ml_model_training,
      mock_compare_and_plot_tier_level_avg_prediction,
      mock_plot_predictions_actuals_distribution,
      mock_model_prediction,
      mock_model_performance_metrics,
      mock_training_feature_importance,
  ):
    mock_compare_and_plot_tier_level_avg_prediction.return_value = None
    mock_plot_predictions_actuals_distribution.return_value = None
    mock_training_feature_importance.return_value = None
    mock_model_performance_metrics.return_value = None
    mock_model_prediction.return_value = None
    predictor = product_return_predictor.ProductReturnPredictor(
        **self.common_args,
        use_ga4_data_for_feature_engineering=False,
        ml_training_table_name="user_training_table",
        transaction_date_col="date_col",
        transaction_id_col="id_col",
        refund_value_col="value_col",
        refund_flag_col="flag_col",
        refund_proportion_col="prop_col",
    )
    predictor.model_training_pipeline_evaluation_and_prediction(
        is_two_step_model=False
    )

    mock_bigquery_ml_model_training.assert_called_once_with(
        preprocessed_table_name="user_training_table",
        project_id=self.common_args["project_id"],
        dataset_id=self.common_args["dataset_id"],
        transaction_date_col="date_col",
        transaction_id_col="id_col",
        num_tiers=10,
        bigquery_client=self.mock_bigquery_client,
        regression_model_type=constant.LinearBigQueryMLModelType.LINEAR_REGRESSION,
        binary_classifier_model_type=constant.LinearBigQueryMLModelType.LOGISTIC_REGRESSION,
        refund_value="value_col",
        refund_flag="flag_col",
        is_two_step_model=False,
        probability_threshold_for_prediction=0.5,
        probability_threshold_for_model_evaluation=0.5,
        bqml_template_files_dir=constant.BQML_QUERY_TEMPLATE_FILES,
    )

  def test_model_training_pipeline_evaluation_and_prediction_if_missing_first_time_purchase_ga4_raises_error(
      self,
  ):
    predictor = product_return_predictor.ProductReturnPredictor(
        **self.common_args,
        use_ga4_data_for_feature_engineering=True,
        ga4_project_id="ga4_project",
        ga4_dataset_id="ga4_dataset",
    )
    with self.assertRaisesRegex(
        ValueError, "first_time_purchase boolean flag must be specified"
    ):
      predictor.model_training_pipeline_evaluation_and_prediction(
          is_two_step_model=True
      )

  @mock.patch.object(utils, "check_bigquery_table_exists", autospec=True)
  def test_model_training_pipeline_evaluation_and_prediction_if_missing_ml_training_table_name_non_ga4_returns_error(
      self, _
  ):
    predictor = product_return_predictor.ProductReturnPredictor(
        **self.common_args,
        use_ga4_data_for_feature_engineering=False,
        ml_training_table_name=None,
        transaction_date_col="date_col",
        transaction_id_col="id_col",
        refund_value_col="value_col",
        refund_flag_col="flag_col",
        refund_proportion_col="prop_col",
    )
    with self.assertRaisesRegex(
        ValueError, "ml_training_table_name must be specified"
    ):
      predictor.model_training_pipeline_evaluation_and_prediction(
          is_two_step_model=False
      )

  @mock.patch.object(model, "bigquery_ml_model_prediction", autospec=True)
  def test_prediction_pipeline_prediction_generation_ga4_existing_customer_calls_correct_args(
      self, mock_bigquery_ml_model_prediction
  ):
    predictor = product_return_predictor.ProductReturnPredictor(
        **self.common_args,
        use_ga4_data_for_feature_engineering=True,
        ga4_project_id="ga4_project",
        ga4_dataset_id="ga4_dataset",
    )
    predictor.prediction_pipeline_prediction_generation(
        is_two_step_model=True, first_time_purchase=False
    )
    expected_table_name = f"{constant.DataPipelineType.PREDICTION.value}_ml_ready_data_for_existing_customers"
    mock_bigquery_ml_model_prediction.assert_called_once_with(
        project_id=self.common_args["project_id"],
        dataset_id=self.common_args["dataset_id"],
        preprocessed_table_name=expected_table_name,
        bigquery_client=self.mock_bigquery_client,
        transaction_date_col=constant.IDColumnNames.TRANSACTION_DATE.value,
        transaction_id_col=constant.IDColumnNames.TRANSACTION_ID.value,
        regression_model_type=constant.LinearBigQueryMLModelType.LINEAR_REGRESSION,
        binary_classifier_model_type=constant.LinearBigQueryMLModelType.LOGISTIC_REGRESSION,
        refund_value=constant.TargetVariable.REFUND_VALUE.value,
        refund_flag=constant.TargetVariable.REFUND_FLAG.value,
        probability_threshold_for_prediction=0.5,
        is_two_step_model=True,
        bqml_template_files_dir=constant.BQML_QUERY_TEMPLATE_FILES,
        preprocessed_training_table_name=(
            "TRAINING_ml_ready_data_for_existing_customers"
        ),
    )

  @mock.patch.object(model, "bigquery_ml_model_prediction", autospec=True)
  def test_prediction_pipeline_prediction_generation_ga4_first_time_purchase_calls_correct_args(
      self, mock_bigquery_ml_model_prediction
  ):
    predictor = product_return_predictor.ProductReturnPredictor(
        **self.common_args,
        use_ga4_data_for_feature_engineering=True,
        ga4_project_id="ga4_project",
        ga4_dataset_id="ga4_dataset",
    )
    predictor.prediction_pipeline_prediction_generation(
        is_two_step_model=False, first_time_purchase=True
    )
    expected_table_name = f"{constant.DataPipelineType.PREDICTION.value}_ml_ready_data_for_first_time_purchase"
    mock_bigquery_ml_model_prediction.assert_called_once_with(
        project_id=self.common_args["project_id"],
        dataset_id=self.common_args["dataset_id"],
        preprocessed_table_name=expected_table_name,
        bigquery_client=self.mock_bigquery_client,
        transaction_date_col=constant.IDColumnNames.TRANSACTION_DATE.value,
        transaction_id_col=constant.IDColumnNames.TRANSACTION_ID.value,
        regression_model_type=constant.LinearBigQueryMLModelType.LINEAR_REGRESSION,
        binary_classifier_model_type=constant.LinearBigQueryMLModelType.LOGISTIC_REGRESSION,
        refund_value=constant.TargetVariable.REFUND_VALUE.value,
        refund_flag=constant.TargetVariable.REFUND_FLAG.value,
        probability_threshold_for_prediction=0.5,
        is_two_step_model=False,
        bqml_template_files_dir=constant.BQML_QUERY_TEMPLATE_FILES,
        preprocessed_training_table_name=(
            "TRAINING_ml_ready_data_for_first_time_purchase"
        ),
    )

  @mock.patch.object(model, "bigquery_ml_model_prediction", autospec=True)
  @mock.patch.object(utils, "check_bigquery_table_exists", autospec=True)
  def test_prediction_pipeline_prediction_generation_non_ga4_calls_correct_args(
      self, mock_check_bigquery_table_exists, mock_bigquery_ml_model_prediction
  ):
    mock_check_bigquery_table_exists.return_value = True
    predictor = product_return_predictor.ProductReturnPredictor(
        **self.common_args,
        use_ga4_data_for_feature_engineering=False,
        ml_prediction_table_name="user_prediction_table",
        transaction_date_col="date_col",
        transaction_id_col="id_col",
        refund_value_col="value_col",
        refund_flag_col="flag_col",
        refund_proportion_col="prop_col",
    )
    predictor.prediction_pipeline_prediction_generation(is_two_step_model=False)

    mock_bigquery_ml_model_prediction.assert_called_once_with(
        project_id=self.common_args["project_id"],
        dataset_id=self.common_args["dataset_id"],
        preprocessed_table_name="user_prediction_table",
        bigquery_client=self.mock_bigquery_client,
        transaction_date_col="date_col",
        transaction_id_col="id_col",
        regression_model_type=constant.LinearBigQueryMLModelType.LINEAR_REGRESSION,
        binary_classifier_model_type=constant.LinearBigQueryMLModelType.LOGISTIC_REGRESSION,
        refund_value="value_col",
        refund_flag="flag_col",
        probability_threshold_for_prediction=0.5,
        is_two_step_model=False,
        bqml_template_files_dir=constant.BQML_QUERY_TEMPLATE_FILES,
        preprocessed_training_table_name=None,
    )

  def test_prediction_pipeline_prediction_generation_if_missing_first_time_purchase_ga4_raises_error(
      self,
  ):
    predictor = product_return_predictor.ProductReturnPredictor(
        **self.common_args,
        use_ga4_data_for_feature_engineering=True,
        ga4_project_id="ga4_project",
        ga4_dataset_id="ga4_dataset",
    )
    with self.assertRaisesRegex(
        ValueError, "first_time_purchase boolean flag must be specified"
    ):
      predictor.prediction_pipeline_prediction_generation(
          is_two_step_model=True
      )

  @mock.patch.object(utils, "check_bigquery_table_exists", autospec=True)
  def test_prediction_pipeline_prediction_generation_if_missing_ml_prediction_table_name_non_ga4_returns_error(
      self, mock_check_bigquery_table_exists
  ):
    mock_check_bigquery_table_exists.return_value = True
    predictor = product_return_predictor.ProductReturnPredictor(
        **self.common_args,
        use_ga4_data_for_feature_engineering=False,
        ml_prediction_table_name=None,
        transaction_date_col="date_col",
        transaction_id_col="id_col",
        refund_value_col="value_col",
        refund_flag_col="flag_col",
        refund_proportion_col="prop_col",
    )
    with self.assertRaisesRegex(
        ValueError, "ml_prediction_table_name must be specified"
    ):
      predictor.prediction_pipeline_prediction_generation(
          is_two_step_model=False
      )


if __name__ == "__main__":
  absltest.main()
