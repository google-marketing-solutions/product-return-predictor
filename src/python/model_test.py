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
from google.cloud.bigquery import job
import immutabledict

from absl.testing import absltest
from product_return_predictor.src.python import constant
from product_return_predictor.src.python import model


_INPUT_HYPERPARAMETERS_SETUP_FOR_DNN_MODEL = {
    "model_type": constant.DNNBigQueryMLModelType.DNN_CLASSIFIER,
    "l1_reg_lower_bound": 0,
    "l1_reg_upper_bound": 10,
    "l2_reg_lower_bound": 0,
    "l2_reg_upper_bound": 10,
    "dnn_batch_size_lower_bound": 16,
    "dnn_batch_size_upper_bound": 32,
    "learning_rate_lower_bound": 0.0,
    "learning_rate_upper_bound": 0.5,
    "dropout_lower_bound": 0.1,
    "dropout_upper_bound": 0.3,
    "num_trials": 20,
}


_INPUT_HYPERPARAMETERS_SETUP_FOR_BOOSTED_TREE_MODEL = {
    "model_type": (
        constant.BoostedTreeBigQueryMLModelType.BOOSTED_TREE_CLASSIFIER
    ),
    "l1_reg_lower_bound": 0,
    "l1_reg_upper_bound": 10,
    "l2_reg_lower_bound": 0,
    "l2_reg_upper_bound": 10,
    "learning_rate_lower_bound": 0.0,
    "learning_rate_upper_bound": 0.5,
    "dropout_lower_bound": 0.1,
    "dropout_upper_bound": 0.3,
    "boosted_tree_booster_type": constant.BoostedTreeBoosterType.GBTREE,
    "boosted_tree_max_tree_depth_lower_bound": 1,
    "boosted_tree_max_tree_depth_upper_bound": 10,
    "boosted_tree_subsample_lower_bound": 0.0,
    "boosted_tree_subsample_upper_bound": 1.0,
    "boosted_tree_num_parallel_tree": frozenset([10]),
    "boosted_tree_colsample_bytree_lower_bound": 0.0,
    "boosted_tree_colsample_bytree_upper_bound": 1.0,
    "num_trials": 20,
}


_INPUT_HYPERPARAMETERS_SETUP_FOR_LINEAR_MODEL = {
    "model_type": constant.LinearBigQueryMLModelType.LINEAR_REGRESSION,
    "l1_reg_lower_bound": 0,
    "l1_reg_upper_bound": 10,
    "l2_reg_lower_bound": 0,
    "l2_reg_upper_bound": 10,
    "num_trials": 20,
}

_BQML_QUERY_TEMPLATE_DIR_FOR_TESTING = "third_party/professional_services/solutions/product_return_predictor/src/sql/bigquery_ml/sql_template"

_BQML_QUERY_TEMPLATE_FILES_FOR_TESTING = immutabledict.immutabledict({
    "regression_only_training": (
        f"{_BQML_QUERY_TEMPLATE_DIR_FOR_TESTING}/regression_model_training_pipeline.sql"
    ),
    "regression_only_prediction": (
        f"{_BQML_QUERY_TEMPLATE_DIR_FOR_TESTING}/regression_model_prediction_pipeline.sql"
    ),
    "classification_regression_training": (
        f"{_BQML_QUERY_TEMPLATE_DIR_FOR_TESTING}/classification_regression_2_steps_model_training_pipeline.sql"
    ),
    "classification_regression_prediction": (
        f"{_BQML_QUERY_TEMPLATE_DIR_FOR_TESTING}/classification_regression_2_steps_model_prediction_pipeline.sql"
    ),
})

_BQML_QUERY_TEMPLATE_FILES_FOR_TESTING_WITH_MISSING_TEMPLATE_FILE = immutabledict.immutabledict({
    "regression_only_training": (
        f"{_BQML_QUERY_TEMPLATE_DIR_FOR_TESTING}/regression_model_training_pipeline.sql"
    ),
    "regression_only_prediction": (
        f"{_BQML_QUERY_TEMPLATE_DIR_FOR_TESTING}/regression_model_prediction_pipeline.sql"
    ),
})


class ModelTest(absltest.TestCase):

  def test_build_hyperparameter_tuning_options_for_bqml_dnn_model(self):
    hyperparameter_tuning_options = (
        model.build_hyperparameter_tuning_options_for_bqml_dnn_model(
            **_INPUT_HYPERPARAMETERS_SETUP_FOR_DNN_MODEL
        )
    )
    self.assertEqual(
        hyperparameter_tuning_options,
        "MODEL_TYPE = 'DNN_CLASSIFIER', "
        "ACTIVATION_FN = HPARAM_CANDIDATES(['RELU']), L1_REG = HPARAM_RANGE(0,"
        " 10), L2_REG = HPARAM_RANGE(0, 10), BATCH_SIZE = HPARAM_RANGE(16, 32),"
        " DROPOUT = HPARAM_RANGE(0.1, 0.3), LEARN_RATE = HPARAM_RANGE(0.0,"
        " 0.5), OPTIMIZER = HPARAM_CANDIDATES(['ADAM']), NUM_TRIALS = 20",
    )

  def test_build_hyperparameter_tuning_options_for_bqml_boosted_tree_model(
      self,
  ):
    hyperparameter_tuning_options = (
        model.build_hyperparameter_tuning_options_for_bqml_boosted_tree_model(
            **_INPUT_HYPERPARAMETERS_SETUP_FOR_BOOSTED_TREE_MODEL
        )
    )
    self.assertEqual(
        hyperparameter_tuning_options,
        "MODEL_TYPE = 'BOOSTED_TREE_CLASSIFIER', "
        "BOOSTER_TYPE = 'GBTREE', LEARN_RATE = HPARAM_RANGE(0.0, 0.5), L1_REG ="
        " HPARAM_RANGE(0, 10), L2_REG = HPARAM_RANGE(0, 10), DROPOUT ="
        " HPARAM_RANGE(0.1, 0.3), MAX_TREE_DEPTH = HPARAM_RANGE(1, 10),"
        " SUBSAMPLE = HPARAM_RANGE(0.0, 1.0), NUM_PARALLEL_TREE ="
        " HPARAM_CANDIDATES([10]), COLSAMPLE_BYTREE = HPARAM_RANGE(0.0, 1.0),"
        " NUM_TRIALS = 20",
    )

  def test_build_hyperparameter_tuning_options_for_bqml_linear_model(self):
    hyperparameter_tuning_options = (
        model.build_hyperparameter_tuning_options_for_bqml_linear_model(
            **_INPUT_HYPERPARAMETERS_SETUP_FOR_LINEAR_MODEL
        )
    )
    self.assertEqual(
        hyperparameter_tuning_options,
        "MODEL_TYPE = 'LINEAR_REG', "
        "L1_REG = HPARAM_RANGE(0, 10), L2_REG = HPARAM_RANGE(0, 10), NUM_TRIALS"
        " = 20",
    )

  @mock.patch("google.cloud.bigquery.Client")
  def test_bigquery_ml_model_training_missing_args_for_two_step(
      self, mock_bigquery_client
  ):
    mock_client = mock_bigquery_client.return_value

    with self.assertRaises(ValueError) as context:
      model.bigquery_ml_model_training(
          project_id="test_project",
          dataset_id="test_dataset",
          transaction_date_col="transaction_date",
          transaction_id_col="transaction_id",
          preprocessed_table_name="ml_ready_table",
          num_tiers=3,
          bigquery_client=mock_client,
          binary_classifier_model_type=constant.DNNBigQueryMLModelType.DNN_CLASSIFIER,
          regression_model_type=constant.LinearBigQueryMLModelType.LINEAR_REGRESSION,
          is_two_step_model=True,
      )
    self.assertIn(
        "refund_flag should be provided for 2 step classification and"
        " regression models.",
        str(context.exception),
    )

  def test_bigquery_ml_model_training_for_two_steps_model(self):
    mock_bigquery_client = mock.create_autospec(bigquery.Client, instance=True)
    mock_bigquery_job = mock.create_autospec(job.QueryJob, instance=True)

    mock_bigquery_client.query.return_value = mock_bigquery_job
    mock_bigquery_job.result.return_value = "Result"
    model.bigquery_ml_model_training(
        project_id="test_project",
        dataset_id="test_dataset",
        transaction_date_col="transaction_date",
        transaction_id_col="transaction_id",
        preprocessed_table_name="ml_ready_table",
        num_tiers=3,
        bigquery_client=mock_bigquery_client,
        bqml_template_files_dir=_BQML_QUERY_TEMPLATE_FILES_FOR_TESTING,
        is_two_step_model=True,
        binary_classifier_model_type=constant.LinearBigQueryMLModelType.LOGISTIC_REGRESSION,
        regression_model_type=(
            constant.LinearBigQueryMLModelType.LINEAR_REGRESSION
        ),
        refund_flag=constant.TargetVariable.REFUND_FLAG,
        refund_value=constant.TargetVariable.REFUND_VALUE,
    )
    call_args = mock_bigquery_client.query.call_args.args[0]
    self.assertIn("CREATE OR REPLACE MODEL", call_args)
    self.assertIn("MODEL_TYPE = 'LOGISTIC_REG'", call_args)
    self.assertIn("binary_classifier_LOGISTIC_REG", call_args)
    self.assertIn("MODEL_TYPE = 'LINEAR_REG'", call_args)
    self.assertIn("regressor_LINEAR_REG", call_args)

  def test_bigquery_ml_model_training_for_regression_only_model(self):
    mock_bigquery_client = mock.create_autospec(bigquery.Client, instance=True)
    mock_bigquery_job = mock.create_autospec(job.QueryJob, instance=True)

    mock_bigquery_client.query.return_value = mock_bigquery_job
    mock_bigquery_job.result.return_value = "Result"
    model.bigquery_ml_model_training(
        project_id="test_project",
        dataset_id="test_dataset",
        transaction_date_col="transaction_date",
        transaction_id_col="transaction_id",
        preprocessed_table_name="ml_ready_table",
        num_tiers=3,
        bigquery_client=mock_bigquery_client,
        bqml_template_files_dir=_BQML_QUERY_TEMPLATE_FILES_FOR_TESTING,
        is_two_step_model=False,
        regression_model_type=(
            constant.LinearBigQueryMLModelType.LINEAR_REGRESSION
        ),
        refund_value=constant.TargetVariable.REFUND_VALUE,
    )
    call_args = mock_bigquery_client.query.call_args.args[0]
    self.assertIn("CREATE OR REPLACE MODEL", call_args)
    self.assertIn("MODEL_TYPE = 'LINEAR_REG'", call_args)
    self.assertIn("regressor_LINEAR_REG", call_args)

  def test_bigquery_ml_model_training_with_missing_template_file(self):
    mock_bigquery_client = mock.create_autospec(bigquery.Client, instance=True)
    mock_bigquery_job = mock.create_autospec(job.QueryJob, instance=True)

    mock_bigquery_client.query.return_value = mock_bigquery_job
    mock_bigquery_job.result.return_value = "Result"
    with self.assertRaises(ValueError):
      model.bigquery_ml_model_training(
          project_id="test_project",
          dataset_id="test_dataset",
          transaction_date_col="transaction_date",
          transaction_id_col="transaction_id",
          preprocessed_table_name="ml_ready_table",
          num_tiers=3,
          bigquery_client=mock_bigquery_client,
          bqml_template_files_dir=_BQML_QUERY_TEMPLATE_FILES_FOR_TESTING_WITH_MISSING_TEMPLATE_FILE,
          is_two_step_model=False,
          regression_model_type=(
              constant.LinearBigQueryMLModelType.LINEAR_REGRESSION
          ),
          refund_value=constant.TargetVariable.REFUND_VALUE,
      )

  def test_bigquery_ml_model_prediction_for_two_steps_model_calls_expected_bqml_query(
      self,
  ):
    mock_bigquery_client = mock.create_autospec(bigquery.Client, instance=True)
    mock_bigquery_job = mock.create_autospec(job.QueryJob, instance=True)

    mock_bigquery_client.query.return_value = mock_bigquery_job
    mock_bigquery_job.result.return_value = "Result"
    model.bigquery_ml_model_prediction(
        project_id="test_project",
        dataset_id="test_dataset",
        transaction_date_col="transaction_date",
        transaction_id_col="transaction_id",
        preprocessed_table_name="ml_ready_table",
        bigquery_client=mock_bigquery_client,
        bqml_template_files_dir=_BQML_QUERY_TEMPLATE_FILES_FOR_TESTING,
        is_two_step_model=True,
        binary_classifier_model_type=constant.LinearBigQueryMLModelType.LOGISTIC_REGRESSION,
        regression_model_type=(
            constant.LinearBigQueryMLModelType.LINEAR_REGRESSION
        ),
        refund_flag=constant.TargetVariable.REFUND_FLAG,
        refund_value=constant.TargetVariable.REFUND_VALUE,
    )
    call_args = mock_bigquery_client.query.call_args.args[0]
    self.assertIn("ML.PREDICT", call_args)
    self.assertIn("refund_flag_binary_classifier_LOGISTIC_REG", call_args)
    self.assertIn("refund_value_regressor_LINEAR_REG", call_args)

  def test_bigquery_ml_model_prediction_for_regression_only_model_calls_expected_bqml_query(
      self,
  ):
    mock_bigquery_client = mock.create_autospec(bigquery.Client, instance=True)
    mock_bigquery_job = mock.create_autospec(job.QueryJob, instance=True)

    mock_bigquery_client.query.return_value = mock_bigquery_job
    mock_bigquery_job.result.return_value = "Result"
    model.bigquery_ml_model_prediction(
        project_id="test_project",
        dataset_id="test_dataset",
        transaction_date_col="transaction_date",
        transaction_id_col="transaction_id",
        preprocessed_table_name="ml_ready_table",
        bigquery_client=mock_bigquery_client,
        bqml_template_files_dir=_BQML_QUERY_TEMPLATE_FILES_FOR_TESTING,
        is_two_step_model=False,
        regression_model_type=(
            constant.LinearBigQueryMLModelType.LINEAR_REGRESSION
        ),
        refund_value=constant.TargetVariable.REFUND_VALUE,
    )
    call_args = mock_bigquery_client.query.call_args.args[0]
    self.assertIn("ML.PREDICT", call_args)
    self.assertIn("refund_value_regressor_LINEAR_REG", call_args)
    self.assertNotIn("refund_flag_binary_classifier_LOGISTIC_REG", call_args)

  def test_bigquery_ml_model_prediction_raises_value_error_for_missing_refund_flag_in_two_step_model(
      self,
  ):
    mock_bigquery_client = mock.create_autospec(bigquery.Client, instance=True)
    mock_bigquery_job = mock.create_autospec(job.QueryJob, instance=True)

    mock_bigquery_client.query.return_value = mock_bigquery_job
    mock_bigquery_job.result.return_value = "Result"
    with self.assertRaisesRegex(
        ValueError,
        "refund_flag should be provided for 2 step classification and"
        " regression models.",
    ):
      model.bigquery_ml_model_prediction(
          project_id="test_project",
          dataset_id="test_dataset",
          transaction_date_col="transaction_date",
          transaction_id_col="transaction_id",
          preprocessed_table_name="ml_ready_table",
          bigquery_client=mock_bigquery_client,
          bqml_template_files_dir=_BQML_QUERY_TEMPLATE_FILES_FOR_TESTING,
          is_two_step_model=True,
          regression_model_type=constant.BoostedTreeBigQueryMLModelType.BOOSTED_TREE_REGRESSOR,
          refund_value=constant.TargetVariable.REFUND_VALUE,
      )


if __name__ == "__main__":
  absltest.main()
