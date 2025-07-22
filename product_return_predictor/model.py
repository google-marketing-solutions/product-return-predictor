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

"""Module for building and training models."""

from collections.abc import Mapping
from typing import Optional

from absl import logging
from google.cloud import bigquery

from product_return_predictor import constant
from product_return_predictor import utils


def build_hyperparameter_tuning_options_for_bqml_linear_model(
    model_type: constant.LinearBigQueryMLModelType,
    l1_reg_lower_bound: int = 0,
    l1_reg_upper_bound: int = 10,
    l2_reg_lower_bound: int = 0,
    l2_reg_upper_bound: int = 10,
    num_trials: int = 10,
) -> str:
  """Build options statement for hyperparameter tuning in BQML for linear models.

  Args:
    model_type: Type of the model (e.g. linear or logistic regression).
    l1_reg_lower_bound: Lower bound for amount of L1 regularization applied.
    l1_reg_upper_bound: Upper bound for amount of L1 regularization applied.
    l2_reg_lower_bound: Lower bound for amount of L2 regularization applied.
    l2_reg_upper_bound: Upper bound for amount of L2 regularization applied.
    num_trials: Number of hyperparameter combinations to try during
      hyperparameter tuning.

  Returns:
    Text string used inside of the options statement of BQML.

  Raises:
    ValueError: If model type is not supported.
  """
  if model_type not in constant.BQML_LINEAR_MODEL_TYPES:
    raise ValueError(f"Model type {model_type.value} is not supported.")
  else:
    l1_reg_str = (
        f"L1_REG = HPARAM_RANGE({l1_reg_lower_bound}, {l1_reg_upper_bound})"
    )
    l2_reg_str = (
        f"L2_REG = HPARAM_RANGE({l2_reg_lower_bound}, {l2_reg_upper_bound})"
    )
    model_type_str = f"MODEL_TYPE = '{model_type.value}'"
    model_option_str = (
        f"{model_type_str}, {l1_reg_str}, {l2_reg_str}, NUM_TRIALS ="
        f" {num_trials}"
    )
    return model_option_str


def build_hyperparameter_tuning_options_for_bqml_dnn_model(
    model_type: constant.DNNBigQueryMLModelType,
    l1_reg_lower_bound: int = 0,
    l1_reg_upper_bound: int = 10,
    l2_reg_lower_bound: int = 0,
    l2_reg_upper_bound: int = 10,
    dnn_batch_size_lower_bound: int = 16,
    dnn_batch_size_upper_bound: int = 1024,
    learning_rate_lower_bound: float = 0.0,
    learning_rate_upper_bound: float = 1.0,
    dropout_lower_bound: float = 0.0,
    dropout_upper_bound: float = 0.8,
    dnn_optimizers: Optional[frozenset[constant.DNNOptimizer]] = frozenset(
        [constant.DNNOptimizer.ADAM]
    ),
    dnn_activation_functions: Optional[
        frozenset[constant.DNNActivationFunction]
    ] = frozenset([constant.DNNActivationFunction.RELU]),
    num_trials: Optional[int] = 10,
) -> str:
  """Build options statement for hyperparameter tuning in BQML for DNN models.

  Args:
    model_type: Type of the model (e.g. dnn regressor or classifier etc.)
    l1_reg_lower_bound: Lower bound for amount of L1 regularization applied.
    l1_reg_upper_bound: Upper bound for amount of L1 regularization applied.
    l2_reg_lower_bound: Lower bound for amount of L2 regularization applied.
    l2_reg_upper_bound: Upper bound for amount of L2 regularization applied.
    dnn_batch_size_lower_bound: Lower bound for the mini batch size of samples
      that are fed to the neural network.
    dnn_batch_size_upper_bound: Upper bound for the mini batch size of samples
      that are fed to the neural network.
    learning_rate_lower_bound: Lower bound for the learn rate for gradient
      descent.
    learning_rate_upper_bound: Upper bound for the learn rate for gradient
      descent.
    dropout_lower_bound: Lower bound for the dropout rate of units in the neural
      network.
    dropout_upper_bound: Upper bound for the dropout rate of units in the neural
      network.
    dnn_optimizers: List of optimizers to be used for the neural network.
    dnn_activation_functions: List of activation functions to be used for the
      neural network.
    num_trials: Number of hyperparameter combinations to try during
      hyperparameter tuning.

  Returns:
    Text string used inside of the options statement of BQML.

  Raises:
    ValueError: If model type is not supported.
  """
  if model_type not in constant.BQML_DNN_MODEL_TYPES:
    raise ValueError(f"Model type {model_type.value} is not supported.")
  else:
    dnn_activation_functions_str = ",".join(
        ["'{}'".format(func.value) for func in set(dnn_activation_functions)]
    )
    activation_functions_str = (
        f"ACTIVATION_FN = HPARAM_CANDIDATES([{dnn_activation_functions_str}])"
    )
    l1_reg_str = (
        f"L1_REG = HPARAM_RANGE({l1_reg_lower_bound}, {l1_reg_upper_bound})"
    )
    l2_reg_str = (
        f"L2_REG = HPARAM_RANGE({l2_reg_lower_bound}, {l2_reg_upper_bound})"
    )
    batch_size_str = (
        f"BATCH_SIZE = HPARAM_RANGE({dnn_batch_size_lower_bound},"
        f" {dnn_batch_size_upper_bound})"
    )
    dropout_str = (
        f"DROPOUT = HPARAM_RANGE({dropout_lower_bound}, {dropout_upper_bound})"
    )
    learning_rate_str = (
        f"LEARN_RATE = HPARAM_RANGE({learning_rate_lower_bound},"
        f" {learning_rate_upper_bound})"
    )
    dnn_optimizers_str = ",".join(
        ["'{}'".format(opt.value) for opt in list(dnn_optimizers)]
    )
    optimizer_str = f"OPTIMIZER = HPARAM_CANDIDATES([{dnn_optimizers_str}])"
    model_type_str = f"MODEL_TYPE = '{model_type.value}'"
    model_option_str = (
        f"{model_type_str}, "
        f"{activation_functions_str}, {l1_reg_str}, {l2_reg_str}, "
        f"{batch_size_str}, {dropout_str}, {learning_rate_str}, "
        f"{optimizer_str}, NUM_TRIALS = {num_trials}"
    )
    return model_option_str


def build_hyperparameter_tuning_options_for_bqml_boosted_tree_model(
    model_type: constant.BoostedTreeBigQueryMLModelType,
    l1_reg_lower_bound: int = 0,
    l1_reg_upper_bound: int = 10,
    l2_reg_lower_bound: int = 0,
    l2_reg_upper_bound: int = 10,
    learning_rate_lower_bound: float = 0.0,
    learning_rate_upper_bound: float = 1.0,
    dropout_lower_bound: float = 0.0,
    dropout_upper_bound: float = 0.8,
    boosted_tree_booster_type: constant.BoostedTreeBoosterType = constant.BoostedTreeBoosterType.GBTREE,
    boosted_tree_max_tree_depth_lower_bound: int = 1,
    boosted_tree_max_tree_depth_upper_bound: int = 10,
    boosted_tree_subsample_lower_bound: float = 0.0,
    boosted_tree_subsample_upper_bound: float = 1.0,
    boosted_tree_num_parallel_tree: frozenset[int] = frozenset([10]),
    boosted_tree_colsample_bytree_lower_bound: float = 0.0,
    boosted_tree_colsample_bytree_upper_bound: float = 1.0,
    num_trials: int = 10,
) -> str:
  """Build options statement for hyperparameter tuning in BQML for Boosted Tree models.

  Args:
    model_type: Type of the model (e.g. dnn regressor or classifier etc.)
    l1_reg_lower_bound: Lower bound for amount of L1 regularization applied.
    l1_reg_upper_bound: Upper bound for amount of L1 regularization applied.
    l2_reg_lower_bound: Lower bound for amount of L2 regularization applied.
    l2_reg_upper_bound: Upper bound for amount of L2 regularization applied.
    learning_rate_lower_bound: Lower bound for the learn rate for gradient
      descent.
    learning_rate_upper_bound: Upper bound for the learn rate for gradient
      descent.
    dropout_lower_bound: Lower bound for the dropout rate of units in the neural
      network.
    dropout_upper_bound: Upper bound for the dropout rate of units in the neural
      network.
    boosted_tree_booster_type: Type of boosted tree to be used. This should be
      either 'GBTREE' or 'DART'.
    boosted_tree_max_tree_depth_lower_bound: Lower bound for the maximum depth
      of the boosted tree.
    boosted_tree_max_tree_depth_upper_bound: Upper bound for the maximum depth
      of the boosted tree.
    boosted_tree_subsample_lower_bound: Lower bound for the subsample ratio of
      the training instances to be used for each boosting step.
    boosted_tree_subsample_upper_bound: Upper bound for the subsample ratio of
      the training instances to be used for each boosting step.
    boosted_tree_num_parallel_tree: List of number of trees to be trained in
      parallel.
    boosted_tree_colsample_bytree_lower_bound: Lower bound for the subsample
      ratio of the columns to be used for each boosting step.
    boosted_tree_colsample_bytree_upper_bound: Upper bound for the subsample
      ratio of the columns to be used for each boosting step.
    num_trials: Number of hyperparameter combinations to try during
      hyperparameter tuning.

  Returns:
    Text string used inside of the options statement of BQML.

  Raises:
    ValueError: If model type is not supported.
  """
  if model_type not in constant.BQML_BOOSTED_TREE_MODEL_TYPES:
    raise ValueError(f"Model type {model_type.value} is not supported.")
  else:
    booster_type_str = f"BOOSTER_TYPE = '{boosted_tree_booster_type.value}'"
    learning_rate_str = (
        f"LEARN_RATE = HPARAM_RANGE({learning_rate_lower_bound},"
        f" {learning_rate_upper_bound})"
    )
    l1_reg_str = (
        f"L1_REG = HPARAM_RANGE({l1_reg_lower_bound}, {l1_reg_upper_bound})"
    )
    l2_reg_str = (
        f"L2_REG = HPARAM_RANGE({l2_reg_lower_bound}, {l2_reg_upper_bound})"
    )
    dropout_str = (
        f"DROPOUT = HPARAM_RANGE({dropout_lower_bound}, {dropout_upper_bound})"
    )
    max_tree_depth_str = (
        "MAX_TREE_DEPTH ="
        f" HPARAM_RANGE({boosted_tree_max_tree_depth_lower_bound},"
        f" {boosted_tree_max_tree_depth_upper_bound})"
    )
    boosted_tree_subsample_str = (
        f"SUBSAMPLE = HPARAM_RANGE({boosted_tree_subsample_lower_bound},"
        f" {boosted_tree_subsample_upper_bound})"
    )
    boosted_tree_num_parallel_tree_str = ",".join(
        [str(num_tree) for num_tree in list(boosted_tree_num_parallel_tree)]
    )
    num_parallel_tree_str = (
        "NUM_PARALLEL_TREE ="
        f" HPARAM_CANDIDATES([{boosted_tree_num_parallel_tree_str}])"
    )
    boosted_tree_colsample_bytree_str = (
        "COLSAMPLE_BYTREE ="
        f" HPARAM_RANGE({boosted_tree_colsample_bytree_lower_bound},"
        f" {boosted_tree_colsample_bytree_upper_bound})"
    )
    model_type_str = f"MODEL_TYPE = '{model_type.value}'"
    model_option_str = (
        f"{model_type_str}, "
        f"{booster_type_str}, {learning_rate_str}, {l1_reg_str}, {l2_reg_str}, "
        f"{dropout_str}, {max_tree_depth_str}, {boosted_tree_subsample_str}, "
        f"{num_parallel_tree_str}, {boosted_tree_colsample_bytree_str}, "
        f"NUM_TRIALS = {num_trials}"
    )
    return model_option_str


def bigquery_ml_model_training(
    project_id: str,
    dataset_id: str,
    num_tiers: int,
    bigquery_client: bigquery.Client,
    transaction_date_col: str,
    transaction_id_col: str,
    preprocessed_table_name: str,
    bqml_template_files_dir: Mapping[
        str, str
    ] = constant.BQML_QUERY_TEMPLATE_FILES,
    regression_model_type: constant.SupportedModelTypes = constant.LinearBigQueryMLModelType.LINEAR_REGRESSION,
    binary_classifier_model_type: constant.SupportedModelTypes = constant.LinearBigQueryMLModelType.LOGISTIC_REGRESSION,
    refund_value: str = constant.TargetVariable.REFUND_VALUE.value,
    refund_flag: str | None = None,
    probability_threshold_for_prediction: float = 0.5,
    probability_threshold_for_model_evaluation: float = 0.5,
    is_two_step_model: bool = False,
    **kwargs,
) -> None:
  """Customize BigQuery ML setup and train product return predictive model.

  Args:
    project_id: Google Cloud Platform project id.
    dataset_id: Google Cloud Platform dataset id where training data is stored.
    num_tiers: Tier number for creating tier level average prediction.
    bigquery_client: Bigquery client for training BQML model.
    transaction_date_col: Column name of transaction date of the training data.
    transaction_id_col: Column name of transaction id of the training data.
    preprocessed_table_name: Name of the preprocessed table after data cleaning
      and feature engineering.
    bqml_template_files_dir: Directory containing BQML query template files.
    regression_model_type: Model type of regression model (e.g. Linear
      regression).
    binary_classifier_model_type: Model type of binary classification (e.g.
      Logistric regression).
    refund_value: Name of refund value column in training data.
    refund_flag: Name of refund flag column in training data.
    probability_threshold_for_prediction: When training classification model
      with refund flag as the label, prediction probability threshold to use to
      define 0 predicted refund value.
    probability_threshold_for_model_evaluation: When training classification
      model with refund flag as the label, prediction probability threshold to
      use for evaluating performance of the trained model.
    is_two_step_model: Whether to train a two-step model (binary classificaiton
      with refund flag (0/1) as label and regression model with refund value as
      label) instead of training a regression model directly with refund value
      as label.
    **kwargs: The keyword arguments containing hyperparameter tuning options.

  Raises:
    ValueError: If regression or classification model type is not supported or
    refund value is not provided for 2 step models. If the BQML template files
    directory does not contain the required template files.
  """
  if regression_model_type not in (
      constant.LinearBigQueryMLModelType.LINEAR_REGRESSION,
      constant.DNNBigQueryMLModelType.DNN_REGRESSOR,
      constant.BoostedTreeBigQueryMLModelType.BOOSTED_TREE_REGRESSOR,
  ):
    raise ValueError(
        f"Regression model type {regression_model_type} is not supported."
    )
  if "classification_regression_training" not in bqml_template_files_dir:
    raise ValueError(
        "BQML template files directory does not contain"
        " classification_regression_training template file."
    )
  if "regression_only_training" not in bqml_template_files_dir:
    raise ValueError(
        "BQML template files directory does not contain"
        " regression_only_training template file."
    )
  if (
      regression_model_type
      == constant.LinearBigQueryMLModelType.LINEAR_REGRESSION
  ):
    regression_model_config = (
        build_hyperparameter_tuning_options_for_bqml_linear_model(
            model_type=regression_model_type, **kwargs
        )
    )
  elif regression_model_type == constant.DNNBigQueryMLModelType.DNN_REGRESSOR:
    regression_model_config = (
        build_hyperparameter_tuning_options_for_bqml_dnn_model(
            model_type=regression_model_type, **kwargs
        )
    )
  else:
    regression_model_config = (
        build_hyperparameter_tuning_options_for_bqml_boosted_tree_model(
            model_type=regression_model_type, **kwargs
        )
    )

  if is_two_step_model:
    if binary_classifier_model_type not in (
        constant.LinearBigQueryMLModelType.LOGISTIC_REGRESSION,
        constant.DNNBigQueryMLModelType.DNN_CLASSIFIER,
        constant.BoostedTreeBigQueryMLModelType.BOOSTED_TREE_CLASSIFIER,
    ):
      raise ValueError(
          f"Binary classifier model type {binary_classifier_model_type} is not"
          " supported."
      )
    if refund_flag is None:
      raise ValueError(
          "refund_flag should be provided for 2 step classification and"
          " regression models."
      )
    if (
        binary_classifier_model_type
        == constant.LinearBigQueryMLModelType.LOGISTIC_REGRESSION
    ):
      classification_model_config = (
          build_hyperparameter_tuning_options_for_bqml_linear_model(
              model_type=binary_classifier_model_type, **kwargs
          )
      )
    elif (
        binary_classifier_model_type
        == constant.DNNBigQueryMLModelType.DNN_CLASSIFIER
    ):
      classification_model_config = (
          build_hyperparameter_tuning_options_for_bqml_dnn_model(
              model_type=binary_classifier_model_type, **kwargs
          )
      )
    else:
      classification_model_config = (
          build_hyperparameter_tuning_options_for_bqml_boosted_tree_model(
              model_type=binary_classifier_model_type, **kwargs
          )
      )
    query_template = utils.read_file(
        bqml_template_files_dir["classification_regression_training"]
    )
    query = query_template.format(
        project_id=project_id,
        dataset_id=dataset_id,
        transaction_date_col=transaction_date_col,
        transaction_id_col=transaction_id_col,
        refund_flag_col=refund_flag,
        refund_value_col=refund_value,
        binary_classifier_model_type=binary_classifier_model_type.value,
        classification_model_config=classification_model_config,
        regression_model_type=regression_model_type.value,
        regression_model_config=regression_model_config,
        num_tiers=num_tiers,
        probability_threshold_for_prediction=probability_threshold_for_prediction,
        probability_threshold_for_model_evaluation=probability_threshold_for_model_evaluation,
        preprocessed_table_name=preprocessed_table_name,
    )

  else:
    query_template = utils.read_file(
        bqml_template_files_dir["regression_only_training"]
    )
    query = query_template.format(
        project_id=project_id,
        dataset_id=dataset_id,
        transaction_date_col=transaction_date_col,
        transaction_id_col=transaction_id_col,
        refund_value_col=refund_value,
        regression_model_type=regression_model_type.value,
        regression_model_config=regression_model_config,
        num_tiers=num_tiers,
        preprocessed_table_name=preprocessed_table_name,
    )
  query_job = bigquery_client.query(query)
  logging.info(
      "BQML model has been trained with query job: %s", str(query_job.result())
  )


def bigquery_ml_model_prediction(
    project_id: str,
    dataset_id: str,
    preprocessed_table_name: str,
    preprocessed_training_table_name: str,
    bigquery_client: bigquery.Client,
    transaction_date_col: str,
    transaction_id_col: str,
    bqml_template_files_dir: Mapping[
        str, str
    ] = constant.BQML_QUERY_TEMPLATE_FILES,
    regression_model_type: (
        constant.LinearBigQueryMLModelType
        | constant.DNNBigQueryMLModelType
        | constant.BoostedTreeBigQueryMLModelType
    ) = constant.LinearBigQueryMLModelType.LINEAR_REGRESSION,
    binary_classifier_model_type: (
        constant.LinearBigQueryMLModelType
        | constant.DNNBigQueryMLModelType
        | constant.BoostedTreeBigQueryMLModelType
    ) = constant.LinearBigQueryMLModelType.LOGISTIC_REGRESSION,
    refund_value: str = constant.TargetVariable.REFUND_VALUE.value,
    refund_flag: str | None = None,
    probability_threshold_for_prediction: float = 0.5,
    is_two_step_model: bool = False,
) -> None:
  """Predict product return from trained BQML model.

  Args:
    project_id: Google Cloud Platform project id.
    dataset_id: Google Cloud Platform dataset id where training data is stored.
    preprocessed_table_name: Name of the preprocessed table after data cleaning
      and feature engineering.
    preprocessed_training_table_name: Name of the preprocessed training table
      that was used for training the model.
    bigquery_client: Bigquery client for training BQML model.
    transaction_date_col: Transaction date column name of the training data.
    transaction_id_col: Transaction id column name of the training data.
    bqml_template_files_dir: Directory mapping with model type (i.e.
      regression_only_training, regression_only_prediction,
      classification_regression_training and
      classification_regression_prediction) as the key and and corresponding
      model prediction BQML query template files as the value.
    regression_model_type: Model type of regression model (e.g. Linear
      regression).
    binary_classifier_model_type: Model type of binary classification (e.g.
      Logistric regression).
    refund_value: Refund value column name in training data.
    refund_flag: Refund flag column name in training data.
    probability_threshold_for_prediction: Prediction probability threshold to
      use when using classification model to predict whether a transaction will
      be refunded. If probability_threshold_for_prediction is 0.5, then
      transactions with probability greater than and equal to 0.5 will be
      predicted as refunded.
    is_two_step_model: Whether to train a two-step model (binary classificaiton
      with refund flag (0/1) as label and the regression model with refund value
      as label) instead of training a single regression model directly with
      refund value as label.

  Raises:
    ValueError: If regression or classification model type is not supported or
    refund value is not provided for 2 step models.
  """
  if regression_model_type not in constant.BQML_REGRESSION_MODEL_TYPES:
    raise ValueError(
        f"Regression model type {regression_model_type} is not supported."
    )
  if (
      is_two_step_model
      and binary_classifier_model_type
      not in constant.BQML_CLASSIFICATION_MODEL_TYPES
  ):
    raise ValueError(
        f"Binary classifier model type {binary_classifier_model_type} is not"
        " supported for two step model."
    )
  if is_two_step_model and refund_flag is None:
    raise ValueError(
        "refund_flag should be provided for 2 step classification and"
        " regression models."
    )
  if is_two_step_model:
    query = utils.read_file(
        bqml_template_files_dir["classification_regression_prediction"]
    )
    destination_table = constant.MODEL_PREDICTION_TABLE_NAMES["prediction"][
        "2_step_model"
    ].format(
        preprocessed_table_name=preprocessed_table_name,
        refund_value_col=refund_value,
        refund_flag_col=refund_flag,
    )
  else:
    query = utils.read_file(
        bqml_template_files_dir["regression_only_prediction"]
    )
    destination_table = constant.MODEL_PREDICTION_TABLE_NAMES["prediction"][
        "regression_model"
    ].format(
        preprocessed_table_name=preprocessed_table_name,
        refund_value_col=refund_value,
        regression_model_type=regression_model_type.value,
    )
  formatted_query = query.format(
      project_id=project_id,
      dataset_id=dataset_id,
      transaction_date_col=transaction_date_col,
      transaction_id_col=transaction_id_col,
      refund_flag_col=(refund_flag if refund_flag else None),
      refund_value_col=refund_value,
      binary_classifier_model_type=binary_classifier_model_type.value,
      regression_model_type=regression_model_type.value,
      probability_threshold_for_prediction=probability_threshold_for_prediction,
      preprocessed_table_name=preprocessed_table_name,
      preprocessed_training_table_name=preprocessed_training_table_name,
  )
  query_job = bigquery_client.query(formatted_query)
  prediction_table_name = f"{project_id}.{dataset_id}.{destination_table}"
  logging.info(
      "BQML model prediction is done with query job: %s, destination table"
      " name: %s",
      str(query_job.result()),
      prediction_table_name,
  )
