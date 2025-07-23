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

"""Product Return Predictor Class."""

from collections.abc import Mapping
import dataclasses

from google.cloud import bigquery
from google.cloud import storage
import pandas as pd

from product_return_predictor import constant
from product_return_predictor import data_cleaning_feature_selection
from product_return_predictor import model
from product_return_predictor import model_prediction_evaluation
from product_return_predictor import utils


@dataclasses.dataclass()
class ProductReturnPredictor:
  """Product Return Predictor pipeline.

  Contains members and member methods for the predictive modeling pipeline.
  Integrates all data processing with GA4 data, feature engineering, model
  training and prediction process on GCP within one module.

  Attributes:
    project_id: The GCP project ID for running the pipeline.
    dataset_id: The GCP dataset ID for running the pipeline.
    ga4_project_id: The GCP project ID for GA4 data. This is optional. When the
      user wants to use GA4 data for feature engineering, this field is
      required.
    ga4_dataset_id: The GCP dataset ID for GA4 data. This is optional. When the
      user wants to use GA4 data for feature engineering, this field is
      required.
    use_ga4_data_for_feature_engineering: Whether to use GA4 data for feature
      engineering. If True, ga4_project_id and ga4_dataset_id must be provided.
    gcp_bq_client: The GCP BigQuery client.
    gcp_storage: The GCP Storage client.
    gcp_bucket_name: The GCP storage bucket name.
    ml_training_table_name: The table name for training the model. This is
      optional. When the user does not use GA4 data for feature engineering,
      this field is required.
    ml_prediction_table_name: The table name for prediction. This is optional.
      When the user does not use GA4 data for feature engineering, this field is
      required.
    location: The location for the GCP resources.
    refund_value_col: Column name for refund value (target variable). This is
      optional. When the user does not use GA4 data for feature engineering,
      this field is required.
    refund_flag_col: Column name for refund flag (target variable). This is
      optional. When the user does not use GA4 data for feature engineering,
      this field is required.
    refund_proportion_col: Column name for refund proportion (target variable).
      This is optional. When the user does not use GA4 data for feature
      engineering, this field is required.
    train_test_split_test_size_proportion: Proportion of the data to be used as
      the test set.
    invalid_value_threshold_for_row_removal: Threshold for removing rows with
      invalid values.
    invalid_value_threshold_for_column_removal: Threshold for removing columns
      with invalid values.
    min_correlation_threshold_with_numeric_labels_for_feature_reduction:
      Threshold for reducing the number of features using correlation with
      numeric labels.
    transaction_date_col: Column name for transaction date. This is optional.
      When the user does not use GA4 data for feature engineering, this field is
      required.
    transaction_id_col: Column name for transaction ID. This is optional. When
      the user does not use GA4 data for feature engineering, this field is
      required.
    ml_training_table_name_for_existing_customers: The table name for training
      the model for existing customers.
    ml_training_table_name_for_first_time_purchase: The table name for training
      model for first time purchase.
    regression_model_type: The type of regression model to use.
    binary_classifier_model_type: The type of binary classifier model to use.
    is_two_step_model: Whether to use a two-step model.
    numeric_labels: The list of numeric labels.
    categorical_labels: The list of categorical labels.
    id_cols: The list of ID columns.
  """

  project_id: str
  dataset_id: str
  gcp_bq_client: bigquery.Client
  gcp_storage: storage.Client
  gcp_bucket_name: str
  location: str = 'US'
  ga4_project_id: str | None = None
  ga4_dataset_id: str | None = None
  use_ga4_data_for_feature_engineering: bool = False
  ml_training_table_name: str | None = None
  ml_prediction_table_name: str | None = None
  transaction_date_col: str | None = None
  transaction_id_col: str | None = None
  refund_value_col: str | None = None
  refund_flag_col: str | None = None
  refund_proportion_col: str | None = None
  train_test_split_test_size_proportion: float | None = 0.1
  invalid_value_threshold_for_row_removal: float = 0.5
  invalid_value_threshold_for_column_removal: float = 0.95
  min_correlation_threshold_with_numeric_labels_for_feature_reduction: (
      float | None
  ) = 0.1
  regression_model_type: constant.SupportedModelTypes = dataclasses.field(
      default_factory=lambda: constant.LinearBigQueryMLModelType.LINEAR_REGRESSION
  )
  binary_classifier_model_type: constant.SupportedModelTypes = (
      dataclasses.field(
          default_factory=lambda: constant.LinearBigQueryMLModelType.LOGISTIC_REGRESSION
      )
  )
  is_two_step_model: bool = dataclasses.field(default_factory=bool)
  numeric_labels: list[str] = dataclasses.field(default_factory=list)
  categorical_labels: list[str] = dataclasses.field(default_factory=list)
  id_cols: list[str] = dataclasses.field(default_factory=list)

  def __post_init__(self):
    if self.use_ga4_data_for_feature_engineering:
      if not self.ga4_project_id:
        raise ValueError(
            'ga4_project_id must be specified when'
            ' use_ga4_data_for_feature_engineering is True.'
        )
      if not self.ga4_dataset_id:
        raise ValueError(
            'ga4_dataset_id must be specified when'
            ' use_ga4_data_for_feature_engineering is True.'
        )
      self.refund_value_col = constant.TargetVariable.REFUND_VALUE.value
      self.refund_flag_col = constant.TargetVariable.REFUND_FLAG.value
      self.refund_proportion_col = (
          constant.TargetVariable.REFUND_PROPORTION.value
      )
      self.transaction_date_col = constant.IDColumnNames.TRANSACTION_DATE.value
      self.transaction_id_col = constant.IDColumnNames.TRANSACTION_ID.value
      self.numeric_labels = [
          self.refund_value_col,
          self.refund_proportion_col,
      ]
      self.categorical_labels = [self.refund_flag_col]
      self.id_cols = [
          self.transaction_date_col,
          self.transaction_id_col,
      ]

    if not self.use_ga4_data_for_feature_engineering:
      if not self.transaction_date_col:
        raise ValueError(
            'transaction_date_col must be specified when'
            ' use_ga4_data_for_feature_engineering is False.'
        )
      if not self.transaction_id_col:
        raise ValueError(
            'transaction_id_col must be specified when'
            ' use_ga4_data_for_feature_engineering is False.'
        )
      if not self.refund_value_col:
        raise ValueError(
            'refund_value_col must be specified when'
            ' use_ga4_data_for_feature_engineering is False.'
        )
      if not self.refund_flag_col:
        raise ValueError(
            'refund_flag_col must be specified when'
            ' use_ga4_data_for_feature_engineering is False.'
        )
      if not self.refund_proportion_col:
        raise ValueError(
            'refund_proportion_col must be specified when'
            ' use_ga4_data_for_feature_engineering is False.'
        )
      if (
          not self.use_ga4_data_for_feature_engineering
          and not utils.check_bigquery_table_exists(
              self.gcp_bq_client,
              self.dataset_id,
              self.ml_training_table_name,
          )
      ):
        raise ValueError(
            'ml_training_table_name must exist when'
            ' use_ga4_data_for_feature_engineering is False. Please make sure'
            ' your preprocessed ml ready data is being created on BigQuery and'
            ' provide the table name in ml_training_table_name parameter.'
        )

  def data_processing_feature_engineering(
      self,
      data_pipeline_type: constant.DataPipelineType,
      recency_of_transaction_for_prediction_in_days: int,
      return_policy_window_in_days: int,
      recency_of_data_in_days: int,
  ) -> None:
    """Data processing and feature engineering.

    Args:
      data_pipeline_type: The type of data pipeline to use, either training or
        prediction.
      recency_of_transaction_for_prediction_in_days: The number of days to look
        back for transactions to be considered for prediction.
      return_policy_window_in_days: Return policy window in days. For example,
        if return_policy_window_in_days is 30, then the model will predict the
        refund only after 30 days of the transaction date.
      recency_of_data_in_days: The number of days to look back in data for model
        training.

    Returns:
      ml_ready_dfs_for_existing_customers: The ml ready dataframes for existing
        customers.
      ml_ready_dfs_for_first_time_purchase: The ml ready dataframes for first
        time purchase.
    """
    use_prediction_pipeline = (
        data_pipeline_type == constant.DataPipelineType.PREDICTION
    )
    if self.use_ga4_data_for_feature_engineering:
      for _, query_file in constant.GA4_DATA_PIPELINE_QUERY_TEMPLATES.items():
        query = utils.read_file(query_file)
        query = query.format(
            ga4_project_id=self.ga4_project_id,
            ga4_dataset_id=self.ga4_dataset_id,
            project_id=self.project_id,
            data_pipeline_type=data_pipeline_type.value,
            dataset_id=self.dataset_id,
            recency_of_transaction_for_prediction_in_days=recency_of_transaction_for_prediction_in_days,
            return_policy_window_in_days=return_policy_window_in_days,
            recency_of_data_in_days=recency_of_data_in_days,
        )
        self.gcp_bq_client.query(query).result()
      ml_ready_data_for_existing_customers = utils.read_bq_table_to_df(
          bigquery_client=self.gcp_bq_client,
          project_id=self.project_id,
          dataset_id=self.dataset_id,
          table_name=(
              f'{data_pipeline_type.value}_ml_ready_data_for_existing_customers'
          ),
      )
      ml_ready_data_for_first_time_purchase = utils.read_bq_table_to_df(
          bigquery_client=self.gcp_bq_client,
          project_id=self.project_id,
          dataset_id=self.dataset_id,
          table_name=(
              f'{data_pipeline_type.value}_ml_ready_data_for_first_time_purchase'
          ),
      )
      data_cleaning_feature_selection.data_preprocessing_for_ml(
          use_prediction_pipeline=use_prediction_pipeline,
          df=ml_ready_data_for_existing_customers,
          bigquery_client=self.gcp_bq_client,
          gcp_storage_client=self.gcp_storage,
          gcp_bucket_name=self.gcp_bucket_name,
          dataset_id=self.dataset_id,
          table_name=(
              f'{data_pipeline_type.value}_ml_ready_data_for_existing_customers'
          ),
          id_cols=self.id_cols,
          numeric_labels=self.numeric_labels,
          categorical_labels=self.categorical_labels,
          train_test_split_order_by_cols=[self.transaction_date_col],
          location=self.location,
          train_test_split_test_size_proportion=self.train_test_split_test_size_proportion,
          invalid_value_threshold_for_row_removal=self.invalid_value_threshold_for_row_removal,
          invalid_value_threshold_for_column_removal=self.invalid_value_threshold_for_column_removal,
          min_correlation_threshold_with_numeric_labels_for_feature_reduction=self.min_correlation_threshold_with_numeric_labels_for_feature_reduction,
      )

      data_cleaning_feature_selection.data_preprocessing_for_ml(
          use_prediction_pipeline=use_prediction_pipeline,
          df=ml_ready_data_for_first_time_purchase,
          bigquery_client=self.gcp_bq_client,
          gcp_storage_client=self.gcp_storage,
          gcp_bucket_name=self.gcp_bucket_name,
          dataset_id=self.dataset_id,
          table_name=(
              f'{data_pipeline_type.value}_ml_ready_data_for_first_time_purchase'
          ),
          id_cols=self.id_cols,
          numeric_labels=self.numeric_labels,
          categorical_labels=self.categorical_labels,
          train_test_split_order_by_cols=[self.transaction_date_col],
          location=self.location,
          train_test_split_test_size_proportion=self.train_test_split_test_size_proportion,
          invalid_value_threshold_for_row_removal=self.invalid_value_threshold_for_row_removal,
          invalid_value_threshold_for_column_removal=self.invalid_value_threshold_for_column_removal,
          min_correlation_threshold_with_numeric_labels_for_feature_reduction=self.min_correlation_threshold_with_numeric_labels_for_feature_reduction,
      )
    else:
      data_cleaning_feature_selection.create_ml_ready_data_for_preprocessed_data_provided_by_user(
          preprocessed_table_name_by_user=self.ml_training_table_name,
          bigquery_client=self.gcp_bq_client,
          project_id=self.project_id,
          dataset_id=self.dataset_id,
          refund_value_col=self.refund_value_col,
          refund_flag_col=self.refund_flag_col,
          refund_proportion_col=self.refund_proportion_col,
          use_prediction_pipeline=use_prediction_pipeline,
          transaction_id_col=self.transaction_id_col,
          train_test_split_test_size_proportion=self.train_test_split_test_size_proportion,
      )

  def model_training_pipeline_evaluation_and_prediction(
      self,
      is_two_step_model: bool,
      regression_model_type: constant.SupportedModelTypes = constant.LinearBigQueryMLModelType.LINEAR_REGRESSION,
      binary_classifier_model_type: constant.SupportedModelTypes = constant.LinearBigQueryMLModelType.LOGISTIC_REGRESSION,
      first_time_purchase: bool | None = None,
      num_tiers_to_create_avg_prediction: int = 10,
      probability_threshold_for_prediction: float = 0.5,
      probability_threshold_for_model_evaluation: float = 0.5,
      bqml_template_files_dir: Mapping[
          str, str
      ] = constant.BQML_QUERY_TEMPLATE_FILES,
      **plot_kwargs,
  ) -> tuple[
      Mapping[str, pd.DataFrame],
      pd.DataFrame,
      Mapping[str, pd.DataFrame],
      Mapping[str, pd.DataFrame],
  ]:
    """Model training, evaluation and prediction.

    This method trains model and evaluates model performance in the following
    scenarios: 1) When use_ga4_data_for_feature_engineering is True, the
    method trains the model for either first time purchase or existing
    customers data preprocessed from data_processing_feature_engineering method.
    2) When use_ga4_data_for_feature_engineering is False, the method trains
    the model on the data provided by the user using ml_training_table_name
    attribute. At the end of the method, the model prediction for the training
    data is also generated. Visualization and dataframes for showing the
    distribution of predictions vs. actuals and feature importance are also
    generated.

    Args:
      is_two_step_model: Whether to use a two-step model.
      regression_model_type: The type of regression model to use.
      binary_classifier_model_type: The type of binary classifier model to use.
      first_time_purchase: Optional boolean flag to indicate whether to train
        the model for first time purchase. When set to False, the model will be
        trained for existing customers. When
        use_ga4_data_for_feature_engineering is False, this argument is not used
        and will be ignored. When use_ga4_data_for_feature_engineering is True,
        this argument is needed to determine whether to train the model for
        first time purchase or existing customers.
      num_tiers_to_create_avg_prediction: The number of tiers to create for
        average prediction for creating tier wise prediction vs. actual refund
        prediction for comparison and model evaluation.
      probability_threshold_for_prediction: The probability threshold to use for
        prediction for binary classifier model when is_two_step_model is True.
      probability_threshold_for_model_evaluation: The probability threshold to
        use for model evaluation for binary classifier model when
        is_two_step_model is True.
      bqml_template_files_dir: Directory mapping with model type (i.e.
        regression_only_training, regression_only_prediction,
        classification_regression_training and
        classification_regression_prediction) as the key and and corresponding
        model prediction BQML query template files as the value.
      **plot_kwargs: Additional arguments to pass to matplotlib.pyplot in
        model_prediction_evaluation.plot_predictions_actuals_distribution and
        model_prediction_evaluation.compare_and_plot_tier_level_avg_prediction

    Returns:
      performance_metrics_dfs: The performance metrics dataframes.
      model_prediction_df: The model prediction dataframe.
      predictions_actuals_distribution: The predictions and actuals
        distribution.
      feature_importance_dfs: The feature importance dataframes.

    Raises:
      ValueError: If first_time_purchase is not specified when
      use_ga4_data_for_feature_engineering is True or if ml_training_table_name
      is not specified when use_ga4_data_for_feature_engineering is False.
    """
    if (
        first_time_purchase is None
        and self.use_ga4_data_for_feature_engineering
    ):
      raise ValueError(
          'first_time_purchase boolean flag must be specified when'
          ' use_ga4_data_for_feature_engineering is True.'
      )
    if (
        not self.use_ga4_data_for_feature_engineering
        and self.ml_training_table_name is None
    ):
      raise ValueError(
          'ml_training_table_name must be specified when'
          ' use_ga4_data_for_feature_engineering is False.'
      )
    if self.use_ga4_data_for_feature_engineering:
      if first_time_purchase:
        preprocessed_table_name = f'{constant.DataPipelineType.TRAINING.value}_ml_ready_data_for_first_time_purchase'
      else:
        preprocessed_table_name = f'{constant.DataPipelineType.TRAINING.value}_ml_ready_data_for_existing_customers'
    else:
      preprocessed_table_name = self.ml_training_table_name
    model.bigquery_ml_model_training(
        preprocessed_table_name=preprocessed_table_name,
        project_id=self.project_id,
        dataset_id=self.dataset_id,
        transaction_date_col=self.transaction_date_col,
        transaction_id_col=self.transaction_id_col,
        num_tiers=num_tiers_to_create_avg_prediction,
        bigquery_client=self.gcp_bq_client,
        regression_model_type=regression_model_type,
        binary_classifier_model_type=binary_classifier_model_type,
        refund_value=self.refund_value_col,
        refund_flag=self.refund_flag_col,
        is_two_step_model=is_two_step_model,
        probability_threshold_for_prediction=probability_threshold_for_prediction,
        probability_threshold_for_model_evaluation=probability_threshold_for_model_evaluation,
        bqml_template_files_dir=bqml_template_files_dir,
    )
    performance_metrics_dfs = (
        model_prediction_evaluation.model_performance_metrics(
            preprocessed_table_name=preprocessed_table_name,
            project_id=self.project_id,
            bigquery_client=self.gcp_bq_client,
            dataset_id=self.dataset_id,
            is_two_step_model=is_two_step_model,
            refund_flag=self.refund_flag_col,
            refund_value=self.refund_value_col,
            regression_model_type=regression_model_type,
            binary_classifier_model_type=binary_classifier_model_type,
        )
    )

    model_prediction_df = model_prediction_evaluation.model_prediction(
        project_id=self.project_id,
        dataset_id=self.dataset_id,
        bigquery_client=self.gcp_bq_client,
        preprocessed_table_name=preprocessed_table_name,
        refund_value=self.refund_value_col,
        refund_flag=self.refund_flag_col,
        regression_model_type=regression_model_type,
        use_prediction_pipeline=False,
        is_two_step_model=is_two_step_model,
    )
    predictions_actuals_distribution = (
        model_prediction_evaluation.plot_predictions_actuals_distribution(
            prediction_df=model_prediction_df, **plot_kwargs
        )
    )
    model_prediction_evaluation.compare_and_plot_tier_level_avg_prediction(
        project_id=self.project_id,
        preprocessed_table_name=preprocessed_table_name,
        dataset_id=self.dataset_id,
        bigquery_client=self.gcp_bq_client,
        refund_flag=self.refund_flag_col,
        refund_value=self.refund_value_col,
        **plot_kwargs,
    )
    feature_importance_dfs = (
        model_prediction_evaluation.training_feature_importance(
            project_id=self.project_id,
            dataset_id=self.dataset_id,
            preprocessed_table_name=preprocessed_table_name,
            bigquery_client=self.gcp_bq_client,
            refund_value=self.refund_value_col,
            refund_flag=self.refund_flag_col,
            regression_model_type=regression_model_type,
            binary_classifier_model_type=binary_classifier_model_type,
            is_two_step_model=is_two_step_model,
        )
    )

    return (
        performance_metrics_dfs,
        model_prediction_df,
        predictions_actuals_distribution,
        feature_importance_dfs,
    )

  def prediction_pipeline_prediction_generation(
      self,
      is_two_step_model: bool,
      regression_model_type: constant.SupportedModelTypes = constant.LinearBigQueryMLModelType.LINEAR_REGRESSION,
      binary_classifier_model_type: constant.SupportedModelTypes = constant.LinearBigQueryMLModelType.LOGISTIC_REGRESSION,
      first_time_purchase: bool | None = None,
      probability_threshold_for_prediction: float = 0.5,
      bqml_template_files_dir: Mapping[
          str, str
      ] = constant.BQML_QUERY_TEMPLATE_FILES,
  ) -> None:
    """Generate model prediction for the prediction pipeline.

    This method picks the pretrained model and generates prediction on
    prediction dataset for the prediction pipeline. The prediction data will be
    stored on BigQuery.

    Args:
      is_two_step_model: Whether to use a two-step pretrained model (binary
        classification with refund flag as label and regression model with
        refund value or refund proportion as label) for prediction.
      regression_model_type: The type of regression model to use for generating
        prediction.
      binary_classifier_model_type: The type of binary classifier model to use
        for generating prediction.
      first_time_purchase: Optional boolean flag to indicate whether to make
        prediction using the model trained for first time purchase. When set to
        False, the prediction will be made using the model trained for existing
        customers. When use_ga4_data_for_feature_engineering is False, this
        argument is not used and will be ignored. When
        use_ga4_data_for_feature_engineering is True, this argument is needed to
        determine whether to use the model trained for first time purchase or
        existing customers.
      probability_threshold_for_prediction: The probability threshold to use for
        prediction for binary classifier model when is_two_step_model is True.
      bqml_template_files_dir: Directory mapping with model type (i.e.
        regression_only_training, regression_only_prediction,
        classification_regression_training and
        classification_regression_prediction) as the key and and corresponding
        model prediction BQML query template files as the value.

    Raises:
      ValueError: If first_time_purchase is not specified when
      use_ga4_data_for_feature_engineering is True or if
      ml_prediction_table_name is not specified when
      use_ga4_data_for_feature_engineering is False.
    """
    if (
        first_time_purchase is None
        and self.use_ga4_data_for_feature_engineering
    ):
      raise ValueError(
          'first_time_purchase boolean flag must be specified when'
          ' use_ga4_data_for_feature_engineering is True.'
      )
    if (
        not self.use_ga4_data_for_feature_engineering
        and self.ml_prediction_table_name is None
    ):
      raise ValueError(
          'ml_prediction_table_name must be specified when'
          ' use_ga4_data_for_feature_engineering is False.'
      )
    if self.use_ga4_data_for_feature_engineering:
      if first_time_purchase:
        preprocessed_table_name = f'{constant.DataPipelineType.PREDICTION.value}_ml_ready_data_for_first_time_purchase'
      else:
        preprocessed_table_name = f'{constant.DataPipelineType.PREDICTION.value}_ml_ready_data_for_existing_customers'
      preprocessed_training_table_name = preprocessed_table_name.replace(
          constant.DataPipelineType.PREDICTION.value,
          constant.DataPipelineType.TRAINING.value,
      )
    else:
      preprocessed_table_name = self.ml_prediction_table_name
      preprocessed_training_table_name = self.ml_training_table_name

    model.bigquery_ml_model_prediction(
        project_id=self.project_id,
        dataset_id=self.dataset_id,
        preprocessed_table_name=preprocessed_table_name,
        bigquery_client=self.gcp_bq_client,
        transaction_date_col=self.transaction_date_col,
        transaction_id_col=self.transaction_id_col,
        regression_model_type=regression_model_type,
        binary_classifier_model_type=binary_classifier_model_type,
        refund_value=self.refund_value_col,
        refund_flag=self.refund_flag_col,
        probability_threshold_for_prediction=probability_threshold_for_prediction,
        is_two_step_model=is_two_step_model,
        bqml_template_files_dir=bqml_template_files_dir,
        preprocessed_training_table_name=preprocessed_training_table_name,
    )
