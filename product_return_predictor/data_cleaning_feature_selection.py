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

"""Module for further cleaning data and feature selection."""

from collections.abc import Mapping, Sequence
import dataclasses

from google.cloud import bigquery
from google.cloud import storage
import numpy as np
import pandas as pd
from sklearn import compose
from sklearn import pipeline
from sklearn import preprocessing

from product_return_predictor import constant
from product_return_predictor import custom_transformer
from product_return_predictor import utils


def _convert_columns_to_right_data_type(
    df: pd.DataFrame,
    string_cols: Sequence[str],
    numeric_cols: Sequence[str],
    date_cols: Sequence[str],
) -> pd.DataFrame:
  """Convert columns to the right data type in the input dataframe.

  Args:
    df: Input pandas dataframe.
    string_cols: Columns of input dataframe with string data type.
    numeric_cols: Columns of input dataframe with numeric data type.
    date_cols: Columns of input dataframe with date data type.

  Returns:
    Dataframe with columns converted to the right data types.
  """
  df[string_cols] = df[string_cols].astype('string')
  df[numeric_cols] = df[numeric_cols].astype('float')
  df[date_cols] = df[date_cols].apply(
      lambda x: pd.to_datetime(x, errors='coerce', format='%Y-%m-%d')
  )
  return df


def _create_distribution_summary_for_target_variables(
    df: pd.DataFrame,
    bigquery_client: bigquery.Client,
    dataset_id: str,
    table_name: str,
    labels: Sequence[str],
    location: str = 'europe-west4',
) -> None:
  """Create distribution summary for target variables.

  Args:
    df: Input pandas dataframe.
    bigquery_client: Bigquery client for querying Bigquery.
    dataset_id: Bigquery dataset id of the destination target variable
      distribution table.
    table_name: Bigquery table name of the destination target variable
      distribution table.
    labels: Columns of the target variables from the input dataframe.
    location: Bigquery location of the destination target variable distribution
      table. Given the solution is built in EMEA, the default location is set to
      'europe-west4'.
  """
  target_variables_distribution = df[labels].describe().reset_index()
  target_variables_distribution.rename(
      columns={'index': 'distribution type'}, inplace=True
  )
  utils.run_load_table_to_bigquery(
      data=target_variables_distribution,
      bigquery_client=bigquery_client,
      dataset_id=dataset_id,
      table_name=table_name,
      location=location,
  )


def _identify_values_with_invalid_string_datatype(
    df: pd.DataFrame,
) -> pd.DataFrame:
  """Determine if the input dataframe has invalid string data value.

  Args:
    df: Input pandas dataframe.

  Returns:
    Dataframe with boolean values indicating whether a value is invalid.
  """

  return df.isin(['unknown', 'Unassigned', 'unassigned', np.nan]) | pd.isnull(
      df
  )


def _identify_zeroes_numeric_datatype(
    df: pd.DataFrame,
) -> pd.DataFrame:
  """Determine if the input dataframe has invalid numeric data value.

  Args:
    df: Input pandas dataframe.

  Returns:
    Dataframe with boolean values indicating whether a value is invalid.
  """
  return (df == 0) | (pd.isnull(df))


def _identify_columns_with_many_invalid_or_zero_values(
    df: pd.DataFrame,
    numeric_cols: Sequence[str],
    string_cols: Sequence[str],
    invalid_value_threshold_for_column_removal: float = 0.95,
) -> Sequence[str]:
  """Identify columns in the input dataframe with high invalid values.

  Args:
    df: Input pandas dataframe.
    numeric_cols: Columns from the input dataframe with numeric data type.
    string_cols: Columns from the input dataframe with string data type.
    invalid_value_threshold_for_column_removal: The minimum threshold for the
      percentage of invalid values in a column to be considered during feature
      reduction.

  Returns:
    Columns with high invalid values.
  """
  invalid_string_data_count_by_columns = (
      df[numeric_cols].apply(_identify_zeroes_numeric_datatype).sum()
  )
  invalid_numeric_data_count_by_columns = (
      df[string_cols].apply(_identify_values_with_invalid_string_datatype).sum()
  )
  invalid_data_count_by_columns = pd.DataFrame(
      pd.concat([
          invalid_string_data_count_by_columns,
          invalid_numeric_data_count_by_columns,
      ]).reset_index()
  )
  invalid_data_count_by_columns.columns = ['col_name', 'invalid_values_count']
  invalid_data_count_by_columns.loc[:, 'invalid_values_count_pct'] = (
      invalid_data_count_by_columns['invalid_values_count'] / df.shape[0]
  )
  invalid_values_cols = list(
      invalid_data_count_by_columns.loc[
          invalid_data_count_by_columns['invalid_values_count_pct']
          > invalid_value_threshold_for_column_removal,
          'col_name',
      ]
  )
  return invalid_values_cols


@dataclasses.dataclass(frozen=True)
class LabelsAndTypes:
  """The result of `_get_labels_and_types`.

  Attributes:
    labels: Label Names.
    label_types: Mapping of label names to their data types.
  """

  labels: Sequence[str]
  label_types: Mapping[str, constant.LabelType]


def _get_labels_and_types(
    numeric_labels: Sequence[str], categorical_labels: Sequence[str]
) -> LabelsAndTypes:
  """Create a dictionary for labels and their data types.

  Args:
    numeric_labels: Numeric label names (e.g. refund value). Value error would
      be raised when 1) numeric labels or categorical labels are empty 2) there
      are duplicates in numeric_labels 3) two input labels are not mutually
      exclusive.
    categorical_labels: Categorical label names (e.g. refund flag).  Value error
      would be raised when 1) categorical labels or categorical labels are empty
      2) there are duplicates in categorical_labels 3) two input labels are not
      mutually exclusive.

  Returns:
    All label names and mapping of label names to their data types.
  """

  labels = [*numeric_labels, *categorical_labels]
  if not labels:
    raise ValueError('No labels provided.')
  if len(set(numeric_labels)) != len(numeric_labels):
    raise ValueError('Duplicate labels provided in numeric_labels.')
  if len(set(categorical_labels)) != len(categorical_labels):
    raise ValueError('Duplicate labels provided in categorical_labels.')
  if set(numeric_labels).intersection(set(categorical_labels)):
    common_labels = set(numeric_labels).intersection(set(categorical_labels))
    raise ValueError(
        'Labels in numeric_labels and categorical_labels are not mutually'
        f' exclusive. See common labels here: {common_labels}.'
    )
  label_types = {
      label: constant.LabelType.NUMERICAL for label in numeric_labels
  } | {label: constant.LabelType.CATEGORICAL for label in categorical_labels}
  return LabelsAndTypes(labels=labels, label_types=label_types)


def _identify_rows_with_high_invalid_values(
    df: pd.DataFrame,
    numeric_cols: Sequence[str],
    string_cols: Sequence[str],
    invalid_value_threshold_for_row_removal: float = 0.5,
) -> pd.Series:
  """Identify rows in the input dataframe with high invalid values.

  Args:
    df: Input pandas dataframe.
    numeric_cols: Columns from the input dataframe with numeric data type.
    string_cols: Columns from the input dataframe with string data type.
    invalid_value_threshold_for_row_removal: The minimum threshold for the
      percentage of columns with invalid values in a row to be considered during
      data cleaning.

  Returns:
   Boolean values indicating whether a row has high invalid values.
  """
  invalid_string_data_count_by_rows = (
      df[numeric_cols].apply(_identify_zeroes_numeric_datatype).sum(axis=1)
  )
  invalid_numeric_data_count_by_rows = (
      df[string_cols]
      .apply(_identify_values_with_invalid_string_datatype)
      .sum(axis=1)
  )
  invalid_data_count_by_rows = pd.concat(
      [invalid_numeric_data_count_by_rows, invalid_string_data_count_by_rows],
      axis=1,
  )
  invalid_data_count_by_rows.columns = [
      'numeric_col_invalid_values_count',
      'string_col_invalid_values_count',
  ]
  invalid_data_count_by_rows.loc[:, 'total_invalid_values_count'] = (
      invalid_data_count_by_rows['numeric_col_invalid_values_count']
      + invalid_data_count_by_rows['string_col_invalid_values_count']
  )
  invalid_data_count_by_rows.loc[:, 'total_invalid_values_count_pct'] = (
      invalid_data_count_by_rows['total_invalid_values_count']
      / (len(numeric_cols) + len(string_cols))
  )
  row_mask_with_high_invalid_values = (
      invalid_data_count_by_rows['total_invalid_values_count_pct']
      > invalid_value_threshold_for_row_removal
  )
  return row_mask_with_high_invalid_values


def _output_invalid_data_summary(
    df: pd.DataFrame,
    invalid_values_cols: Sequence[str],
    row_mask_with_high_invalid_values: pd.Series,
) -> None:
  """Output summary of invalid data in input dataframe and export the summary.

  Args:
    df: Input pandas dataframe.
    invalid_values_cols: Columns from the input dataframe with high invalid
      values.
    row_mask_with_high_invalid_values: Boolean values indicating whether a row
      has high invalid values.
  """
  invalid_values_analysis_txt = (
      'Total number of rows: {n_rows} \nTotal number of columns:'
      ' {n_cols}\nNumber of rows with high invalid values:'
      ' {n_rows_with_high_invalid_values} \nNumber of columns with high invalid'
      ' values: {n_columns_with_high_invalid_values} \nColumns with high'
      ' invalid values: \n{invalid_cols}'.format(
          invalid_cols=', '.join(invalid_values_cols),
          n_rows_with_high_invalid_values=row_mask_with_high_invalid_values.sum(),
          n_columns_with_high_invalid_values=len(invalid_values_cols),
          n_rows=df.shape[0],
          n_cols=df.shape[1],
      )
  )
  with open('invalid_values_analysis_txt.txt', 'w') as f:
    f.write(invalid_values_analysis_txt)
    f.close()


def _clean_and_handle_invalid_data(
    df: pd.DataFrame,
    bigquery_client: bigquery.Client,
    dataset_id: str,
    table_name: str,
    id_cols: Sequence[str],
    numeric_labels: Sequence[str],
    categorical_labels: Sequence[str],
    location: str = 'europe-west4',
    invalid_value_threshold_for_row_removal: float = 0.5,
    invalid_value_threshold_for_column_removal: float = 0.95,
    use_prediction_pipeline: bool = False,
) -> pd.DataFrame:
  """Clean data and remove rows/columns with high invalid values.

   If not in prediction pipeline remove row/columns with high invalid values
   would be skipped.

  Args:
    df: Input dataframe with features and labels columns.
    bigquery_client: Bigquery client for loading dataframe to BigQuery.
    dataset_id: Bigquery dataset id of the destination ML ready table.
    table_name: Bigquery table name of the destination ML ready table.
    id_cols: Columns from the input dataframe that uniquely identify each row.
    numeric_labels: Names of numerical labels.
    categorical_labels: Names of categorical labels.
    location: Location of the Bigquery table (e.g. 'europe-west4',
      'us-central1', 'us') for exporting the cleaned dataframe. Given the
      solution is built in EMEA, the default location is set to 'europe-west4'.
    invalid_value_threshold_for_row_removal: The minimum threshold for the
      percentage of columns with invalid values in a row to be considered during
      data cleaning.
    invalid_value_threshold_for_column_removal: The minimum threshold for the
      percentage of invalid values in a column to be considered during feature
      reduction.
    use_prediction_pipeline: Whether to preprocess data for prediction pipeline.

  Returns:
    Cleaned data.
  """

  cleaned_data = _data_cleaning(
      df=df,
      bigquery_client=bigquery_client,
      dataset_id=dataset_id,
      table_name=table_name,
      numeric_labels=numeric_labels,
      categorical_labels=categorical_labels,
      location=location,
  )

  if not use_prediction_pipeline:
    cleaned_data = _remove_high_invalid_values_row_columns(
        df=cleaned_data,
        id_cols=id_cols,
        numeric_labels=numeric_labels,
        categorical_labels=categorical_labels,
        invalid_value_threshold_for_row_removal=invalid_value_threshold_for_row_removal,
        invalid_value_threshold_for_column_removal=invalid_value_threshold_for_column_removal,
    )
  return cleaned_data


def _train_test_split(
    df: pd.DataFrame,
    order_by_col: Sequence[str] | None,
    asc_order: bool = False,
    test_size_proportion: float = 0.1,
) -> pd.DataFrame:
  """Split the input dataframe into train and test sets.

  Args:
    df: Input pandas dataframe.
    order_by_col: Column name used to order the rows in the input dataframe for
      splitting data into train and test sets. If order_by_col is None, then the
      dataframe will be shuffled.
    asc_order: Whether to order input data in ascending order based on given
      order_by_col before splitting.
    test_size_proportion: Proportion of the input dataframe to be used as the
      test set. Note: If there are more than 100k rows in the input dataframe,
      the test set will be 10k rows automatically.

  Returns:
    Dataframe with train test column indicating whether a row belongs to the
    train or test set.
  """
  if order_by_col:
    df = df.sort_values(by=order_by_col, ascending=asc_order)
  else:
    df = df.sample(n=df.shape[0])
  df.reset_index(drop=True, inplace=True)
  if df.shape[0] < 100000:
    test_set_start_index = int(test_size_proportion * len(df) * -1)
    df.loc[:, constant.TRAIN_TEST_COL_NAME] = constant.TrainTest.TRAIN.value
    df.iloc[test_set_start_index:, -1] = constant.TrainTest.TEST.value
  else:
    df.loc[:, constant.TRAIN_TEST_COL_NAME] = constant.TrainTest.TRAIN.value
    df.iloc[-10000:, -1] = constant.TrainTest.TEST.value
  return df


def _data_cleaning(
    df: pd.DataFrame,
    bigquery_client: bigquery.Client,
    dataset_id: str,
    table_name: str,
    numeric_labels: Sequence[str],
    categorical_labels: Sequence[str],
    location: str = 'europe-west4',
) -> pd.DataFrame:
  """Clean input data for product return model training & prediction.

  Args:
    df: Input pandas dataframe.
    bigquery_client: Bigquery client for querying Bigquery.
    dataset_id: Bigquery dataset id for exporting the cleaned dataframe.
    table_name: Bigquery table name for exporting the cleaned dataframe.
    numeric_labels: Columns from the input dataframe that are numeric target
      variables for product return prediction.
    categorical_labels: Columns from the input dataframe that are categorical
      target variables for product return prediction.
    location: Location of the Bigquery table (e.g. 'europe-west4',
      'us-central1', 'us') for exporting the cleaned dataframe. Given the
      solution is built in EMEA, the default location is set to 'europe-west4'.

  Returns:
    Cleaned dataframe.
  """
  labels = [*numeric_labels, *categorical_labels]
  date_cols, string_cols, numeric_cols = (
      custom_transformer.date_string_numeric_cols_from_input_dataframe(df)
  )
  df = _convert_columns_to_right_data_type(
      df, string_cols, numeric_cols, date_cols
  )
  fillna_for_string_variables = dict(
      list(zip(string_cols, ['unknown'] * len(string_cols)))
  )
  fillna_for_numeric_variables = dict(
      list(zip(numeric_cols, [0] * len(numeric_cols)))
  )
  df.fillna(fillna_for_numeric_variables, inplace=True)
  df.fillna(fillna_for_string_variables, inplace=True)
  _create_distribution_summary_for_target_variables(
      df=df,
      labels=labels,
      bigquery_client=bigquery_client,
      dataset_id=dataset_id,
      table_name='{}_target_variable_distribution'.format(table_name),
      location=location,
  )
  return df


def _remove_high_invalid_values_row_columns(
    df: pd.DataFrame,
    id_cols: Sequence[str],
    numeric_labels: Sequence[str],
    categorical_labels: Sequence[str],
    invalid_value_threshold_for_row_removal: float = 0.5,
    invalid_value_threshold_for_column_removal: float = 0.95,
) -> pd.DataFrame:
  """Remove rows & columns with high invalid values and export a summary.

  Args:
    df: Input pandas dataframe.
    id_cols: Columns from the input dataframe that uniquely identify each row.
    numeric_labels: Columns from the input dataframe that are numeric target
      variables for product return prediction.
    categorical_labels: Columns from the input dataframe that are categorical
      target variables for product return prediction.
    invalid_value_threshold_for_row_removal: The minimum threshold for the
      percentage of columns with invalid values in a row to be considered during
      data cleaning.
    invalid_value_threshold_for_column_removal: The minimum threshold for the
      percentage of invalid values in a column to be considered during feature
      reduction.

  Returns:
    DataFrame after having removed columns and rows with high invalid values.
  """
  labels = [*numeric_labels, *categorical_labels]
  features = [col for col in df.columns if col not in [*id_cols, *labels]]
  _, string_cols, numeric_cols = (
      custom_transformer.date_string_numeric_cols_from_input_dataframe(df)
  )
  invalid_values_cols = _identify_columns_with_many_invalid_or_zero_values(
      df, numeric_cols, string_cols, invalid_value_threshold_for_column_removal
  )
  row_mask_with_high_invalid_values = _identify_rows_with_high_invalid_values(
      df, numeric_cols, string_cols, invalid_value_threshold_for_row_removal
  )
  _output_invalid_data_summary(
      df, invalid_values_cols, row_mask_with_high_invalid_values
  )

  features_with_valid_values = [
      col for col in features if col not in invalid_values_cols
  ]
  return df.loc[
      ~row_mask_with_high_invalid_values,
      [*id_cols, *labels, *features_with_valid_values],
  ]


@dataclasses.dataclass(frozen=True)
class TrainTestFeaturesLabels:
  """The result of `_create_train_test_features_labels`.

  Attributes:
    x_train: Features for training set.
    x_test: Features for testing set.
    y_train: Target variable for training set.
    y_test: Target variable for testing set.
    training_data_index: Id columns values for the training set.
    testing_data_index: Id columns values for the testing set.
  """

  x_train: pd.DataFrame
  x_test: pd.DataFrame
  y_train: pd.Series
  y_test: pd.Series
  training_data_index: pd.DataFrame
  testing_data_index: pd.DataFrame


def _create_train_test_features_labels(
    df: pd.DataFrame, label: str, train_test_col: str, id_cols: Sequence[str]
) -> TrainTestFeaturesLabels:
  """Create train and test sets and seperate labels & features for input data.

  Args:
    df: Input machine learning dataset.
    label: Target variable of a given machine learning dataset.
    train_test_col: Name of the column that differentiates train and test data.
    id_cols: Columns from the input dataframe that uniquely identify each row.

  Returns:
    A tuple (x_train, x_test, y_train, y_test, training_data_index,
    testing_data_index).
  """
  x_train = (
      df.loc[df[train_test_col] == constant.TrainTest.TRAIN.value]
      .drop([train_test_col, label], axis=1)
      .reset_index(drop=True)
  )
  x_test = (
      df.loc[df[train_test_col] == constant.TrainTest.TEST.value]
      .drop([train_test_col, label], axis=1)
      .reset_index(drop=True)
  )
  y_train = df.loc[
      df[train_test_col] == constant.TrainTest.TRAIN.value, label
  ].reset_index(drop=True)
  y_test = df.loc[
      df[train_test_col] == constant.TrainTest.TEST.value, label
  ].reset_index(drop=True)
  training_data_index = x_train[id_cols]
  testing_data_index = x_test[id_cols]
  return TrainTestFeaturesLabels(
      x_train=x_train,
      x_test=x_test,
      y_train=y_train,
      y_test=y_test,
      training_data_index=training_data_index,
      testing_data_index=testing_data_index,
  )


def _create_data_transformation_pipeline(
    label_type: constant.LabelType,
    categorical_features: Sequence[str],
    numerical_features: Sequence[str],
) -> pipeline.Pipeline:
  """Create data transformation pipeline for transforming features.

  Args:
    label_type: Type of the target variable (i.e. numerical or categorical).
    categorical_features: Column names of categorical features.
    numerical_features: Column names of numerical features.

  Returns:
    Customized sklearn preprocessing pipeline.
  """

  categorical_transformer = pipeline.Pipeline(
      steps=[(
          'onehot',
          preprocessing.OneHotEncoder(
              handle_unknown='ignore', sparse_output=False
          ),
      )]
  )
  numerical_transformer = pipeline.Pipeline(
      steps=[('scaler', preprocessing.MinMaxScaler())]
  )

  preprocessor = compose.ColumnTransformer(
      transformers=[
          ('cat', categorical_transformer, categorical_features),
          ('num', numerical_transformer, numerical_features),
      ]
  )

  return pipeline.Pipeline(
      steps=[
          ('preprocessor', preprocessor),
          (
              'resampler',
              custom_transformer.ResamplingTransformer(label_type=label_type),
          ),
      ]
  )


def _get_data_transformation_pipeline(
    use_prediction_pipeline: bool = True,
    numerical_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
    label_type: constant.LabelType | None = None,
    gcp_storage_client: storage.Client | None = None,
    gcp_bucket_name: str | None = None,
    pipeline_name: str | None = None,
) -> pipeline.Pipeline:
  """Create or load data transformation pipeline for transforming features.

  Args:
    use_prediction_pipeline: If True, then a pretrained transformer needs to be
      loaded. Otherwise, a custom transfomer needs to be created.
    numerical_features: Numerical features.
    categorical_features: Categorical features.
    label_type: Type of the target variable (i.e. numerical or categorical).
    gcp_storage_client: Google Cloud Storage client for loading pretrained
      pipeline.
    gcp_bucket_name: Name of Google Cloud Storage bucket where pretrained
      pipeline is saved.
    pipeline_name: Name of the pretrained pipeline.

  Returns:
    Customized sklearn preprocessing pipeline.
  """
  if use_prediction_pipeline:
    if (gcp_storage_client is None) or (not gcp_storage_client):
      raise ValueError(
          'Missing required argument gcp_storage_client for loading pre-trained'
          ' pipeline.'
      )
    if (gcp_bucket_name is None) or (not gcp_bucket_name):
      raise ValueError(
          'Missing required argument gcp_bucket_name for loading pre-trained'
          ' pipeline.'
      )
    if (pipeline_name is None) or (not pipeline_name):
      raise ValueError(
          'Missing required argument pipeline_name for loading pre-trained'
          ' pipeline.'
      )
  else:
    if (categorical_features is None or not categorical_features) and (
        numerical_features is None or not numerical_features
    ):
      raise ValueError(
          'Missing required arguments categorical_features or'
          ' numerical_features for data pipeline creation.'
      )
    if label_type is None:
      raise ValueError(
          'Missing required argument label_type for data pipeline creation.'
      )

  if use_prediction_pipeline:
    try:
      return utils.load_pipeline_from_cloud_storage(
          gcp_storage_client=gcp_storage_client,
          gcp_bucket_name=gcp_bucket_name,
          pipeline_name=pipeline_name,
      )
    except Exception as e:
      raise RuntimeError(f'Error loading pre-trained pipeline: {e}') from e
  else:
    return _create_data_transformation_pipeline(
        label_type=label_type,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
    )


@dataclasses.dataclass(frozen=True)
class DataTransformationOutput:
  """The result of `_data_transformation`.

  Attributes:
    ml_ready_data: ML ready dataset w. train & test features & label.
    custom_data_transformer: Trained data processing pipeline.
  """

  ml_ready_data: pd.DataFrame
  custom_data_transformer: pipeline.Pipeline


def _data_transformation(
    fit_data_bool: bool,
    id_cols: Sequence[str],
    x: pd.DataFrame,
    data_index: pd.DataFrame,
    custom_data_transformer: pipeline.Pipeline,
    y: pd.Series | None = None,
) -> DataTransformationOutput:
  """Transform machine learning data using custom data transformation pipeline.

  Args:
    fit_data_bool: Whether to fit the data first before transformation.
    id_cols: Columns from the input dataframe that uniquely identify each row.
    x: Features for the dataset.
    data_index: Id columns values for the dataset.
    custom_data_transformer: Customizied sklearn data preprocessing pipeline.
    y: Label for the dataset.

  Returns:
    A tuple (ml_ready_data, custom_data_transformer).
  """
  if fit_data_bool:
    x_preprocessed = custom_data_transformer.fit_transform(x, y)
    custom_data_transformer.named_steps['resampler'].fitted_ = True
  else:
    x_preprocessed = custom_data_transformer.transform(x)

  feature_names = custom_data_transformer.named_steps[
      'preprocessor'
  ].get_feature_names_out()

  x_preprocessed = pd.DataFrame(x_preprocessed, columns=feature_names)

  x_preprocessed = pd.merge(
      data_index,
      x_preprocessed,
      left_index=True,
      right_index=True,
  )
  if y is not None:
    y = pd.merge(data_index, y, left_index=True, right_index=True)
    ml_ready_data = pd.merge(x_preprocessed, y, on=id_cols)
  else:
    ml_ready_data = x_preprocessed

  if fit_data_bool:
    ml_ready_data.loc[:, constant.TRAIN_TEST_COL_NAME] = (
        constant.TrainTest.TRAIN.value
    )
  else:
    ml_ready_data.loc[:, constant.TRAIN_TEST_COL_NAME] = (
        constant.TrainTest.TEST.value
    )
  return DataTransformationOutput(
      ml_ready_data=ml_ready_data,
      custom_data_transformer=custom_data_transformer,
  )


def _data_preprocessing_training_prediction_pipeline(
    use_prediction_pipeline: bool,
    df: pd.DataFrame,
    label: str,
    id_cols: Sequence[str],
    gcp_storage_client: storage.Client,
    gcp_bucket_name: str,
    pipeline_name: str,
    train_test_split_order_by_cols: Sequence[str] | None = None,
    train_test_split_asc_order: bool = True,
    train_test_split_test_size_proportion: float = 0.3,
    numerical_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
    label_type: constant.LabelType | None = None,
) -> pd.DataFrame:
  """Preprocess ML data for prediction or training pipelines.

  Args:
    use_prediction_pipeline: Whether to preprocess data for prediction pipeline.
    df: Input data.
    label: label/target variable for the dataset.
    id_cols: Id columns values for the dataset that identify each row.
    gcp_storage_client: Google Cloud Storage client.
    gcp_bucket_name: Google Cloud Storage bucket name where trained pipeline has
      been or will be stored.
    pipeline_name: Name of the pipeline file (without the .pkl extension)that
      has been or will be saved.
    train_test_split_order_by_cols: Columns used to order the data before the
      split.
    train_test_split_asc_order: Whether to order dataframe in ascending order
      based on train_test_split_col before splitting the data.
    train_test_split_test_size_proportion: Proportion of the input dataframe to
      be used as the test set. Note: If there are more than 100k rows in the
      input dataframe, the test set will be 10k rows automatically.
    numerical_features: Column names of numerical features.
    categorical_features: Column names of categorical features.
    label_type: Type of the target variable (i.e. numerical or categorical).

  Returns:
    Preprocessed ML ready dataset.
  """
  if use_prediction_pipeline:
    if (gcp_storage_client is None) or (not gcp_storage_client):
      raise ValueError(
          'Missing required arguments for prediction: gcp_storage_client, '
          'for prediction pipeline.'
      )
    if (gcp_bucket_name is None) or (not gcp_bucket_name):
      raise ValueError(
          'Missing required arguments for prediction: gcp_bucket_name, '
          'for prediction pipeline.'
      )
    if (pipeline_name is None) or (not pipeline_name):
      raise ValueError(
          'Missing required arguments for prediction: pipeline_name, '
          'for prediction pipeline.'
      )
  else:
    if (numerical_features is None or not numerical_features) and (
        categorical_features is None or not categorical_features
    ):
      raise ValueError(
          'Missing required arguments for training: numerical_features, '
          'or categorical_features for training pipeline.'
      )
    if label_type is None:
      raise ValueError(
          'Missing required arguments for training: label_type '
          'for training pipeline.'
      )
  if not use_prediction_pipeline:
    categorical_features = [f for f in categorical_features if f in df.columns]
    numerical_features = [f for f in numerical_features if f in df.columns]
  custom_data_transformer = _get_data_transformation_pipeline(
      use_prediction_pipeline=use_prediction_pipeline,
      numerical_features=numerical_features,
      categorical_features=categorical_features,
      label_type=label_type,
      gcp_storage_client=gcp_storage_client,
      gcp_bucket_name=gcp_bucket_name,
      pipeline_name=pipeline_name,
  )
  if use_prediction_pipeline:
    return _data_transformation(
        fit_data_bool=False,
        id_cols=id_cols,
        x=df,
        data_index=df[id_cols],
        custom_data_transformer=custom_data_transformer,
    ).ml_ready_data
  else:
    df_with_train_test_split_col = _train_test_split(
        df=df,
        order_by_col=train_test_split_order_by_cols,
        asc_order=train_test_split_asc_order,
        test_size_proportion=train_test_split_test_size_proportion,
    )
    result = _create_train_test_features_labels(
        df=df_with_train_test_split_col,
        id_cols=id_cols,
        label=label,
        train_test_col=constant.TRAIN_TEST_COL_NAME,
    )
    training_data_transformation_result = _data_transformation(
        fit_data_bool=True,
        id_cols=id_cols,
        x=result.x_train,
        y=result.y_train,
        data_index=result.training_data_index,
        custom_data_transformer=custom_data_transformer,
    )
    training_data = training_data_transformation_result.ml_ready_data
    trained_data_transformer = (
        training_data_transformation_result.custom_data_transformer
    )
    utils.save_pipeline_to_cloud_storage(
        trained_pipeline=trained_data_transformer,
        gcp_storage_client=gcp_storage_client,
        gcp_bucket_name=gcp_bucket_name,
        pipeline_name=pipeline_name,
    )
    test_data_transformation_result = _data_transformation(
        fit_data_bool=False,
        id_cols=id_cols,
        x=result.x_test,
        y=result.y_test,
        data_index=result.testing_data_index,
        custom_data_transformer=trained_data_transformer,
    )
    test_data = test_data_transformation_result.ml_ready_data
    return pd.concat([training_data, test_data])


def _create_feature_selection_pipeline(
    id_cols: Sequence[str],
    labels: Sequence[str],
    label_types: Mapping[str, constant.LabelType],
    min_correlation_threshold_with_numeric_labels_for_feature_reduction: (
        float
    ) = 0.1,
) -> pipeline.Pipeline:
  """Create feature selection pipeline.

  Args:
    id_cols: Id columns values for the dataset that identify each row.
    labels: labels/target variables for the dataset.
    label_types: Types of the target variables mapping (i.e. numerical or
      categorical).
    min_correlation_threshold_with_numeric_labels_for_feature_reduction: Minimum
      threshold for correlation between a numeric feature and a numeric target
      variable for the feature to be considered during feature reduction.

  Returns:
    custom_pipeline: Feature selection pipeline.
  """
  return pipeline.Pipeline(
      steps=[
          (
              'feature_selection',
              custom_transformer.FeatureSelector(
                  id_cols=id_cols,
                  labels=labels,
                  label_types=label_types,
                  min_correlation_threshold=min_correlation_threshold_with_numeric_labels_for_feature_reduction,
              ),
          ),
      ]
  )


def _get_feature_selection_pipeline(
    use_prediction_pipeline: bool = True,
    id_cols: Sequence[str] | None = None,
    labels: Sequence[str] | None = None,
    label_types: Mapping[str, constant.LabelType] | None = None,
    min_correlation_threshold_with_numeric_labels_for_feature_reduction: (
        float
    ) = 0.1,
    gcp_storage_client: storage.Client | None = None,
    gcp_bucket_name: str | None = None,
    pipeline_name: str | None = None,
) -> pipeline.Pipeline:
  """Create or load feature selection pipeline.

  Args:
    use_prediction_pipeline: If True, then a pretrained feature selector needs
      to be loaded. Otherwise, a custom transfomer needs to be created.
    id_cols: Id columns values for the dataset that identify each row.
    labels: labels/target variables for the dataset.
    label_types: Types of the target variables mapping (i.e. numerical or
      categorical).
    min_correlation_threshold_with_numeric_labels_for_feature_reduction: Minimum
      threshold for correlation between a numeric feature and a numeric target
      variable for the feature to be considered during feature reduction.
    gcp_storage_client: Google Cloud Storage client for loading pretrained
      pipeline.
    gcp_bucket_name: Name of Google Cloud Storage bucket where pretrained
      pipeline is saved.
    pipeline_name: Name of the pretrained pipeline.

  Returns:
    Customized feature selection pipeline.
  """
  if use_prediction_pipeline:
    if (gcp_storage_client is None) or (not gcp_storage_client):
      raise ValueError(
          'Missing required argument gcp_storage_client for loading pre-trained'
          ' feature selection pipeline.'
      )
    if (gcp_bucket_name is None) or (not gcp_bucket_name):
      raise ValueError(
          'Missing required argument gcp_bucket_name for loading pre-trained'
          ' feature selection pipeline.'
      )
    if (pipeline_name is None) or (not pipeline_name):
      raise ValueError(
          'Missing required argument pipeline_name for loading pre-trained'
          ' feature selection pipeline.'
      )
  else:
    if (id_cols is None) or (not id_cols):
      raise ValueError(
          'Missing required argument id_cols for feature selection pipeline'
          ' creation.'
      )
    if (labels is None) or (not labels):
      raise ValueError(
          'Missing required argument labels for feature selection pipeline'
          ' creation.'
      )
    if (label_types is None) or (not label_types):
      raise ValueError(
          'Missing required argument label_types for feature selection pipeline'
          ' creation.'
      )

  if use_prediction_pipeline:
    try:
      return utils.load_pipeline_from_cloud_storage(
          gcp_storage_client=gcp_storage_client,
          gcp_bucket_name=gcp_bucket_name,
          pipeline_name=pipeline_name,
      )
    except Exception as e:
      raise RuntimeError(
          f'Error loading pre-trained feature selection pipeline: {e}'
      ) from e
  else:
    return _create_feature_selection_pipeline(
        id_cols=id_cols,
        labels=labels,
        label_types=label_types,
        min_correlation_threshold_with_numeric_labels_for_feature_reduction=min_correlation_threshold_with_numeric_labels_for_feature_reduction,
    )


def _feature_selection_training_prediction_pipeline(
    use_prediction_pipeline: bool,
    df: pd.DataFrame,
    gcp_storage_client: storage.Client,
    gcp_bucket_name: str,
    pipeline_name: str,
    id_cols: Sequence[str] | None = None,
    labels: Sequence[str] | None = None,
    label_types: Mapping[str, constant.LabelType] | None = None,
    min_correlation_threshold_with_numeric_labels_for_feature_reduction: (
        float
    ) = 0.1,
) -> Mapping[str, pd.DataFrame]:
  """Preprocess ML data for prediction or training pipelines.

  Args:
    use_prediction_pipeline: Whether to preprocess data for prediction pipeline.
    df: Input data.
    gcp_storage_client: Google Cloud Storage client.
    gcp_bucket_name: Google Cloud Storage bucket name where trained pipeline has
      been or will be stored.
    pipeline_name: Name of the pipeline file (without the .pkl extension)that
      has been or will be saved.
    id_cols: Id columns values for the dataset that identify each row.
    labels: labels/target variables for the dataset.
    label_types: Types of the target variables mapping (i.e. numerical or
      categorical).
    min_correlation_threshold_with_numeric_labels_for_feature_reduction: Minimum
      threshold for correlation between a numeric feature and a numeric target
      variable for the feature to be considered during feature reduction.

  Returns:
    Data with selected features.
  """

  custom_feature_selector = _get_feature_selection_pipeline(
      use_prediction_pipeline=use_prediction_pipeline,
      id_cols=id_cols,
      labels=labels,
      label_types=label_types,
      min_correlation_threshold_with_numeric_labels_for_feature_reduction=min_correlation_threshold_with_numeric_labels_for_feature_reduction,
      gcp_storage_client=gcp_storage_client,
      gcp_bucket_name=gcp_bucket_name,
      pipeline_name=pipeline_name,
  )
  if not use_prediction_pipeline:
    custom_feature_selector.fit(df)
    custom_feature_selector.named_steps['feature_selection'].fitted_ = True
    utils.save_pipeline_to_cloud_storage(
        trained_pipeline=custom_feature_selector,
        gcp_storage_client=gcp_storage_client,
        gcp_bucket_name=gcp_bucket_name,
        pipeline_name=pipeline_name,
    )
  return custom_feature_selector.transform(df)


def data_preprocessing_for_ml(
    use_prediction_pipeline: bool,
    df: pd.DataFrame,
    bigquery_client: bigquery.Client,
    gcp_storage_client: storage.Client,
    gcp_bucket_name: str,
    dataset_id: str,
    table_name: str,
    id_cols: Sequence[str],
    numeric_labels: Sequence[str] | None = None,
    categorical_labels: Sequence[str] | None = None,
    train_test_split_order_by_cols: Sequence[str] | None = None,
    train_test_split_asc_order: bool | None = False,
    train_test_split_test_size_proportion: float | None = 0.1,
    location: str = 'europe-west4',
    feature_selection_pipeline_name: str = 'feature_selection_pipeline',
    data_processing_pipeline_name: str = 'data_processing_pipeline',
    invalid_value_threshold_for_row_removal: float = 0.5,
    invalid_value_threshold_for_column_removal: float = 0.95,
    min_correlation_threshold_with_numeric_labels_for_feature_reduction: (
        float | None
    ) = 0.1,
) -> Mapping[str, pd.DataFrame]:
  """Process data for BigQuery ML and export processed data to BigQuery.

  The processing involves converting data to the right data types, removing
  columns & rows with high amount of invalid data, feature selection, splitting
  data into training & testing, scaling data and resampling data when there's
  significant inbalance in the label. When use_prediction_pipeline is True, data
  is processed for prediction pipeline. In this case, pipelines that are
  pretrained for feature selection and data processing is loaded from the Google
  Cloud Storage bucket for implementating the task. Otherwise, data is processed
  for training pipeline. In this case, pipelines for feature selection and data
  processing are created and saved to the Google Cloud Storage bucket.

  Args:
    use_prediction_pipeline: Whether to preprocess data for prediction pipeline.
    df: Input dataframe with each transaction as a row and features and labels
      as columns.
    bigquery_client: Bigquery client for loading dataframe to BigQuery.
    gcp_storage_client: Google Cloud Storage client for saving trained pipeline.
    gcp_bucket_name: Name of Google Cloud Storage bucket where trained pipeline
      is saved.
    dataset_id: Bigquery dataset id of the destination ML ready table.
    table_name: Bigquery table name of the destination ML ready table.
    id_cols: Columns from the input dataframe that uniquely identify each row.
    numeric_labels: Numerical labels.
    categorical_labels: Categorical labels.
    train_test_split_order_by_cols: Column names used to order the rows in the
      input dataframe for splitting data into train and test sets.
    train_test_split_asc_order: Whether to order dataframe in ascending order
      based on train_test_split_col before splitting the data.
    train_test_split_test_size_proportion: Proportion of the input dataframe to
      be used as the test set. Note: If there are more than 100k rows in the
      input dataframe, the test set will be 10k rows automatically.
    location: Location of the Bigquery table (e.g. 'europe-west4',
      'us-central1', 'us') for exporting the cleaned dataframe. Given the
      solution is built in EMEA, the default location is set to 'europe-west4'.
    feature_selection_pipeline_name: Name of the feature selection pipeline file
      (without the .pkl extension) that has been or will be saved.
    data_processing_pipeline_name: Name of the data processing pipeline file
      (without the .pkl extension) that has been or will be saved.
    invalid_value_threshold_for_row_removal: The minimum threshold for the
      percentage of columns with invalid values in a row to be considered during
      data cleaning.
    invalid_value_threshold_for_column_removal: The minimum threshold for the
      percentage of invalid values in a column to be considered during feature
      reduction.
    min_correlation_threshold_with_numeric_labels_for_feature_reduction: Minimum
      threshold for correlation between a numeric feature and a numeric target
      variable for the feature to be considered during feature reduction.

  Returns:
    Mapping of label and preprocessed data with selected features.

  Raises:
    ValueError: If required arguments are missing or labels have only one unique
      value.
  """
  if use_prediction_pipeline:
    if (feature_selection_pipeline_name is None) or (
        not feature_selection_pipeline_name
    ):
      raise ValueError(
          'Missing required argument feature_selection_pipeline_name for'
          ' trained pipeline in prediction.'
      )
    if (data_processing_pipeline_name is None) or (
        not data_processing_pipeline_name
    ):
      raise ValueError(
          'Missing required argument data_processing_pipeline_name for'
          ' trained pipeline in prediction.'
      )
  else:
    if (numeric_labels is None) or (not numeric_labels):
      raise ValueError(
          'Missing required argument numeric_labels for creating pipeline for'
          ' training.'
      )
    if (categorical_labels is None) or (not categorical_labels):
      raise ValueError(
          'Missing required argument categorical_labels for creating pipeline'
          ' for training.'
      )
  df = utils.clean_dataframe_column_names(df.copy())
  cleaned_data = _clean_and_handle_invalid_data(
      df=df,
      bigquery_client=bigquery_client,
      dataset_id=dataset_id,
      table_name=table_name,
      id_cols=id_cols,
      numeric_labels=numeric_labels,
      categorical_labels=categorical_labels,
      location=location,
      invalid_value_threshold_for_row_removal=invalid_value_threshold_for_row_removal,
      invalid_value_threshold_for_column_removal=invalid_value_threshold_for_column_removal,
      use_prediction_pipeline=use_prediction_pipeline,
  )

  result = _get_labels_and_types(numeric_labels, categorical_labels)
  if use_prediction_pipeline:
    numerical_features, categorical_features = None, None
  else:
    _, numerical_features, categorical_features = (
        custom_transformer.feature_cols(cleaned_data, id_cols, result.labels)
    )

  if not use_prediction_pipeline:
    for col in cleaned_data.columns:
      if cleaned_data[col].nunique() == 1:
        cleaned_data.drop([col], axis=1, inplace=True)
    for label in [*numeric_labels, *categorical_labels]:
      if cleaned_data[label].nunique() == 1:
        raise ValueError(
            f'Label {label} only has one unique value. Please validate your'
            ' data source.'
        )

  dfs_with_selected_features = _feature_selection_training_prediction_pipeline(
      use_prediction_pipeline=use_prediction_pipeline,
      df=cleaned_data,
      gcp_storage_client=gcp_storage_client,
      gcp_bucket_name=gcp_bucket_name,
      pipeline_name=feature_selection_pipeline_name,
      id_cols=id_cols,
      labels=result.labels,
      label_types=result.label_types,
      min_correlation_threshold_with_numeric_labels_for_feature_reduction=min_correlation_threshold_with_numeric_labels_for_feature_reduction,
  )
  ml_ready_dfs = dict()
  for label, ml_df in dfs_with_selected_features.items():
    ml_ready_data = _data_preprocessing_training_prediction_pipeline(
        use_prediction_pipeline=use_prediction_pipeline,
        df=ml_df,
        id_cols=id_cols,
        label=label,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        label_type=result.label_types[label],
        train_test_split_order_by_cols=train_test_split_order_by_cols,
        train_test_split_asc_order=train_test_split_asc_order,
        train_test_split_test_size_proportion=train_test_split_test_size_proportion,
        gcp_storage_client=gcp_storage_client,
        gcp_bucket_name=gcp_bucket_name,
        pipeline_name=f'{data_processing_pipeline_name}_for_label_{label}',
    )
    ml_ready_data = utils.clean_dataframe_column_names(ml_ready_data)
    ml_ready_dfs[label] = ml_ready_data
    if use_prediction_pipeline:
      ml_ready_data_table_name = (
          'PREDICTION_ml_data_{}_with_target_variable_{}'.format(
              table_name, label
          )
      )
    else:
      ml_ready_data_table_name = (
          'TRAINING_ml_data_{}_with_target_variable_{}'.format(
              table_name, label
          )
      )

    utils.run_load_table_to_bigquery(
        data=ml_ready_data,
        bigquery_client=bigquery_client,
        dataset_id=dataset_id,
        table_name=ml_ready_data_table_name,
        location=location,
    )
  return ml_ready_dfs


def create_ml_ready_data_for_preprocessed_data_provided_by_user(
    preprocessed_table_name_by_user: str,
    bigquery_client: bigquery.Client,
    project_id: str,
    dataset_id: str,
    refund_value_col: str,
    refund_flag_col: str,
    refund_proportion_col: str,
    use_prediction_pipeline: bool,
    transaction_id_col: str,
    train_test_split_test_size_proportion: float | None = 0.1,
) -> None:
  """Creates ML ready data for preprocessed data provided by user.

  Args:
    preprocessed_table_name_by_user: Name of the preprocessed table provided by
      the user.
    bigquery_client: Bigquery client for loading dataframe to BigQuery.
    project_id: Project id of the destination ML ready table.
    dataset_id: Dataset id of the destination ML ready table.
    refund_value_col: Column name of the refund value column.
    refund_flag_col: Column name of the refund flag column.
    refund_proportion_col: Column name of the refund proportion column.
    use_prediction_pipeline: Whether to create ML ready data for prediction
      pipeline.
    transaction_id_col: Column name of the transaction id column.
    train_test_split_test_size_proportion: Proportion of the data to be used as
      the test set.
  """
  query_for_preparing_user_provided_data = """
  CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{ml_ready_data_table_name}`
  AS
  SELECT
    *
    EXCEPT({refund_value_col}, {refund_flag_col}, {refund_proportion_col}),
    {label_col},
    CASE
    WHEN MOD(FARM_FINGERPRINT(CAST({transaction_id_col} AS STRING)), 100) < ({train_test_split_test_size_proportion} * 100) 
    THEN 'test' 
    ELSE 'train'
    END AS train_test
  FROM
    `{project_id}.{dataset_id}.{preprocessed_table_name_by_user}`;
  """
  for label in [refund_value_col, refund_flag_col, refund_proportion_col]:
    if use_prediction_pipeline:
      ml_ready_data_table_name = (
          'PREDICTION_ml_data_{}_with_target_variable_{}'.format(
              preprocessed_table_name_by_user, label
          )
      )
    else:
      ml_ready_data_table_name = (
          'TRAINING_ml_data_{}_with_target_variable_{}'.format(
              preprocessed_table_name_by_user, label
          )
      )
    query = query_for_preparing_user_provided_data.format(
        project_id=project_id,
        dataset_id=dataset_id,
        ml_ready_data_table_name=ml_ready_data_table_name,
        preprocessed_table_name_by_user=preprocessed_table_name_by_user,
        label_col=label,
        refund_value_col=refund_value_col,
        refund_flag_col=refund_flag_col,
        refund_proportion_col=refund_proportion_col,
        transaction_id_col=transaction_id_col,
        train_test_split_test_size_proportion=train_test_split_test_size_proportion,
    )
    bigquery_client.query(query).result()
