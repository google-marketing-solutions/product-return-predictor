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

"""Resampling & feature selection custom transformers.

There are two main custom sklearn transformers created: ResamplingTransformer
used for resampling training data in case of class imbalance and FeatureSelector
used for feature selection.
"""

from collections.abc import Mapping, Sequence
import dataclasses

import numpy as np
import pandas as pd
from imblearn import over_sampling
from imblearn import under_sampling
from scipy import stats
from sklearn import base

from product_return_predictor.product_return_predictor import constant


def date_string_numeric_cols_from_input_dataframe(
    df: pd.DataFrame,
) -> tuple[Sequence[str], Sequence[str], Sequence[str]]:
  """Return string, numeric and date columns from the input dataframe.

  Args:
    df: Input pandas dataframe.

  Returns:
    date_cols: List of columns from the input dataframe with date data type.
    string_cols: List of columns from the input dataframe with string data type.
    numeric_cols: List of columns from the input dataframe with numeric data
    type.
  """
  column_dtypes = pd.DataFrame(df.dtypes, columns=['data type'])
  column_dtypes['data type'] = column_dtypes['data type'].astype('string')
  date_cols = list(
      column_dtypes.loc[column_dtypes['data type'].str.contains('date')].index
  )
  numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

  string_cols = [
      col for col in df.columns if col not in [*date_cols, *numeric_cols]
  ]
  return date_cols, string_cols, numeric_cols


def feature_cols(
    df: pd.DataFrame,
    id_cols: Sequence[str],
    labels: Sequence[str],
) -> tuple[Sequence[str], Sequence[str], Sequence[str]]:
  """Get numerical and string features based on labels and id cols.

  Args:
    df: Input pandas dataframe.
    id_cols: List of columns from the input dataframe that uniquely identify
      each row.
    labels: Columns that represent the target variables in the input dataframe.

  Returns:
    A tuple of (features, numeric_features, string_features), where features
    represent all features and numeric_features and string_features represent
    features of each type from the input dataframe.
  """
  _, string_cols, numeric_cols = date_string_numeric_cols_from_input_dataframe(
      df
  )
  features = [col for col in df.columns if col not in [*id_cols, *labels]]
  numeric_features = [col for col in features if col in numeric_cols]
  string_features = [col for col in features if col in string_cols]
  return features, numeric_features, string_features


def _identify_correlated_numeric_features_with_numeric_target_variables(
    df: pd.DataFrame,
    numeric_features: Sequence[str],
    labels: Sequence[str],
    min_correlation_threshold_with_numeric_labels_for_feature_reduction: float = 0.1,
) -> Mapping[str, Sequence[str]]:
  """Identify features correlated with numeric labels based on Pearson score.

  Args:
    df: Input pandas dataframe.
    numeric_features: List of columns from the input dataframe that are numeric
      features for product return prediction.
    labels: Columns that represent the target variables in the input dataframe.
    min_correlation_threshold_with_numeric_labels_for_feature_reduction: Minimum
      threshold for correlation between a numeric feature and a numeric target
      variable for the feature to be considered during feature reduction.

  Returns:
    Dictionary with each numeric target variable as the key and a list of
    numeric features that are correlated with that target variable (based on
    pearson correlation score threshold) as the value.
  """
  numeric_features_with_correlation_with_numeric_target_variables = dict()
  for numeric_label in labels:
    numeric_features_correlation_df = df[
        [*numeric_features, numeric_label]
    ].corr()
    cols_to_keep = list(
        numeric_features_correlation_df.loc[
            np.abs(numeric_features_correlation_df.loc[numeric_label])
            >= min_correlation_threshold_with_numeric_labels_for_feature_reduction
        ].index
    )
    numeric_features_with_correlation_with_numeric_target_variables[
        numeric_label
    ] = cols_to_keep
  return numeric_features_with_correlation_with_numeric_target_variables


def _identify_correlated_numeric_features_with_binary_target_variables(
    df: pd.DataFrame,
    numeric_features: Sequence[str],
    labels: Sequence[str],
) -> Mapping[str, Sequence[str]]:
  """Identify significant numeric features based on ANOVA test.

  Args:
    df: Input pandas dataframe.
    numeric_features: List of columns from the input dataframe that are numeric
      features for product return prediction.
    labels: Columns that represent the binary categorical target variables in
      the input dataframe (e.g. refund_flag).

  Returns:
    Dictionary with each binary categorical target variable as the key and a
    list of numeric features that are correlated with that target variable
    (based on ANOVA test p value <=0.05) as the value.
  """
  df[labels] = df[labels].astype('float')
  numeric_features_with_correlation_with_categorical_target_variables = dict()
  for binary_label in labels:
    p = stats.kruskal(
        df.loc[df[binary_label] == 0, numeric_features],
        df.loc[df[binary_label] == 1, numeric_features],
    ).pvalue
    feature_importance_df = pd.DataFrame(
        list(zip(numeric_features, p, strict=True)),
        columns=['feature', 'p value'],
    )
    significant_numeric_features_for_categorical_target_for_feature_reduction = list(
        feature_importance_df.loc[
            feature_importance_df['p value'] <= 0.05, 'feature'
        ]
    )
    numeric_features_with_correlation_with_categorical_target_variables[
        binary_label
    ] = significant_numeric_features_for_categorical_target_for_feature_reduction
  return numeric_features_with_correlation_with_categorical_target_variables


def _resampling_strategy(
    y: pd.Series,
    label_type: constant.LabelType,
) -> str | over_sampling.RandomOverSampler | under_sampling.RandomUnderSampler:
  """Determines resampling strategy based on class imbalance and data size.

  When there's huge discrepancy between minority and majority class, we use
  resampling strategy, otherwise passthrough. When there's good amount of data,
  use undersampling, otherwise use oversampling.

  Args:
      y: The target variable array.
      label_type: Type of the target variable (i.e. numerical or categorical).

  Returns:
      A resampling object (RandomOverSampler or RandomUnderSampler) or
      'passthrough' if no resampling is needed.
  """
  if label_type == constant.LabelType.NUMERICAL:
    y_binary = (y > 0).astype(int)
  else:
    y_binary = y

  class_counts = y_binary.value_counts()

  if len(class_counts) == 1:
    return 'passthrough'

  majority_class_count = class_counts.max()
  minority_class_count = class_counts.min()

  if (
      majority_class_count / minority_class_count
      > constant.MAJORITY_TO_MINORITY_CLASS_RATIO_THRESHOLD
  ):
    if len(y) > constant.MINIMUM_DATA_POINTS_FOR_DOWNSAMPLING:
      return under_sampling.RandomUnderSampler(sampling_strategy='majority')
    else:
      return over_sampling.RandomOverSampler(sampling_strategy='minority')
  else:
    return 'passthrough'


@dataclasses.dataclass
class ResamplingTransformer(base.BaseEstimator, base.TransformerMixin):
  """Customized resampling transformer.

  Attributes:
    label_type: Type of the target variable (i.e. numerical or categorical).
    resampler: The resampling object (RandomOverSampler or RandomUnderSampler)
      or 'passthrough' if no resampling is needed.
  """

  label_type: constant.LabelType
  resampler: (
      str | over_sampling.RandomOverSampler | under_sampling.RandomUnderSampler
  ) = None

  def fit(self, x: pd.DataFrame, y: pd.Series) -> 'ResamplingTransformer':
    """Fits the resampler with data.

    Args:
      x: Features.
      y: Target variable.

    Returns:
      ResamplingTransformer with resampler fitted.
    """
    self.resampler = _resampling_strategy(y, self.label_type)
    return self

  def transform(
      self, x: pd.DataFrame, y: pd.Series | None = None
  ) -> pd.DataFrame:
    """Transform the features using fitted resampler.

    Args:
      x: Features.
      y: Target variable. If y is None, the resampling step will be skipped.

    Returns:
      Transformed features.

    Raises:
      ValueError: If transform is called before fit.
    """
    if y is None:
      return x
    elif self.resampler is None:
      raise ValueError('ResamplingTransformer must be fit before transform.')
    elif isinstance(self.resampler, str):
      return x
    else:
      x_resampled, _ = self.resampler.fit_resample(x, y)
      return x_resampled


def feature_selection(
    df: pd.DataFrame,
    id_cols: Sequence[str],
    labels: Sequence[str],
    label_types: Mapping[str, constant.LabelType],
    min_correlation_threshold_with_numeric_labels_for_feature_reduction: float = 0.1,
) -> Mapping[str, pd.DataFrame]:
  """Select features for product return prediction model training.

  The function checks whether numeric features are relevant for given labels.
  For numeric label, it uses minimum correlation threshold to remove features
  with low pearson correlation with numeric label. For categorical label, it
  uses F score (p value >0.05) to remove insignificant features.

  Args:
    df: Input pandas dataframe.
    id_cols: List of columns from the input dataframe that uniquely identify
      each row.
    labels: Column names of the target variables.
    label_types: Mapping of label types (numerical or categorical).
    min_correlation_threshold_with_numeric_labels_for_feature_reduction: Minimum
      threshold for correlation between a numeric feature and a numeric target
      variable for the feature to be considered during feature reduction.

  Returns:
    Dictionary with label name as key and machine learning ready dataframe with
    selected features as the value.
  """
  df_copy = df.copy()
  _, numeric_features, string_features = feature_cols(df_copy, id_cols, labels)
  ml_dfs_with_selected_features = dict()
  for label in labels:
    if label_types[label] == constant.LabelType.NUMERICAL:
      numeric_features_with_correlation_with_numeric_target_variables = _identify_correlated_numeric_features_with_numeric_target_variables(
          df_copy,
          numeric_features,
          [label],
          min_correlation_threshold_with_numeric_labels_for_feature_reduction,
      )
      numeric_feature_columns = (
          numeric_features_with_correlation_with_numeric_target_variables[label]
      )
    else:
      numeric_features_with_correlation_with_categorical_target_variables = (
          _identify_correlated_numeric_features_with_binary_target_variables(
              df_copy, numeric_features, [label]
          )
      )
      numeric_feature_columns = [
          *numeric_features_with_correlation_with_categorical_target_variables[
              label
          ],
          label,
      ]

    selected_feature_columns = [
        *numeric_feature_columns,
        *id_cols,
        *string_features,
    ]
    ml_dfs_with_selected_features[label] = df_copy[selected_feature_columns]
  return ml_dfs_with_selected_features


@dataclasses.dataclass
class FeatureSelector(base.BaseEstimator, base.TransformerMixin):
  """Customized Feature Selector.

  Attributes:
    id_cols: Columns from the input dataframe that uniquely identify each row.
    labels: Column names of the target variables.
    label_types: Mapping of label types (numerical or categorical).
    min_correlation_threshold_with_numeric_labels_for_feature_reduction: Minimum
      threshold of pearson correlation between a numeric feature and a numeric
      target variable for the feature to be considered during feature selection.
    selected_features: Datasets with selected features for each label.
  """

  id_cols: Sequence[str]
  labels: Sequence[str]
  label_types: Mapping[str, constant.LabelType]
  min_correlation_threshold: float
  selected_features: Mapping[str, pd.Index] = dataclasses.field(
      default_factory=dict
  )

  def fit(
      self, x: pd.DataFrame, y: pd.Series | None = None
  ) -> 'FeatureSelector':
    """Fit data and identify important features to keep.

    Args:
      x: Input dataset with features and labels.
      y: Label data (optional). Scikit learn estimators require y to be present
        during fit. However, y is not used in this feature selector.

    Returns:
      FeatureSelector with selected features.
    """
    dfs = feature_selection(
        df=x,
        id_cols=self.id_cols,
        labels=self.labels,
        label_types=self.label_types,
        min_correlation_threshold_with_numeric_labels_for_feature_reduction=self.min_correlation_threshold,
    )
    self.selected_features = {
        label: dfs[label].columns for label in self.labels
        }
    return self

  def transform(self, x: pd.DataFrame) -> Mapping[str, pd.DataFrame]:
    """Transform the features using fitted feature selector.

    Args:
      x: Input dataset with features and labels.

    Returns:
      Dictionary with label as key and dataframe with selected features as
      value.

    Raises:
      ValueError: If transform is called before fit or data to be transformed
      does not have the same columns as fitted data.
    """
    if self.selected_features is None:
      raise ValueError('FeatureSelector must be fit before transform.')
    for _, features in self.selected_features.items():
      if not features.isin(x.columns).all():
        raise ValueError(
            'The data to be transformed does not have the same columns as'
            ' fitted data.'
        )

    return {label: x[self.selected_features[label]] for label in self.labels}
