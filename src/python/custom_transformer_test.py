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

import os

import pandas as pd
from imblearn import over_sampling
from imblearn import under_sampling

import os
import os
from absl.testing import absltest
from product_return_predictor.src.python import constant
from product_return_predictor.src.python import custom_transformer


_ID_COLS = ['transaction_id', 'transaction_date']
_LABELS = ['refund_value', 'refund_flag', 'refund_proportion']
_NUMERIC_FEATURES_FOR_CORRELATION_TEST = [
    'transaction_value',
    'shipping_value',
    'item_count',
    'min_price',
    'max_price',
    'event_count_click',
    'event_count_first_visit',
    'event_count_purchase',
]
_NUMERIC_LABELS = ['refund_value', 'refund_proportion']


def _read_csv(path: str) -> pd.DataFrame:
  path = os.path.join(
      'product_return_predictor/src/python/test_data/',
      path,
  )
  with open(path) as f:
    return pd.read_csv(f)


class CustomTransformTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.input_data_for_data_cleaning_feature_selection = _read_csv(
        'test_input_data_for_data_cleaning_feature_selection_step.csv'
    )
    self.input_data_for_data_cleaning_feature_selection['transaction_date'] = (
        pd.to_datetime(
            self.input_data_for_data_cleaning_feature_selection[
                'transaction_date'
            ]
        )
    )
    self.input_data_for_data_cleaning_feature_selection['transaction_id'] = (
        self.input_data_for_data_cleaning_feature_selection[
            'transaction_id'
        ].astype('str')
    )

  def test_resampling_strategy_single_class_passthrough(self):
    y = pd.Series([0, 0, 0])
    result = custom_transformer._resampling_strategy(
        y, constant.LabelType.NUMERICAL
    )
    self.assertEqual(result, 'passthrough')

  def test_resampling_strategy_balanced_classes_passthrough(self):
    y = pd.Series([0, 1, 0, 1])
    result = custom_transformer._resampling_strategy(
        y, constant.LabelType.CATEGORICAL
    )
    self.assertEqual(result, 'passthrough')

  def test_resampling_strategy_imbalanced_small_data_over_sampling(self):
    y = pd.Series([0, 0, 0, 0, 0, 0, 1])
    result = custom_transformer._resampling_strategy(
        y, constant.LabelType.NUMERICAL
    )
    self.assertIsInstance(result, over_sampling.RandomOverSampler)

  def test_resampling_strategy_imbalanced_large_data_under_sampling(self):
    y = pd.Series([0] * 50000 + [1] * 50)
    result = custom_transformer._resampling_strategy(
        y, constant.LabelType.CATEGORICAL
    )
    self.assertIsInstance(result, under_sampling.RandomUnderSampler)

  def test_resampling_transformer_fit_transform_with_balanced_class(self):
    x = pd.DataFrame({'feature1': [1, 2, 3, 4], 'feature2': [5, 6, 7, 8]})
    y = pd.Series([0, 0, 1, 1])

    transformer = custom_transformer.ResamplingTransformer(
        label_type=constant.LabelType.CATEGORICAL
    )
    x_transformed = transformer.fit(x, y).transform(x, y)
    pd.testing.assert_frame_equal(x_transformed, x)

  def test_resampling_transformer_fit_transform_with_imbalanced_class(self):
    x = pd.DataFrame({'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    y_imbalanced = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    transformer = custom_transformer.ResamplingTransformer(
        label_type=constant.LabelType.CATEGORICAL
    )
    x_resampled = transformer.fit(x, y_imbalanced).transform(x, y_imbalanced)
    self.assertGreater(len(x_resampled), len(x))

  def test_resampling_transformer_transform_before_fit(self):
    x = pd.DataFrame({'feature1': [1, 2, 3, 4], 'feature2': [5, 6, 7, 8]})
    y = pd.Series([0, 0, 1, 1])

    transformer = custom_transformer.ResamplingTransformer(
        label_type=constant.LabelType.CATEGORICAL
    )
    with self.assertRaises(ValueError):
      transformer.transform(x, y)

  def test_date_string_numeric_cols_from_input_dataframe(self):
    self.assertEqual(
        custom_transformer.date_string_numeric_cols_from_input_dataframe(
            self.input_data_for_data_cleaning_feature_selection
        ),
        (
            ['transaction_date'],
            ['transaction_id', 'past_product_returned_descriptions'],
            [
                'transaction_value',
                'refund_value',
                'refund_flag',
                'refund_proportion',
                'shipping_value',
                'item_count',
                'min_price',
                'max_price',
                'event_count_click',
                'event_count_first_visit',
                'event_count_purchase',
            ],
        ),
    )

  def test_correct_feature_cols(self):
    output = custom_transformer.feature_cols(
        df=self.input_data_for_data_cleaning_feature_selection,
        id_cols=_ID_COLS,
        labels=_LABELS,
    )
    self.assertSequenceEqual(
        output,
        (
            [
                'transaction_value',
                'shipping_value',
                'item_count',
                'min_price',
                'max_price',
                'event_count_click',
                'event_count_first_visit',
                'event_count_purchase',
                'past_product_returned_descriptions',
            ],
            [
                'transaction_value',
                'shipping_value',
                'item_count',
                'min_price',
                'max_price',
                'event_count_click',
                'event_count_first_visit',
                'event_count_purchase',
            ],
            ['past_product_returned_descriptions'],
        ),
    )

  def test_identify_correct_correlated_numeric_features_with_numeric_target_variables(
      self,
  ):
    output = custom_transformer._identify_correlated_numeric_features_with_numeric_target_variables(
        df=self.input_data_for_data_cleaning_feature_selection,
        numeric_features=_NUMERIC_FEATURES_FOR_CORRELATION_TEST,
        labels=_NUMERIC_LABELS,
        min_correlation_threshold_with_numeric_labels_for_feature_reduction=0.5,
    )
    expected_output = {
        'refund_value': [
            'transaction_value',
            'item_count',
            'min_price',
            'max_price',
            'event_count_first_visit',
            'refund_value',
        ],
        'refund_proportion': [
            'transaction_value',
            'item_count',
            'min_price',
            'max_price',
            'event_count_first_visit',
            'refund_proportion',
        ],
    }

    self.assertDictEqual(output, expected_output)

  def test_identify_correlated_numeric_features_with_binary_target_variables_no_correlation(
      self,
  ):
    data_with_no_feature_label_correlation = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [5, 4, 3, 2, 1],
        'label': [0, 1, 0, 1, 0],
    }
    df = pd.DataFrame(data_with_no_feature_label_correlation)
    numeric_features = ['feature1', 'feature2']
    labels = ['label']
    result = custom_transformer._identify_correlated_numeric_features_with_binary_target_variables(
        df, numeric_features, labels
    )
    expected_result = {'label': []}
    self.assertEqual(result, expected_result)

  def test_identify_correlated_numeric_features_with_binary_target_variables_correlation_exists(
      self,
  ):
    data_with_feature1_correlated_with_label = {
        'feature1': [0.5, 0.5, 1, 1, 4, 4, 7],
        'feature2': [5, 4, 2, 1, 3, 2, 1],
        'label': [0, 0, 0, 0, 1, 1, 1],
    }
    df = pd.DataFrame(data_with_feature1_correlated_with_label)
    numeric_features = ['feature1', 'feature2']
    labels = ['label']
    result = custom_transformer._identify_correlated_numeric_features_with_binary_target_variables(
        df, numeric_features, labels
    )
    expected_result = {'label': ['feature1']}
    self.assertEqual(result, expected_result)

  def test_identify_correlated_numeric_features_with_binary_target_variables_multiple_labels(
      self,
  ):
    data_with_multiple_binary_labels = {
        'feature1': [0.5, 0.5, 1, 1, 4, 4, 7],
        'feature2': [5, 4, 2, 1, 3, 2, 1],
        'label1': [0, 0, 0, 0, 1, 1, 1],
        'label2': [0, 1, 0, 1, 0, 1, 0],
    }
    df = pd.DataFrame(data_with_multiple_binary_labels)

    numeric_features = ['feature1', 'feature2']
    labels = ['label1', 'label2']

    result = custom_transformer._identify_correlated_numeric_features_with_binary_target_variables(
        df, numeric_features, labels
    )

    expected_result = {'label1': ['feature1'], 'label2': []}
    self.assertEqual(result, expected_result)

  def test_feature_selection(self):
    df = pd.DataFrame({
        'id_col': [1, 2, 3],
        'numeric_feature_1': [10, 20, 30],
        'numeric_feature_2': [5, 15, 25],
        'string_feature': ['A', 'B', 'A'],
        'numeric_label': [0.5, 1.2, 0.8],
        'categorical_label': [0, 1, 1],
    })
    id_cols = ['id_col']
    labels = ['numeric_label', 'categorical_label']
    label_types = {
        'numeric_label': constant.LabelType.NUMERICAL,
        'categorical_label': constant.LabelType.CATEGORICAL,
    }
    min_correlation_threshold = 0.1
    result = custom_transformer.feature_selection(
        df,
        id_cols,
        labels,
        label_types,
        min_correlation_threshold,
    )

    expected_numeric_features = [
        'numeric_feature_1',
        'numeric_feature_2',
        'numeric_label',
        'id_col',
        'string_feature',
    ]
    self.assertSetEqual(
        set(result['numeric_label'].columns), set(expected_numeric_features)
    )

  def test_feature_selector_fit_transform(self):
    feature_selector_sample_data = {
        'id': [1, 2, 3, 4],
        'label_numeric': [10, 20, 30, 40],
        'label_categorical': [0, 1, 1, 0],
        'feature_numeric_1': [1.1, 2.2, 3.3, 4.4],
        'feature_numeric_2': [5, 10, 15, 20],
        'feature_string': ['X', 'Y', 'Z', 'W'],
    }
    df = pd.DataFrame(feature_selector_sample_data)

    selector = custom_transformer.FeatureSelector(
        id_cols=['id'],
        labels=['label_numeric', 'label_categorical'],
        label_types={
            'label_numeric': constant.LabelType.NUMERICAL,
            'label_categorical': constant.LabelType.CATEGORICAL,
        },
        min_correlation_threshold=0.5,
    )
    trained_selector = selector.fit(df)
    expected_output = {
        'label_numeric': pd.Index(
            [
                'feature_numeric_1',
                'feature_numeric_2',
                'label_numeric',
                'id',
                'feature_string',
            ],
            dtype='object',
            name='index',
        ),
        'label_categorical': pd.Index(
            ['label_categorical', 'id', 'feature_string'],
            dtype='object',
            name='index',
        ),
    }
    self.assertIsInstance(trained_selector.selected_features, dict)
    self.assertSequenceEqual(
        list(trained_selector.selected_features.keys()),
        list(expected_output.keys()),
    )
    self.assertSetEqual(
        set(trained_selector.selected_features['label_numeric']),
        set(expected_output['label_numeric']),
    )
    self.assertSetEqual(
        set(trained_selector.selected_features['label_categorical']),
        set(expected_output['label_categorical']),
    )
    result = trained_selector.transform(df)
    pd.testing.assert_frame_equal(
        result['label_numeric'],
        df[[
            'feature_numeric_1',
            'feature_numeric_2',
            'label_numeric',
            'id',
            'feature_string',
        ]],
    )
    pd.testing.assert_frame_equal(
        result['label_categorical'],
        df[['label_categorical', 'id', 'feature_string']],
    )


if __name__ == '__main__':
  absltest.main()
