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
import numpy as np
import pandas as pd
from sklearn import compose
from sklearn import pipeline
from sklearn import preprocessing

import os
import os
from absl.testing import absltest
from product_return_predictor.src.python import constant
from product_return_predictor.src.python import custom_transformer
from product_return_predictor.src.python import data_cleaning_feature_selection
from product_return_predictor.src.python import utils


_STRING_COLS = ['past_product_returned_descriptions', 'transaction_id']
_NUMERIC_COLS = [
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
]
_DATE_COLS = ['transaction_date']


def _read_csv(path: str) -> pd.DataFrame:
  path = (
      'product_return_predictor/src/python/test_data/'
      + path
  )
  with open(path) as f:
    return pd.read_csv(f)

class DataCleaningFeatureSelectionTest(absltest.TestCase):

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

    self.input_data_for_data_cleaning_feature_selection_with_more_rows = _read_csv(
        'test_input_data_for_data_cleaning_feature_selection_with_more_rows.csv'
    )
    self.input_data_for_data_cleaning_feature_selection_with_more_rows[
        'transaction_date'
    ] = pd.to_datetime(
        self.input_data_for_data_cleaning_feature_selection_with_more_rows[
            'transaction_date'
        ]
    )

    self.train_test_split_input_data = _read_csv(
        'train_test_split_input_data.csv'
    )

    self.id_cols = ['transaction_id', 'transaction_date']
    self.numeric_labels = ['refund_value', 'refund_proportion']
    self.categorical_labels = ['refund_flag']
    self.labels = [*self.numeric_labels, *self.categorical_labels]
    self.train_test_split_order_by_cols = ['transaction_date']
    self.mock_bigquery_client = mock.create_autospec(bigquery.Client)
    self.mock_storage_client = mock.create_autospec(storage.Client)
    self.feature_selection_pipeline = pipeline.Pipeline(
        steps=[(
            'feature_selection',
            custom_transformer.FeatureSelector(
                id_cols=['transaction_id', 'transaction_date'],
                labels=['refund_value', 'refund_proportion', 'refund_flag'],
                label_types={
                    'refund_flag': constant.LabelType.CATEGORICAL,
                    'refund_proportion': constant.LabelType.NUMERICAL,
                    'refund_value': constant.LabelType.NUMERICAL,
                },
                min_correlation_threshold=0.1,
                selected_features={
                    'refund_flag': [
                        'event_count_click',
                        'refund_flag',
                        'transaction_id',
                        'transaction_date',
                    ],
                    'refund_proportion': [
                        'refund_proportion',
                        'transaction_id',
                        'transaction_date',
                    ],
                    'refund_value': [
                        'refund_value',
                        'transaction_id',
                        'transaction_date',
                    ],
                },
            ),
        )]
    )
    self.data_transformation_pipeline = pipeline.Pipeline(
        steps=[
            (
                'preprocessor',
                compose.ColumnTransformer(
                    transformers=[
                        (
                            'cat',
                            pipeline.Pipeline(
                                steps=[(
                                    'onehot',
                                    preprocessing.OneHotEncoder(
                                        handle_unknown='ignore',
                                        sparse_output=False,
                                    ),
                                )]
                            ),
                            [],
                        ),
                        (
                            'num',
                            pipeline.Pipeline(
                                steps=[('scaler', preprocessing.MinMaxScaler())]
                            ),
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
                        ),
                    ]
                ),
            ),
            (
                'resampler',
                custom_transformer.ResamplingTransformer(
                    label_type=constant.LabelType.NUMERICAL,
                    resampler='passthrough',
                ),
            ),
        ]
    )

  def test_convert_columns_to_right_data_type_for_datetime_columns(self):
    converted_df = (
        data_cleaning_feature_selection._convert_columns_to_right_data_type(
            df=self.input_data_for_data_cleaning_feature_selection,
            string_cols=_STRING_COLS,
            date_cols=_DATE_COLS,
            numeric_cols=_NUMERIC_COLS,
        )
    )
    self.assertSetEqual(
        set(converted_df.select_dtypes(include=['datetime64']).columns),
        set(_DATE_COLS),
    )

  def test_convert_columns_to_right_data_type_for_numeric_columns(self):
    converted_df = (
        data_cleaning_feature_selection._convert_columns_to_right_data_type(
            df=self.input_data_for_data_cleaning_feature_selection,
            string_cols=_STRING_COLS,
            date_cols=_DATE_COLS,
            numeric_cols=_NUMERIC_COLS,
        )
    )
    self.assertSetEqual(
        set(converted_df.select_dtypes(include=['float64']).columns),
        set(_NUMERIC_COLS),
    )

  def test_convert_columns_to_right_data_type_for_string_columns(self):
    converted_df = (
        data_cleaning_feature_selection._convert_columns_to_right_data_type(
            df=self.input_data_for_data_cleaning_feature_selection,
            string_cols=_STRING_COLS,
            date_cols=_DATE_COLS,
            numeric_cols=_NUMERIC_COLS,
        )
    )
    self.assertSetEqual(
        set(converted_df.select_dtypes(include=['string']).columns),
        set(_STRING_COLS),
    )

  def test_identify_values_with_invalid_string_datatype_returns_only_true_or_false(
      self,
  ):
    output = data_cleaning_feature_selection._identify_values_with_invalid_string_datatype(
        self.input_data_for_data_cleaning_feature_selection
    )
    unique_values_from_output = pd.unique(output.values.ravel('K'))
    self.assertSetEqual(set(unique_values_from_output), set([False, True]))

  def test_identify_values_with_invalid_string_datatype_returns_same_number_of_rows_columns_as_input(
      self,
  ):
    output = data_cleaning_feature_selection._identify_values_with_invalid_string_datatype(
        self.input_data_for_data_cleaning_feature_selection
    )
    self.assertEqual(
        output.shape, self.input_data_for_data_cleaning_feature_selection.shape
    )

  def test_identify_correct_indices_with_invalid_string_datatype(self):
    output = data_cleaning_feature_selection._identify_values_with_invalid_string_datatype(
        self.input_data_for_data_cleaning_feature_selection
    )
    np.testing.assert_array_equal(
        list(np.where(output)), [np.array([1, 2]), np.array([13, 13])]
    )

  def test_identify_columns_with_many_invalid_or_zero_values(self):
    output = data_cleaning_feature_selection._identify_columns_with_many_invalid_or_zero_values(
        df=self.input_data_for_data_cleaning_feature_selection,
        numeric_cols=_NUMERIC_COLS,
        string_cols=_STRING_COLS,
        invalid_value_threshold_for_column_removal=0.5,
    )
    self.assertSequenceEqual(
        output, ['shipping_value', 'past_product_returned_descriptions']
    )

  def test_identify_correct_zeroes_numeric_datatype(self):
    output = data_cleaning_feature_selection._identify_zeroes_numeric_datatype(
        self.input_data_for_data_cleaning_feature_selection
    )
    np.testing.assert_array_equal(
        np.where(output),
        [
            np.array([0, 0, 0, 0, 0, 1, 1, 2]),
            np.array([3, 4, 5, 6, 11, 6, 13, 13]),
        ],
    )

  def test_identify_correct_rows_with_high_invalid_values(self):
    output = (
        data_cleaning_feature_selection._identify_rows_with_high_invalid_values(
            df=self.input_data_for_data_cleaning_feature_selection,
            numeric_cols=_NUMERIC_COLS,
            string_cols=_STRING_COLS,
            invalid_value_threshold_for_row_removal=0.3,
        )
    )
    np.testing.assert_array_equal(np.where(output), [np.array([0])])

  def test_train_test_split_has_train_test_column(self):
    output = data_cleaning_feature_selection._train_test_split(
        df=self.train_test_split_input_data,
        order_by_col=['transaction_date'],
        test_size_proportion=0.3,
    )
    self.assertIn('train_test', output.columns)

  def test_train_test_split_has_correct_proportion_of_train_test_data(self):
    output = data_cleaning_feature_selection._train_test_split(
        df=self.train_test_split_input_data,
        order_by_col=['transaction_date'],
        test_size_proportion=0.4,
    )
    self.assertLen(output.loc[output['train_test'] == 'train'], 6)
    self.assertLen(output.loc[output['train_test'] == 'test'], 4)

  def test_train_test_split_has_correct_order_of_data(
      self,
  ):
    output = data_cleaning_feature_selection._train_test_split(
        df=self.train_test_split_input_data,
        asc_order=False,
        order_by_col=['transaction_date'],
        test_size_proportion=0.4,
    )
    self.assertSequenceEqual(
        output.loc[
            output['transaction_date'] == '1/1/2021', 'train_test'
        ].unique(),
        np.array(['test']),
    )

  @mock.patch.object(
      data_cleaning_feature_selection,
      '_output_invalid_data_summary',
      autospec=True,
  )
  def test_remove_high_invalid_values_row_columns(self, mock_output_summary):
    data_with_some_invalid_values = {
        'id': [1, 2, 3, 4],
        'feature1': [10, None, 30, 40],
        'feature2': ['A', 'B', None, 'D'],
        'numeric_label': [1.5, 2.0, None, 3.1],
        'categorical_label': [None, 'Y', 'X', None],
    }
    df = pd.DataFrame(data_with_some_invalid_values)

    id_cols = ['id']
    numeric_labels = ['numeric_label']
    categorical_labels = ['categorical_label']
    cleaned_df = (
        data_cleaning_feature_selection._remove_high_invalid_values_row_columns(
            df=df,
            id_cols=id_cols,
            numeric_labels=numeric_labels,
            categorical_labels=categorical_labels,
            invalid_value_threshold_for_row_removal=0.2,
            invalid_value_threshold_for_column_removal=0.2,
        )
    )
    cleaned_df.reset_index(drop=True, inplace=True)

    expected_data = {
        'id': [1, 2, 4],
        'numeric_label': [1.5, 2.0, 3.1],
        'categorical_label': [None, 'Y', None],
    }
    expected_df = pd.DataFrame(expected_data)

    pd.testing.assert_frame_equal(cleaned_df, expected_df)
    mock_output_summary.assert_called_once()

  def test_create_training_testing_array(self):
    data = {
        'id': [1, 2, 3, 4, 5, 6],
        'feature1': [10, 20, 30, 40, 50, 60],
        'feature2': ['A', 'B', 'C', 'D', 'E', 'F'],
        'label': [0, 1, 0, 1, 0, 1],
        'train_test_col': ['train', 'train', 'test', 'test', 'train', 'test'],
    }
    df = pd.DataFrame(data)

    label = 'label'
    train_test_col = 'train_test_col'
    id_cols = ['id']

    result = data_cleaning_feature_selection._create_train_test_features_labels(
        df, label, train_test_col, id_cols
    )
    expected_x_train = pd.DataFrame(
        {'id': [1, 2, 5], 'feature1': [10, 20, 50], 'feature2': ['A', 'B', 'E']}
    )
    expected_x_test = pd.DataFrame(
        {'id': [3, 4, 6], 'feature1': [30, 40, 60], 'feature2': ['C', 'D', 'F']}
    )
    expected_y_train = pd.Series([0, 1, 0], dtype=np.int64, name='label')
    expected_y_test = pd.Series([0, 1, 1], dtype=np.int64, name='label')
    expected_train_index = pd.DataFrame({'id': [1, 2, 5]})
    expected_test_index = pd.DataFrame({'id': [3, 4, 6]})

    pd.testing.assert_frame_equal(result.x_train, expected_x_train)
    pd.testing.assert_frame_equal(result.x_test, expected_x_test)
    pd.testing.assert_series_equal(result.y_train, expected_y_train)
    pd.testing.assert_series_equal(result.y_test, expected_y_test)
    pd.testing.assert_frame_equal(
        result.training_data_index, expected_train_index
    )
    pd.testing.assert_frame_equal(
        result.testing_data_index, expected_test_index
    )

  def test_create_data_transformation_pipeline(self):
    label_type = constant.LabelType.NUMERICAL
    categorical_features = ['color', 'size']
    numerical_features = ['price', 'quantity']
    custom_pipeline = (
        data_cleaning_feature_selection._create_data_transformation_pipeline(
            label_type, categorical_features, numerical_features
        )
    )

    with self.subTest(name='CheckCustomPipelineStructure'):
      self.assertIsInstance(custom_pipeline, pipeline.Pipeline)
      self.assertLen(custom_pipeline.steps, 2)

    with self.subTest(name='CheckCustomTransformerStructure'):
      custom_preprocessor = custom_pipeline.named_steps['preprocessor']
      self.assertIsInstance(custom_preprocessor, compose.ColumnTransformer)
      self.assertLen(custom_preprocessor.transformers, 2)
      categorical_transformer = custom_preprocessor.transformers[0][1]
      self.assertIsInstance(categorical_transformer, pipeline.Pipeline)
      self.assertEqual(categorical_transformer.steps[0][0], 'onehot')
      self.assertIsInstance(
          categorical_transformer.steps[0][1], preprocessing.OneHotEncoder
      )
      numerical_transformer = custom_preprocessor.transformers[1][1]
      self.assertIsInstance(numerical_transformer, pipeline.Pipeline)
      self.assertEqual(numerical_transformer.steps[0][0], 'scaler')
      self.assertIsInstance(
          numerical_transformer.steps[0][1], preprocessing.MinMaxScaler
      )
    with self.subTest(name='CheckCustomResampler'):
      resampler = custom_pipeline.named_steps['resampler']
      self.assertIsInstance(resampler, custom_transformer.ResamplingTransformer)
      self.assertEqual(resampler.label_type, label_type)

  @mock.patch.object(utils, 'load_pipeline_from_cloud_storage', autospec=True)
  def test_get_data_transformation_pipeline_prediction(
      self, mock_load_pipeline
  ):
    mock_estimator = mock.Mock()
    mock_estimator.fit.return_value = mock_estimator
    mock_load_pipeline.return_value = pipeline.Pipeline(
        steps=[('mock_step', mock_estimator)]
    )
    mock_gcp_storage_client = mock.Mock(spec=storage.Client)
    custom_pipeline = (
        data_cleaning_feature_selection._get_data_transformation_pipeline(
            use_prediction_pipeline=True,
            gcp_storage_client=mock_gcp_storage_client,
            gcp_bucket_name='my_bucket',
            pipeline_name='my_pipeline',
        )
    )
    mock_load_pipeline.assert_called_once_with(
        gcp_storage_client=mock_gcp_storage_client,
        gcp_bucket_name='my_bucket',
        pipeline_name='my_pipeline',
    )
    self.assertIsInstance(custom_pipeline, pipeline.Pipeline)

  def test_data_transformation_fit(self):
    x = pd.DataFrame({
        'id': [1, 2, 3],
        'color': ['red', 'blue', 'green'],
        'price': [10.5, 20.0, 15.75],
    })
    y = pd.Series([0, 1, 0], name='label')
    data_index = x[['id']]
    mock_transformer = mock.Mock(spec=pipeline.Pipeline)
    mock_transformer.fit_transform.return_value = np.array(
        [[1, 0, 0, 10.5], [0, 1, 0, 20.0], [0, 0, 1, 15.75]]
    )
    mock_transformer.named_steps = {
        'preprocessor': mock.Mock(
            get_feature_names_out=mock.Mock(
                return_value=['color_red', 'color_blue', 'color_green', 'price']
            )
        )
    }
    result = data_cleaning_feature_selection._data_transformation(
        fit_data_bool=True,
        id_cols=['id'],
        x=x,
        data_index=data_index,
        custom_data_transformer=mock_transformer,
        y=y,
    )
    ml_ready_data = result.ml_ready_data
    mock_transformer.fit_transform.assert_called_once_with(x, y)
    expected_data = pd.DataFrame({
        'id': [1, 2, 3],
        'color_red': [1.0, 0.0, 0.0],
        'color_blue': [0.0, 1.0, 0.0],
        'color_green': [0.0, 0.0, 1.0],
        'price': [10.5, 20.0, 15.75],
        'label': [0, 1, 0],
        'train_test': ['train', 'train', 'train'],
    })
    pd.testing.assert_frame_equal(ml_ready_data, expected_data)

  @mock.patch.object(utils, 'save_pipeline_to_cloud_storage', autospec=True)
  @mock.patch.object(
      utils,
      'run_load_table_to_bigquery',
      autospec=True,
  )
  @mock.patch.object(
      data_cleaning_feature_selection,
      '_output_invalid_data_summary',
      autospec=True,
  )
  def test_data_preprocessing_for_ml_for_training_pipeline(
      self,
      mock_save_pipeline_to_cloud_storage,
      mock_run_load_table_to_bigquery,
      mock_output_summary,
  ):
    mock_save_pipeline_to_cloud_storage.return_value = None
    mock_run_load_table_to_bigquery.return_value = None
    mock_output_summary.return_value = None
    result = data_cleaning_feature_selection.data_preprocessing_for_ml(
        use_prediction_pipeline=False,
        bigquery_client=self.mock_bigquery_client,
        df=self.input_data_for_data_cleaning_feature_selection_with_more_rows,
        gcp_storage_client=self.mock_storage_client,
        gcp_bucket_name='test_bucket',
        dataset_id='test_dataset',
        table_name='test_table_name',
        id_cols=self.id_cols,
        numeric_labels=self.numeric_labels,
        categorical_labels=self.categorical_labels,
        train_test_split_order_by_cols=self.train_test_split_order_by_cols,
        train_test_split_asc_order=True,
        train_test_split_test_size_proportion=0.1,
    )
    result['refund_value'].reset_index(drop=True, inplace=True)

    expected_data_refund_value = {
        'transaction_id': [
            1001.0,
            1008.0,
            1002.0,
            1005.0,
            1010.0,
            1003.0,
            1009.0,
            1007.0,
            1004.0,
            1006.0,
        ],
        'transaction_date': [
            '2023-02-14',
            '2023-03-03',
            '2023-04-08',
            '2023-04-12',
            '2023-04-26',
            '2023-05-15',
            '2023-06-25',
            '2023-09-05',
            '2023-11-14',
            '2023-12-10',
        ],
        'num__transaction_value': [
            0.2111818433434819,
            0.3351231663437587,
            0.22286188762800996,
            0.0,
            0.1100470523110988,
            0.9999999999999999,
            0.6405756988652089,
            0.4045945197896485,
            0.6339883753113755,
            0.8946028231386659,
        ],
        'num__shipping_value': [
            0.8545454545454545,
            0.8385026737967913,
            0.11336898395721928,
            0.6417112299465241,
            0.332620320855615,
            1.0,
            0.0,
            0.39893048128342246,
            0.8117647058823529,
            0.6171122994652406,
        ],
        'num__item_count': [
            0.3333333333333333,
            0.0,
            0.0,
            1.0,
            0.3333333333333333,
            0.3333333333333333,
            0.6666666666666667,
            0.6666666666666667,
            0.0,
            0.0,
        ],
        'num__min_price': [
            0.0,
            0.4576639166304819,
            0.4194528875379939,
            0.7520625271385151,
            0.4164133738601824,
            1.0,
            0.5236647850629613,
            0.9960920538428137,
            0.17021276595744678,
            0.54363873208858,
        ],
        'num__max_price': [
            0.9238744248354593,
            0.16529792067097682,
            0.5417321917409283,
            0.9109441435144738,
            1.0,
            0.22738656881588915,
            0.0,
            0.2171355349758286,
            0.5273457976585707,
            0.7957365018347022,
        ],
        'num__event_count_first_visit': [
            0.3333333333333333,
            0.3333333333333333,
            0.0,
            1.0,
            0.3333333333333333,
            0.6666666666666666,
            0.3333333333333333,
            0.3333333333333333,
            0.0,
            1.0,
        ],
        'refund_value': [
            38.09,
            35.09,
            17.78,
            33.56,
            19.2,
            7.21,
            11.3,
            1.88,
            6.4,
            38.86,
        ],
        'train_test': [
            'train',
            'train',
            'train',
            'train',
            'train',
            'train',
            'train',
            'train',
            'train',
            'test',
        ],
    }
    expected_data_refund_value = pd.DataFrame(expected_data_refund_value)
    columns_to_convert_to_float = [
        'transaction_id',
        'num__transaction_value',
        'num__shipping_value',
        'num__item_count',
        'num__min_price',
        'num__max_price',
        'num__event_count_first_visit',
        'refund_value',
    ]
    expected_data_refund_value[columns_to_convert_to_float] = (
        expected_data_refund_value[columns_to_convert_to_float].astype(float)
    )
    expected_data_refund_value['transaction_date'] = pd.to_datetime(
        expected_data_refund_value['transaction_date']
    )
    expected_data_refund_value['train_test'] = expected_data_refund_value[
        'train_test'
    ].astype(str)

    self.assertSequenceEqual(self.labels, list(result.keys()))
    pd.testing.assert_frame_equal(
        result['refund_value'], expected_data_refund_value
    )

  @mock.patch.object(utils, 'save_pipeline_to_cloud_storage', autospec=True)
  @mock.patch.object(
      utils,
      'run_load_table_to_bigquery',
      autospec=True,
  )
  @mock.patch.object(
      data_cleaning_feature_selection,
      '_output_invalid_data_summary',
      autospec=True,
  )
  @mock.patch.object(
      data_cleaning_feature_selection,
      '_feature_selection_training_prediction_pipeline',
      autospec=True,
  )
  @mock.patch.object(
      data_cleaning_feature_selection,
      '_data_preprocessing_training_prediction_pipeline',
      autospec=True,
  )
  def test_data_preprocessing_for_ml_for_prediction_pipeline(
      self,
      mock_data_preprocessing_training_prediction_pipeline,
      mock_feature_selection_training_prediction_pipeline,
      mock_output_summary,
      mock_run_load_table_to_bigquery,
      mock_save_pipeline_to_cloud_storage,
  ):
    mock_save_pipeline_to_cloud_storage.return_value = None
    mock_run_load_table_to_bigquery.return_value = None
    mock_output_summary.return_value = None
    mock_feature_selection_training_prediction_pipeline.return_value = {
        'refund_value': pd.DataFrame({
            'transaction_id': [1, 2, 3],
            'transaction_date': ['2023-01-01', '2023-02-01', '2023-03-01'],
            'feature1': [10, 20, 30],
        })
    }
    mock_data_preprocessing_training_prediction_pipeline.return_value = (
        pd.DataFrame({
            'transaction_id': [1, 2, 3],
            'transaction_date': ['2023-01-01', '2023-02-01', '2023-03-01'],
            'feature1': [0.5, 1, 1.5],
        })
    )
    result = data_cleaning_feature_selection.data_preprocessing_for_ml(
        use_prediction_pipeline=True,
        bigquery_client=self.mock_bigquery_client,
        df=self.input_data_for_data_cleaning_feature_selection_with_more_rows,
        gcp_storage_client=self.mock_storage_client,
        gcp_bucket_name='test_bucket',
        dataset_id='test_dataset',
        table_name='test_table_name',
        id_cols=self.id_cols,
        numeric_labels=self.numeric_labels,
        categorical_labels=self.categorical_labels,
        train_test_split_order_by_cols=self.train_test_split_order_by_cols,
        train_test_split_asc_order=True,
        train_test_split_test_size_proportion=0.1,
    )
    expected_data_refund_value = pd.DataFrame({
        'transaction_id': [1, 2, 3],
        'transaction_date': ['2023-01-01', '2023-02-01', '2023-03-01'],
        'feature1': [0.5, 1, 1.5],
        'train_test': ['train', 'train', 'test'],
    })
    expected_data_refund_value['transaction_date'] = pd.to_datetime(
        expected_data_refund_value['transaction_date']
    )
    expected_data_refund_value['train_test'] = expected_data_refund_value[
        'train_test'
    ].astype(str)
    print(result.keys())
    print(result)
    self.assertSequenceEqual(list(result.keys()), ['refund_value'])


if __name__ == '__main__':
  absltest.main()
