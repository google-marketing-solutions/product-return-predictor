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
import pickle
import tempfile
from unittest import mock

from google.api_core import exceptions
from google.cloud import bigquery
import pandas as pd
from sklearn import pipeline
from sklearn import preprocessing

from absl.testing import absltest
from absl.testing import parameterized
from product_return_predictor import utils


class UtilsTest(parameterized.TestCase):

  @mock.patch('google.cloud.bigquery.Client')
  def test_run_load_table_to_bigquery(self, mock_bigquery_client):
    mock_client = mock_bigquery_client.return_value
    mock_client.project = 'test_project'
    mock_job = mock.MagicMock()
    mock_client.load_table_from_dataframe.return_value = mock_job

    data = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    dataset_id = 'test_dataset'
    table_name = 'test_table'

    utils.run_load_table_to_bigquery(data, mock_client, dataset_id, table_name)

    mock_client.load_table_from_dataframe.assert_called_once_with(
        dataframe=data,
        destination='test_project.test_dataset.test_table',
        job_config=mock.ANY,
        location='europe-west4',
    )

    mock_job.result.assert_called_once()

  @mock.patch('google.cloud.bigquery.Client')
  def test_read_bq_table_to_df(self, mock_bigquery_client):
    mock_client = mock_bigquery_client.return_value
    mock_query_job = mock.MagicMock()
    mock_query_job.to_dataframe.return_value = pd.DataFrame(
        {'col1': [1, 2], 'col2': ['a', 'b']}
    )
    mock_client.query.return_value = mock_query_job

    project_id = 'test_project'
    dataset_id = 'test_dataset'
    table_name = 'test_table'

    df = utils.read_bq_table_to_df(
        project_id, mock_client, dataset_id, table_name
    )

    mock_client.query.assert_called_once_with(
        'SELECT * FROM `test_project.test_dataset.test_table`'
    )
    mock_query_job.to_dataframe.assert_called_once()
    self.assertLen(df, 2)

  @mock.patch('google.cloud.storage.Client')
  def test_save_pipeline_to_cloud_storage(self, mock_storage_client):
    mock_client = mock_storage_client.return_value
    mock_bucket = mock.MagicMock()
    mock_blob = mock.MagicMock()
    mock_client.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob

    mock_pipeline = pipeline.Pipeline(
        steps=[('scaler', preprocessing.StandardScaler())]
    )
    gcp_bucket_name = 'test_bucket'
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
      filename_without_extension = os.path.splitext(temp_file.name)[0]
      # Call the function
      utils.save_pipeline_to_cloud_storage(
          mock_pipeline,
          mock_client,
          gcp_bucket_name,
          filename_without_extension,
      )

      mock_client.bucket.assert_called_once_with(gcp_bucket_name)
      mock_bucket.blob.assert_called_once_with(temp_file.name)
      mock_blob.upload_from_filename.assert_called_once_with(temp_file.name)

      with open(temp_file.name, 'rb') as f:
        loaded_pipeline = pickle.load(f)
        self.assertSequenceEqual(
            [
                (name, type(estimator))
                for name, estimator in mock_pipeline.steps
            ],
            [
                (name, type(estimator))
                for name, estimator in loaded_pipeline.steps
            ],
        )

  def test_clean_dataframe_for_bigquery(self):
    data = {
        'column with spaces': [1, 2, 3],
        'column_with_invalid_chars!!': ['a', 'b', 'c'],
        '123_starts_with_number': [True, False, True],
        '_starts_with_underscore': [1.5, 2.3, 3.1],
        'very_long_column_name_that_exceeds_the_128_character_limit_in_BigQuery_very_long_column_name_that_exceeds_the_128_character_limit_in_BigQuery': [
            'x',
            'y',
            'z',
        ],
        'datetime_utc': pd.to_datetime(
            [
                '2023-01-01 10:00:00+00:00',
                '2023-02-15 12:30:00+00:00',
                pd.NaT,
            ],
            utc=True,
        ),
    }
    df = pd.DataFrame(data)
    cleaned_df = utils.clean_dataframe_for_bigquery(df)
    expected_data = {
        'column_with_spaces': [1, 2, 3],
        'column_with_invalid_chars__': ['a', 'b', 'c'],
        '123_starts_with_number': [True, False, True],
        'starts_with_underscore': [1.5, 2.3, 3.1],
        'very_long_column_name_that_exceeds_the_128_character_limit_in_bigquery_very_long_column_name_that_exceeds_the_128_character_limi': [
            'x',
            'y',
            'z',
        ],
        'datetime_utc': pd.to_datetime(
            ['2023-01-01 10:00:00', '2023-02-15 12:30:00', pd.NaT]
        ),
    }
    expected_df = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(cleaned_df, expected_df)

  def test_check_bigquery_table_exists_returns_true_when_table_exists(self):
    mock_client = mock.create_autospec(bigquery.Client)
    mock_client.get_table.return_value = None
    result = utils.check_bigquery_table_exists(
        mock_client, 'test_dataset', 'test_table'
    )
    self.assertTrue(result)
    mock_client.get_table.assert_called_once()

  def test_check_bigquery_table_exists_returns_false_when_table_does_not_exist(
      self,
  ):
    mock_client = mock.create_autospec(bigquery.Client)
    mock_client.get_table.side_effect = exceptions.NotFound('Table not found')
    result = utils.check_bigquery_table_exists(
        mock_client, 'test_dataset', 'non_existent_table'
    )
    self.assertFalse(result)
    mock_client.get_table.assert_called_once()

  def test_replace_special_chars_with_underscore_returns_expected_result(self):
    self.assertEqual(
        utils.replace_special_chars_with_underscore('Hello World!'),
        'hello_world',
    )
    self.assertEqual(
        utils.replace_special_chars_with_underscore(
            'This is a Test String With $pecial Ch@racters.'
        ),
        'this_is_a_test_string_with_pecial_ch_racters',
    )
    self.assertEqual(
        utils.replace_special_chars_with_underscore(
            'Another one: 123-456_789 (and spaces).'
        ),
        'another_one_123_456_789_and_spaces',
    )
    self.assertEqual(
        utils.replace_special_chars_with_underscore(
            '  Leading and trailing spaces!  '
        ),
        'leading_and_trailing_spaces',
    )
    self.assertEqual(
        utils.replace_special_chars_with_underscore(
            '__Multiple____underscores__here__'
        ),
        'multiple_underscores_here',
    )
    self.assertEqual(
        utils.replace_special_chars_with_underscore('NoSpecialCharsHere'),
        'nospecialcharshere',
    )
    self.assertEqual(utils.replace_special_chars_with_underscore(''), '')
    self.assertEqual(utils.replace_special_chars_with_underscore('   '), '')
    self.assertEqual(
        utils.replace_special_chars_with_underscore('!@#$%^&*()'), ''
    )
    self.assertEqual(
        utils.replace_special_chars_with_underscore('123 Test_Name 456'),
        '123_test_name_456',
    )
    self.assertEqual(utils.replace_special_chars_with_underscore('123'), '123')
    self.assertEqual(
        utils.replace_special_chars_with_underscore('123.45'), '123_45'
    )
    self.assertEqual(utils.replace_special_chars_with_underscore('True'), 'true')

  def test_clean_dataframe_column_names_returns_expected_result(self):
    df_messy = pd.DataFrame({
        'First Name': [1, 2],
        'Product ID#': [3, 4],
        'Order_Date (YYYY-MM-DD)': [5, 6],
        '  SALES %': [7, 8],
    })
    expected_columns_messy = [
        'first_name',
        'product_id',
        'order_date_yyyy_mm_dd',
        'sales',
    ]
    df_cleaned = utils.clean_dataframe_column_names(df_messy)
    self.assertListEqual(df_cleaned.columns.tolist(), expected_columns_messy)


if __name__ == '__main__':
  absltest.main()
