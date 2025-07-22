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

"""Module for utility functions on Google Cloud Platform."""

import pathlib
import pickle
import re
from typing import Optional

from absl import logging
from google.cloud import bigquery
from google.cloud import exceptions
from google.cloud import storage
import pandas as pd
from sklearn import pipeline


def run_load_table_to_bigquery(
    data: pd.DataFrame,
    bigquery_client: bigquery.Client,
    dataset_id: str,
    table_name: str,
    location: str = 'europe-west4',
    write_disposition: str = bigquery.WriteDisposition.WRITE_TRUNCATE,
) -> None:
  """Load a Pandas Dataframe to Bigquery table.

  Args:
    data: Pandas Dataframe to be loaded to Bigquery.
    bigquery_client: Bigquery client for querying Bigquery.
    dataset_id: Bigquery dataset id.
    table_name: Bigquery table name.
    location: Location of the Bigquery table (e.g. 'europe-west4',
      'us-central1', 'us').
    write_disposition: The action that occurs if destination table already
      exists. The default value is :attr:`WRITE_TRUNCATE`: If the table already
      exists, BigQuery overwrites the table data.
  """
  table_id = f'{bigquery_client.project}.{dataset_id}.{table_name}'
  job_config = bigquery.job.LoadJobConfig(write_disposition=write_disposition)
  if write_disposition == bigquery.WriteDisposition.WRITE_TRUNCATE:
    logging.info('Creating table %r in location %r', table_id, location)
  else:
    logging.info('Appending to table %r in location %r', table_id, location)

  data.columns = data.columns.astype(str)
  bigquery_client.load_table_from_dataframe(
      dataframe=data,
      destination=table_id,
      job_config=job_config,
      location=location,
  ).result()


def read_bq_table_to_df(
    project_id: str,
    bigquery_client: bigquery.Client,
    dataset_id: str,
    table_name: str,
    limit_rows: Optional[int] = None,
) -> pd.DataFrame:
  """Read a Bigquery table to a Pandas Dataframe.

  Args:
    project_id: Project id of the Bigquery table on Google Cloud Platform.
    bigquery_client: Bigquery client for querying Bigquery.
    dataset_id: Bigquery dataset id of the Bigquery table.
    table_name: Bigquery table name of the Bigquery table.
    limit_rows: Limit for the number of rows to be read from the Bigquery table.

  Returns:
    A Pandas Dataframe containing the data from the Bigquery table.
  """
  bq_table_name = '`{project_id}.{dataset_id}.{table_name}`'.format(
      project_id=project_id, dataset_id=dataset_id, table_name=table_name
  )
  if limit_rows:
    select_table_sql_statement = (
        'SELECT * FROM {bq_table_name} limit {limit_rows}'.format(
            bq_table_name=bq_table_name, limit_rows=limit_rows
        )
    )
  else:
    select_table_sql_statement = 'SELECT * FROM {bq_table_name}'.format(
        bq_table_name=bq_table_name
    )
  df = bigquery_client.query(select_table_sql_statement).to_dataframe()
  logging.info('Executed BigQuery query:%s', select_table_sql_statement)

  return df


def save_pipeline_to_cloud_storage(
    trained_pipeline: pipeline.Pipeline,
    gcp_storage_client: storage.Client,
    gcp_bucket_name: str,
    pipeline_name: str,
) -> None:
  """Save trained data processing pipeline to Google Cloud Storage.

  Args:
    trained_pipeline: Trained data processing pipeline.
    gcp_storage_client: Google Cloud Storage client.
    gcp_bucket_name: Google Cloud Storage bucket name to save trained pipeline.
    pipeline_name: Name used for saving trained pipeline.
  """
  pickle_file_name = '{}.pkl'.format(pipeline_name)
  with open(pickle_file_name, 'wb') as f:
    pickle.dump(trained_pipeline, f)
  bucket = gcp_storage_client.bucket(gcp_bucket_name)
  blob = bucket.blob(pickle_file_name)
  blob.upload_from_filename(pickle_file_name)
  logging.info(
      'Pipeline uploaded to gs://%s/%s', gcp_bucket_name, pickle_file_name
  )


def load_pipeline_from_cloud_storage(
    gcp_storage_client: storage.Client,
    gcp_bucket_name: str,
    pipeline_name: str,
) -> pipeline.Pipeline:
  """Load a trained data processing pipeline from Google Cloud Storage.

  Args:
    gcp_storage_client: Google Cloud Storage client.
    gcp_bucket_name: Google Cloud Storage bucket name where pipeline is stored.
    pipeline_name: Name of the saved pipeline file (without the .pkl extension).

  Returns:
    The loaded trained data processing pipeline.
  """

  pickle_file_name = f'{pipeline_name}.pkl'
  bucket = gcp_storage_client.bucket(gcp_bucket_name)
  blob = bucket.blob(pickle_file_name)

  with open(pickle_file_name, 'wb') as f:
    blob.download_to_file(f)

  with open(pickle_file_name, 'rb') as f:
    trained_pipeline = pickle.load(f)
  logging.info(
      'Pipeline loaded from gs://%s/%s', gcp_bucket_name, pickle_file_name
  )
  return trained_pipeline


def clean_dataframe_for_bigquery(df: pd.DataFrame) -> pd.DataFrame:
  """Clean input pandas dataframe to make it suitable for BigQuery upload.

  Clean up dataframe columns based on the following standard: BigQuery column
  names must be alphanumeric (letters, numbers, underscores), start with a
  letter or underscore, have no more than 128 characters. Convert timezone-aware
  datetime to TIMESTAMP and remove timezone information

  Args:
      df: Input DataFrame.

  Returns:
      Cleaned DataFrame ready for BigQuery upload.
  """
  df.columns = df.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
  df.columns = df.columns.str.lstrip('_').str.lower()
  df.columns = df.columns.str[:128]

  for col in df.columns:
    if df[col].dtype == 'datetime64[ns, UTC]':
      df[col] = df[col].dt.tz_localize(None)

  return df


def read_file(file_path: str | pathlib.Path) -> str:
  """Reads a file as text."""
  if isinstance(file_path, str):
    file_path = pathlib.Path(file_path)
  return file_path.read_text()


def check_bigquery_table_exists(
    bigquery_client: bigquery.Client,
    dataset_id: str,
    table_id: str,
) -> bool:
  """Checks if a BigQuery table exists.

  Args:
      bigquery_client: The BigQuery client to use for querying.
      dataset_id: The ID of the dataset containing the table.
      table_id: The ID of the table.

  Returns:
      True if the table exists, False if the table is not found.
  """
  table_ref = bigquery_client.dataset(dataset_id).table(table_id)
  try:
    bigquery_client.get_table(table_ref)
    return True
  except exceptions.NotFound:
    logging.info('Table %r not found.', table_id)
    return False


def replace_special_chars_with_underscore(text_string: str) -> str:
  """Replaces all special characters (incl space) in a string with underscores.

  The function also converts the string to lowercase for consistency.

  Args:
      text_string: The input string.

  Returns:
      Modified string with special characters replaced by underscores,
      and converted to lowercase.
  """
  if not isinstance(text_string, str):
    text_string = str(text_string)

  text_string = text_string.lower()
  cleaned_string = re.sub(r'[\W\s]+', '_', text_string)
  cleaned_string = re.sub(r'_{2,}', '_', cleaned_string)
  cleaned_string = cleaned_string.strip('_')
  return cleaned_string


def clean_dataframe_column_names(df: pd.DataFrame):
  """Replacing special character and spacesith underscores for column names.

  The function also converts the column names to lowercase for consistency.

  Args:
      df: The input pandas dataframe.

  Returns:
      A new DataFrame with cleaned column names.
  """
  new_columns = [
      replace_special_chars_with_underscore(col) for col in df.columns
  ]
  df.columns = new_columns
  return df
