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

"""Module for saving constants values."""

import enum
from typing import TypeAlias
import immutabledict


@enum.unique
class TargetVariable(enum.Enum):
  """Distinct names for target variables in product return predictor."""

  REFUND_VALUE = 'refund_value'
  REFUND_FLAG = 'refund_flag'
  REFUND_PROPORTION = 'refund_proportion'


@enum.unique
class IDColumnNames(enum.Enum):
  """Distinct names for id columns in product return predictor."""

  TRANSACTION_ID = 'transaction_id'
  TRANSACTION_DATE = 'transaction_date'


@enum.unique
class LinearBigQueryMLModelType(enum.Enum):
  """Distinct names for linear model types in BigQueryML."""

  LINEAR_REGRESSION = 'LINEAR_REG'
  LOGISTIC_REGRESSION = 'LOGISTIC_REG'


@enum.unique
class DNNBigQueryMLModelType(enum.Enum):
  """Distinct names for DNN model types in BigQueryML."""

  DNN_CLASSIFIER = 'DNN_CLASSIFIER'
  DNN_REGRESSOR = 'DNN_REGRESSOR'


@enum.unique
class BoostedTreeBigQueryMLModelType(enum.Enum):
  """Distinct names for boosted tree model types in BigQueryML."""

  BOOSTED_TREE_CLASSIFIER = 'BOOSTED_TREE_CLASSIFIER'
  BOOSTED_TREE_REGRESSOR = 'BOOSTED_TREE_REGRESSOR'


@enum.unique
class DNNOptimizer(enum.Enum):
  """Distinct names for DNN optimizer types."""

  ADAM = 'ADAM'
  ADAGRAD = 'ADAGRAD'
  FTRL = 'FTRL'
  RMSPROP = 'RMSPROP'
  SGD = 'SGD'


@enum.unique
class DNNActivationFunction(enum.Enum):
  """Distinct names for DNN activation function types."""

  RELU = 'RELU'
  RELU6 = 'RELU6'
  CRELU = 'CRELU'
  ELU = 'ELU'
  SELU = 'SELU'
  SIGMOID = 'SIGMOID'
  TANH = 'TANH'


@enum.unique
class BoostedTreeBoosterType(enum.Enum):
  """Distinct names for booster types in BQML boosted tree models."""

  GBTREE = 'GBTREE'
  DART = 'DART'


@enum.unique
class LabelType(enum.Enum):
  """Distinct names for label types."""

  NUMERICAL = 'numerical'
  CATEGORICAL = 'categorical'


MINIMUM_DATA_POINTS_FOR_DOWNSAMPLING = 10000
MAJORITY_TO_MINORITY_CLASS_RATIO_THRESHOLD = 5.0


@enum.unique
class TrainTest(enum.Enum):
  """Distinct names labeling training and testing data."""

  TRAIN = 'train'
  TEST = 'test'


@enum.unique
class DataPipelineType(enum.Enum):
  """Data pipeline types."""

  TRAINING = 'TRAINING'
  PREDICTION = 'PREDICTION'


MINIMUM_DATA_POINTS_FOR_DOWNSAMPLING = 10000
MAJORITY_TO_MINORITY_CLASS_RATIO_THRESHOLD = 5.0
TRAIN_TEST_COL_NAME = 'train_test'
BOOSTED_TREE_BOOSTER_TYPES = frozenset([
    BoostedTreeBoosterType.GBTREE,
    BoostedTreeBoosterType.DART,
])

DNN_ACTIVATION_FUNCTIONS = frozenset([
    DNNActivationFunction.RELU,
    DNNActivationFunction.RELU6,
    DNNActivationFunction.CRELU,
    DNNActivationFunction.ELU,
    DNNActivationFunction.SELU,
    DNNActivationFunction.SIGMOID,
    DNNActivationFunction.TANH,
])

DNN_OPTIMIZERS = frozenset([
    DNNOptimizer.ADAM,
    DNNOptimizer.ADAGRAD,
    DNNOptimizer.FTRL,
    DNNOptimizer.RMSPROP,
    DNNOptimizer.SGD,
])

LABELS = frozenset([
    TargetVariable.REFUND_VALUE,
    TargetVariable.REFUND_FLAG,
    TargetVariable.REFUND_PROPORTION,
])

NUMERIC_LABELS = frozenset(
    [TargetVariable.REFUND_VALUE, TargetVariable.REFUND_PROPORTION]
)
CATEGORICAL_LABELS = frozenset([TargetVariable.REFUND_FLAG])
ID_COLS = frozenset(
    [IDColumnNames.TRANSACTION_ID, IDColumnNames.TRANSACTION_DATE]
)
BQML_LINEAR_MODEL_TYPES = frozenset([
    LinearBigQueryMLModelType.LINEAR_REGRESSION,
    LinearBigQueryMLModelType.LOGISTIC_REGRESSION,
])
BQML_DNN_MODEL_TYPES = frozenset([
    DNNBigQueryMLModelType.DNN_CLASSIFIER,
    DNNBigQueryMLModelType.DNN_REGRESSOR,
])
BQML_BOOSTED_TREE_MODEL_TYPES = frozenset([
    BoostedTreeBigQueryMLModelType.BOOSTED_TREE_CLASSIFIER,
    BoostedTreeBigQueryMLModelType.BOOSTED_TREE_REGRESSOR,
])

BQML_REGRESSION_MODEL_TYPES = frozenset([
    LinearBigQueryMLModelType.LINEAR_REGRESSION,
    DNNBigQueryMLModelType.DNN_REGRESSOR,
    BoostedTreeBigQueryMLModelType.BOOSTED_TREE_REGRESSOR,
])

BQML_CLASSIFICATION_MODEL_TYPES = frozenset([
    LinearBigQueryMLModelType.LOGISTIC_REGRESSION,
    DNNBigQueryMLModelType.DNN_CLASSIFIER,
    BoostedTreeBigQueryMLModelType.BOOSTED_TREE_CLASSIFIER,
])

REGRESSION_MODEL_PERFORMANCE_METRICS_TABLE_NAME = 'TRAINING_{preprocessed_table_name}_with_target_variable_{refund_value_col}_regression_{regression_model_type}_test_set_evaluation_metrics'

CLASSIFICATION_MODEL_PERFORMANCE_METRICS_TABLE_NAME = 'TRAINING_{preprocessed_table_name}_with_target_variable_{refund_flag_col}_binary_classifier_{binary_classifier_model_type}_test_set_evaluation_metrics'

TWO_STEP_PREDICTION_PERFORMANCE_METRICS_TABLE_NAME = 'TRAINING_{preprocessed_table_name}_with_target_variable_{refund_flag_col}_{refund_value_col}_2_step_prediction_performance_metrics'

TRAINING_PHASE_TWO_STEP_MODEL_PREDICTION_TABLE_NAME = 'TRAINING_{preprocessed_table_name}_with_target_variable_{refund_flag_col}_{refund_value_col}_2_step_predictions'

TRAINING_PHASE_REGRESSION_MODEL_PREDICTION_TABLE_NAME = 'TRAINING_{preprocessed_table_name}_with_target_variable_{refund_value_col}_regression_{regression_model_type}_predictions'

PREDICTION_PHASE_TWO_STEP_MODEL_PREDICTION_TABLE_NAME = 'PREDICTION_{preprocessed_table_name}_with_target_variable_{refund_flag_col}_{refund_value_col}_2_step_predictions'

PREDICTION_PHASE_REGRESSION_MODEL_PREDICTION_TABLE_NAME = 'PREDICTION_{preprocessed_table_name}_with_target_variable_{refund_value_col}_regression_{regression_model_type}_predictions'

TIER_LEVEL_AVG_PREDICTION_COMPARISON_TABLE_NAME = 'TRAINING_{preprocessed_table_name}_with_target_variable_{refund_flag_col}_{refund_value_col}_tier_level_avg_prediction_comparison'

MODEL_PERFORMANCE_METRICS_TABLE_NAMES = immutabledict.immutabledict({
    '2_step_model': TWO_STEP_PREDICTION_PERFORMANCE_METRICS_TABLE_NAME,
    'classification_model': CLASSIFICATION_MODEL_PERFORMANCE_METRICS_TABLE_NAME,
    'regression_model': REGRESSION_MODEL_PERFORMANCE_METRICS_TABLE_NAME,
})

MODEL_PREDICTION_TABLE_NAMES = immutabledict.immutabledict({
    'training': {
        '2_step_model': TRAINING_PHASE_TWO_STEP_MODEL_PREDICTION_TABLE_NAME,
        'regression_model': (
            TRAINING_PHASE_REGRESSION_MODEL_PREDICTION_TABLE_NAME
        ),
    },
    'prediction': {
        '2_step_model': PREDICTION_PHASE_TWO_STEP_MODEL_PREDICTION_TABLE_NAME,
        'regression_model': (
            PREDICTION_PHASE_REGRESSION_MODEL_PREDICTION_TABLE_NAME
        ),
    },
})

REGRESSION_FEATURE_IMPORTANCE_TABLE_NAME = 'TRAINING_{preprocessed_table_name}_with_target_variable_{refund_value_col}_regression_{regression_model_type}_feature_importance'
CLASSIFICATION_FEATURE_IMPORTANCE_TABLE_NAME = 'TRAINING_{preprocessed_table_name}_with_target_variable_{refund_flag_col}_binary_classifier_{binary_classifier_model_type}_feature_importance'

BQML_QUERY_TEMPLATE_DIR = (
    'product_return_predictor/sql/bigquery_ml/sql_template'
)

BQML_QUERY_TEMPLATE_FILES = immutabledict.immutabledict({
    'regression_only_training': (
        f'{BQML_QUERY_TEMPLATE_DIR}/regression_model_training_pipeline.sql'
    ),
    'regression_only_prediction': (
        f'{BQML_QUERY_TEMPLATE_DIR}/regression_model_prediction_pipeline.sql'
    ),
    'classification_regression_training': (
        f'{BQML_QUERY_TEMPLATE_DIR}/classification_regression_2_steps_model_training_pipeline.sql'
    ),
    'classification_regression_prediction': (
        f'{BQML_QUERY_TEMPLATE_DIR}/classification_regression_2_steps_model_prediction_pipeline.sql'
    ),
})


GA4_DATA_PIPELINE_QUERY_TEMPLATES = immutabledict.immutabledict({
    'train_query_step1': (
        'product_return_predictor/sql/data_pipeline/step1_create_lookup_tables.sql'
    ),
    'train_query_step2': (
        'product_return_predictor/sql/data_pipeline/step2_create_preprocessed_tables.sql'
    ),
    'train_query_step3': (
        'product_return_predictor/sql/data_pipeline/step3_create_staging_tables.sql'
    ),
    'train_query_step4': (
        'product_return_predictor/sql/data_pipeline/step4_create_website_events_pivot_data.sql'
    ),
    'train_query_step5': (
        'product_return_predictor/sql/data_pipeline/step5_aggregate_event_pivot_tables.sql'
    ),
    'train_query_step6': (
        'product_return_predictor/sql/data_pipeline/step6_combine_features.sql'
    ),
    'train_query_step7': (
        'product_return_predictor/sql/data_pipeline/step7_create_ml_ready_tables.sql'
    ),
    'train_query_step8': (
        'product_return_predictor/sql/data_pipeline/step8_drop_intermediate_tables.sql'
    ),
})

SupportedModelTypes: TypeAlias = (
    LinearBigQueryMLModelType
    | DNNBigQueryMLModelType
    | BoostedTreeBigQueryMLModelType
)
