import logging
from typing import List

import tensorflow as tf
import tensorflow_transform as tft
import pprint

# Features with string data types that will be converted to indices
from tfx.components.trainer.fn_args_utils import DataAccessor, FnArgs
from tfx_bsl.tfxio import dataset_options

CATEGORICAL_FEATURE_KEYS = [
    'product_age_group', 'product_id', 'product_title', 'user_id', 'device_type', 'audience_id', 'product_gender',
    'product_brand', 'product_category(1)', 'product_category(2)', 'product_category(3)', 'product_category(4)',
    'product_category(5)', 'product_category(6)', 'product_category(7)', 'product_country', 'partner_id'
]

# Numerical features that are marked as continuous
NUMERIC_FEATURE_KEYS = ['nb_clicks_1week']

# Feature that can be grouped into buckets


# Feature that the model will predict
TIME_KEY = 'time_delay_for_conversion'
SALE = 'Sale'


# Utility function for renaming the feature
def transformed_name(key):
    key = key.replace('(', '')
    key = key.replace(')', '')
    return key


_NUMERIC_FEATURE_KEYS = NUMERIC_FEATURE_KEYS
_CATEGORICAL_FEATURE_KEYS = CATEGORICAL_FEATURE_KEYS
_SALE = SALE
_transformed_name = transformed_name


# Define the transformations
def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.
    Args:
        inputs: map from feature keys to raw not-yet-transformed features.
    Returns:
        Map from string feature key to transformed feature operations.
    """
    outputs = {}

    # Scale these features to the range [0,1]
    for key in _NUMERIC_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.scale_to_0_1(inputs[key])

    # Bucketize these features
    # for key in _SALE:
    #     print(key)

    #     pprint(_TIME_KEY)
    # Convert strings to indices in a vocabulary
    for key in _CATEGORICAL_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.compute_and_apply_vocabulary(
            (inputs[key]),
            num_oov_buckets=1,
            vocab_filename=key)
    # outputs[_transformed_name('time_delay_for_conversion')] = inputs['time_delay_for_conversion']
    outputs['Sale'] = inputs['Sale']
    # Convert the label strings to an index

    return outputs


_DENSE_FLOAT_FEATURE_KEYS = NUMERIC_FEATURE_KEYS
_VOCAB_FEATURE_KEYS = CATEGORICAL_FEATURE_KEYS
_VOCAB_SIZE = 40
_VOCAB_SIZE = 1000
_transformed_name = transformed_name

# Count of out-of-vocab buckets in which unrecognized VOCAB_FEATURES are hashed.
_OOV_SIZE = 10
_LABEL_KEY = SALE


def _transformed_names(keys):
    return [_transformed_name(key) for key in keys]


def _get_tf_examples_serving_signature(model, tf_transform_output):
    """Returns a serving signature that accepts `tensorflow.Example`."""

    # We need to track the layers in the model in order to save it.
    # TODO(b/162357359): Revise once the bug is resolved.
    model.tft_layer_inference = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def serve_tf_examples_fn(serialized_tf_example):
        """Returns the output to be used in the serving signature."""
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        # Remove label feature since these will not be present at serving time.
        raw_feature_spec.pop(_LABEL_KEY)
        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
        transformed_features = model.tft_layer_inference(raw_features)
        logging.info('serve_transformed_features = %s', transformed_features)

        outputs = model(transformed_features)
        # TODO(b/154085620): Convert the predicted labels from the model using a
        # reverse-lookup (opposite of transform.py).
        return {'outputs': outputs}

    return serve_tf_examples_fn


def _get_transform_features_signature(model, tf_transform_output):
    """Returns a serving signature that applies tf.Transform to features."""

    # We need to track the layers in the model in order to save it.
    # TODO(b/162357359): Revise once the bug is resolved.
    model.tft_layer_eval = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def transform_features_fn(serialized_tf_example):
        """Returns the transformed_features to be fed as input to evaluator."""
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
        transformed_features = model.tft_layer_eval(raw_features)
        logging.info('eval_transformed_features = %s', transformed_features)
        return transformed_features

    return transform_features_fn


def _input_fn(file_pattern: List[str],
              data_accessor: DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:
    """Generates features and label for tuning/training.
    Args:
      file_pattern: List of paths or patterns of input tfrecord files.
      data_accessor: DataAccessor for converting input to RecordBatch.
      tf_transform_output: A TFTransformOutput.
      batch_size: representing the number of consecutive elements of returned
        dataset to combine in a single batch
    Returns:
      A dataset that contains (features, indices) tuple where features is a
        dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    return data_accessor.tf_dataset_factory(
        file_pattern,
        dataset_options.TensorFlowDatasetOptions(
            batch_size=batch_size, label_key=(_LABEL_KEY)),
        tf_transform_output.transformed_metadata.schema).repeat()


def _build_keras_model(hidden_units: List[int] = None) -> tf.keras.Model:
    """Creates a DNN Keras model for classifying  data.
  Args:
    hidden_units: [int], the layer sizes of the DNN (input layer first).
  Returns:
    A keras Model.
  """
    hidden_units = [96
        , 120
        , 140,
                    100]
    real_valued_columns = [
        tf.feature_column.numeric_column(key, shape=())
        for key in _transformed_names(_DENSE_FLOAT_FEATURE_KEYS)
    ]
    categorical_columns = [
        tf.feature_column.categorical_column_with_identity(
            key, num_buckets=_VOCAB_SIZE + 1, default_value=0)
        for key in _transformed_names(_VOCAB_FEATURE_KEYS)
    ]

    indicator_column = [
        tf.feature_column.indicator_column(categorical_column)
        for categorical_column in categorical_columns
    ]

    model = _wide_and_deep_classifier(
        # TODO(b/139668410) replace with premade wide_and_deep keras model
        wide_columns=indicator_column,
        deep_columns=real_valued_columns,
        dnn_hidden_units=hidden_units or [100, 70, 50, 25], lr=0.001)
    return model


def _wide_and_deep_classifier(wide_columns, deep_columns, dnn_hidden_units, lr):
    """Build a simple keras wide and deep model.
   Args:
     wide_columns: Feature columns wrapped in indicator_column for wide (linear)
       part of the model.
     deep_columns: Feature columns for deep part of the model.
     dnn_hidden_units: [int], the layer sizes of the hidden DNN.
   Returns:
     A Wide and Deep Keras model
   """
    # Following values are hard coded for simplicity in this example,
    # However prefarably they should be passsed in as hparams.

    # Keras needs the feature definitions at compile time.
    # TODO(b/139081439): Automate generation of input layers from FeatureColumn.


    input_layers = {
        colname: tf.keras.layers.Input(name=colname, shape=(), dtype=tf.float32)
        for colname in (_DENSE_FLOAT_FEATURE_KEYS)
    }

    input_layers.update({
        colname: tf.keras.layers.Input(name=colname, shape=(), dtype='int32')
        for colname in _transformed_names(_VOCAB_FEATURE_KEYS)
    })

    # TODO(b/161952382): Replace with Keras premade models and
    # Keras preprocessing layers.
    deep = tf.keras.layers.DenseFeatures(deep_columns)(input_layers)
    for numnodes in dnn_hidden_units:
        deep = tf.keras.layers.Dense(numnodes, activation="sigmoid")(deep)
        deep = tf.keras.layers.Dropout(0.3)(deep)
    wide = tf.keras.layers.DenseFeatures(wide_columns)(input_layers)

    output = tf.keras.layers.Dense(
        1, activation='sigmoid')(
        tf.keras.layers.concatenate([deep, wide]))
    output = tf.squeeze(output, -1)

    model = tf.keras.Model(input_layers, output)
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr=lr),
        metrics=[tf.keras.metrics.AUC()])
    model.summary(print_fn=logging.info)
    return model


def run_fn(fn_args: FnArgs):
    """Train the model based on given args.
    Args:
      fn_args: Holds args used to train the model as name/value pairs.
    """
    # Number of nodes in the first layer of the DNN
    print("fn_args: ", fn_args)
    print("transform output: ", fn_args.transform_output)
    print("data accessor: ", fn_args.data_accessor)
    print("train files: ", fn_args.train_files)
    print(fn_args.eval_files)
    first_dnn_layer_size = 100
    num_dnn_layers = 4
    dnn_decay_factor = 0.7

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor,
                              tf_transform_output, 40)
    eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor,
                             tf_transform_output, 40)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = _build_keras_model(
            # Construct layers sizes with exponetial decay
            hidden_units=[
                max(2, int(first_dnn_layer_size * dnn_decay_factor ** i))
                for i in range(num_dnn_layers)
            ])

    # Write logs to path
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir, update_freq='batch')

    history = model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback])

    print(history.history)

    signatures = {
        'serving_default':
            _get_tf_examples_serving_signature(model, tf_transform_output),
        'transform_features':
            _get_transform_features_signature(model, tf_transform_output),

    }
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
