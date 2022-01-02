import datetime
import os
from typing import List

import absl
import tensorflow_model_analysis as tfma
from tfx.components import CsvExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import Pusher
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.components.trainer.executor import Executor
from tfx.dsl.components.base import executor_spec
from tfx.dsl.components.common import resolver
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.airflow.airflow_dag_runner import AirflowDagRunner
from tfx.orchestration.airflow.airflow_dag_runner import AirflowPipelineConfig
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.proto import pusher_pb2, example_gen_pb2
from tfx.proto import trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing

#######
print("import completed...")
_pipeline_name = 'criteo_pipeline_airflow'
_criteo_root = './pipeline/'

# Directory of the raw data files
_data_root = './data/criteo/data3'
_module_file = os.path.join(_criteo_root, 'taxi_utils.py')
_serving_model_dir = os.path.join(_criteo_root, 'serving_model', _pipeline_name)
_tfx_root = os.path.join(".", 'tfx')
_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)
_metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name,
                              'metadata.db')
# Pipeline arguments for Beam powered Components.
_beam_pipeline_args = [
    '--direct_running_mode=multi_processing',
    # 0 means auto-detect based on on the number of CPUs available
    # during execution time.
    '--direct_num_workers=0',
]

# Airflow-specific configs; these will be passed directly to airflow
_airflow_config = {
    'schedule_interval': None,
    'start_date': datetime.datetime(2019, 1, 1),
}


def _create_pipeline(pipeline_name: str, pipeline_root: str, data_root: str,
                     module_file: str, serving_model_dir: str,
                     metadata_path: str,
                     beam_pipeline_args: List[str]) -> pipeline.Pipeline:
    data_root_runtime = data_types.RuntimeParameter(
        'data_root', ptype=str, default=data_root
    )
    output = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=8),
            example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=2),
            example_gen_pb2.SplitConfig.Split(name='test', hash_buckets=2),

        ]))

    # Ingest the data through ExampleGen
    example_gen = CsvExampleGen(input_base=_data_root, output_config=output)

    statistics_gen = StatisticsGen(
        examples=example_gen.outputs['examples'])

    # Run SchemaGen
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs['statistics'], infer_feature_shape=True)

    # Run ExampleValidator
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'])

    # Run transformer
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=module_file)

    # Run trainer
    trainer = Trainer(
        module_file=module_file,
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(num_steps=1000),
        eval_args=trainer_pb2.EvalArgs(num_steps=150))

    # Get the latest blessed model for model validation.
    model_resolver = resolver.Resolver(
        strategy_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
        model=Channel(type=Model),
        model_blessing=Channel(
            type=ModelBlessing)).with_id('latest_blessed_model_resolver')

    # Uses TFMA to compute a evaluation statistics over features of a model and
    # perform quality validation of a candidate model (compared to a baseline).
    eval_config = tfma.EvalConfig(
        model_specs=[
            # This assumes a serving model with signature 'serving_default'. If
            tfma.ModelSpec(label_key='Sale')
        ],
        metrics_specs=[
            tfma.MetricsSpec(

                metrics=[
                    tfma.MetricConfig(class_name='ExampleCount'),
                    tfma.MetricConfig(class_name='AUC'),
                    tfma.MetricConfig(class_name='F1Score'),

                    tfma.MetricConfig(
                        class_name='AUC',
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={'value': 0.65}),
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                absolute={'value': -1e-10})))
                ]
            )
        ],
        slicing_specs=[
            # An empty slice spec means the overall slice, i.e. the whole dataset.
            tfma.SlicingSpec(),
            # Data can be sliced along a feature column. In this case, data is
            # sliced along feature column trip_start_hour.
        ])
    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config)

    # Checks whether the model passed the validation steps and pushes the model
    # to a file destination if check passed.
    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir)))

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=[
            example_gen,
            statistics_gen,
            schema_gen,
            example_validator,
            transform,
            trainer,
            model_resolver,
            evaluator,
            pusher,
        ],
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path),
        beam_pipeline_args=beam_pipeline_args)


if __name__ == '__main__':
    absl.logging.set_verbosity(absl.logging.INFO)
    AirflowDagRunner(AirflowPipelineConfig(_airflow_config)).run(
        _create_pipeline(
            pipeline_name=_pipeline_name,
            pipeline_root=_pipeline_root,
            data_root=_data_root,
            module_file=_module_file,
            serving_model_dir=_serving_model_dir,
            metadata_path=_metadata_path,
            beam_pipeline_args=_beam_pipeline_args))
