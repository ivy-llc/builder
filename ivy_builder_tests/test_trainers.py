# global
import os
import pytest
import numpy as np
import tensorflow as tf

# local
import ivy_builder.builder as builder
import ivy_tests.helpers as helpers
import ivy_builder_tests.helpers as builder_helpers

# Simple Example #

from demos.tf import full_tf_example, simple_tf_example
from demos.tf.simple_tf_example import ExampleTFDataLoader as ExampleTFDataLoaderMin
from demos.tf.simple_tf_example import ExampleTFNetwork as ExampleTFNetworkMin
from demos.tf.simple_tf_example import ExampleTFTrainer as ExampleTFTrainerMin

ExampleDataLoadersMin = [ExampleTFDataLoaderMin]
ExampleNetworksMin = [ExampleTFNetworkMin]
ExampleTrainersMin = [ExampleTFTrainerMin]

# Full Example #

from demos.tf.full_tf_example import ExampleTFDatasetDirs, ExampleTFDatasetSpec,\
    ExampleTFDataLoaderSpec, ExampleTFDataLoader, ExampleTFNetworkSpec, ExampleTFNetwork, ExampleTFTrainerSpec,\
    ExampleTFTrainer

ExampleDatasetDirs = [ExampleTFDatasetDirs]
ExampleDatasetSpecs = [ExampleTFDatasetSpec]
ExampleDataLoaderSpecs = [ExampleTFDataLoaderSpec]
ExampleDataLoaders = [ExampleTFDataLoader]
ExampleNetworkSpecs = [ExampleTFNetworkSpec]
ExampleNetworks = [ExampleTFNetwork]
ExampleTrainerSpecs = [ExampleTFTrainerSpec]
ExampleTrainers = [ExampleTFTrainer]

# Serializable Networks #


class SerializableTFNetwork(ExampleTFNetworkMin):
    def __init__(self, network_spec):
        super().__init__(network_spec)

    # noinspection PyMethodOverriding
    def get_serializable_model(self, _):
        x = tf.keras.Input((1,))
        return tf.keras.Model(inputs=x, outputs=self._l1(x))

    # noinspection PyMethodOverriding
    def test_serializable_model(self, model_in, _):
        dummy_input = np.ones((1, 1)).astype(np.float32)
        assert model_in(dummy_input).numpy().shape


# Tests #
# ------#

def test_simple_trainers(dev_str, call):
    if call not in [helpers.tf_call, helpers.tf_graph_call]:
        # ivy builder currently onlu supports tensorflow
        pytest.skip()
    builder_helpers.remove_dirs()
    simple_tf_example.main()
    builder_helpers.remove_dirs()


def test_full_trainers(dev_str, call):
    if call not in [helpers.tf_call, helpers.tf_graph_call]:
        # ivy builder currently onlu supports tensorflow
        pytest.skip()
    builder_helpers.remove_dirs()
    full_tf_example.main()
    builder_helpers.remove_dirs()


def test_visualizing(dev_str, call):
    if call not in [helpers.tf_call, helpers.tf_graph_call]:
        # ivy builder currently onlu supports tensorflow
        pytest.skip()
    for data_loader_class, network_class, trainer_class in zip(ExampleDataLoadersMin,
                                                               ExampleNetworksMin,
                                                               ExampleTrainersMin):
        builder_helpers.remove_dirs()
        data_loader_spec_args = {'batch_size': 1}
        trainer_spec_args = {'total_iterations': 10, 'ld_chkpt': False, 'save_freq': 1}
        trainer = builder.build_trainer(data_loader_class, network_class, trainer_class,
                                        data_loader_spec_args=data_loader_spec_args,
                                        trainer_spec_args=trainer_spec_args)
        trainer.setup()
        try:
            trainer.visualize()
        except OSError:
            pass
        builder_helpers.remove_dirs()


def test_checkpoint_loading(dev_str, call):
    if call not in [helpers.tf_call, helpers.tf_graph_call]:
        # ivy builder currently onlu supports tensorflow
        pytest.skip()
    for data_loader_class, network_class, trainer_class in zip(ExampleDataLoadersMin,
                                                               ExampleNetworksMin,
                                                               ExampleTrainersMin):
        builder_helpers.remove_dirs()
        data_loader_spec_args = {'batch_size': 1}
        trainer_spec_args = {'total_iterations': 10, 'ld_chkpt': False, 'save_freq': 1}
        trainer = builder.build_trainer(data_loader_class, network_class, trainer_class,
                                        data_loader_spec_args=data_loader_spec_args,
                                        trainer_spec_args=trainer_spec_args)
        trainer.setup()
        trainer.train()
        trainer_spec_args = {'total_iterations': 20, 'ld_chkpt': True, 'save_freq': 1}
        trainer = builder.build_trainer(data_loader_class, network_class, trainer_class,
                                        data_loader_spec_args=data_loader_spec_args,
                                        trainer_spec_args=trainer_spec_args)
        trainer.setup()
        trainer.train()
        if data_loader_class is ExampleTFDataLoaderMin:
            with open('log/chkpts/checkpoint') as file:
                first_line = file.readline()
            assert first_line == 'model_checkpoint_path: "ckpt-20"\n'
        builder_helpers.remove_dirs()


def test_correct_lr_after_checkpoint_load(dev_str, call):
    if call not in [helpers.tf_call, helpers.tf_graph_call]:
        # ivy builder currently onlu supports tensorflow
        pytest.skip()

    example_dir = os.path.relpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '../demos'))

    # dataset dirs specification
    dataset_dirs_args = dict()

    # dataset specification
    dataset_spec_filepath = os.path.join(example_dir, 'json_specs', 'dataset_spec.json.example')
    dataset_spec_args = builder.parse_json_to_dict(dataset_spec_filepath)

    # data loader specification
    data_loader_spec_filepath = os.path.join(example_dir, 'json_specs', 'data_loader_spec.json.example')
    data_loader_spec_args = builder.parse_json_to_dict(data_loader_spec_filepath)

    # network specification
    network_spec_filepath = os.path.join(example_dir, 'json_specs', 'network_spec.json.example')
    network_spec_args = builder.parse_json_to_dict(network_spec_filepath)

    for dataset_dirs_class, dataset_spec_class, data_loader_spec_class, data_loader_class, network_spec_class,\
        network_class, trainer_spec_class, trainer_class \
        in zip(ExampleDatasetDirs, ExampleDatasetSpecs, ExampleDataLoaderSpecs, ExampleDataLoaders,
                     ExampleNetworkSpecs, ExampleNetworks, ExampleTrainerSpecs, ExampleTrainers):

        builder_helpers.remove_dirs()
        trainer_spec_args = {'total_iterations': 0, 'ld_chkpt': False, 'save_freq': 1,
                             'learning_decrement': 0.5, 'learning_decrement_rate': 0.2e6, 'staircase': False}

        trainer = builder.build_trainer(data_loader_class, network_class, trainer_class,
                                        dataset_dirs_args, dataset_dirs_class, dataset_spec_args,
                                        dataset_spec_class, data_loader_spec_args, data_loader_spec_class,
                                        network_spec_args, network_spec_class, trainer_spec_args,
                                        trainer_spec_class)
        trainer.setup()
        trainer.train()
        if data_loader_class is ExampleTFDataLoader:
            initial_lr = trainer.learning_rate.numpy()
            assert trainer._global_step.numpy() == 0
        else:
            initial_lr = trainer.learning_rate
            assert trainer._global_step == 0

        steps_to_take_first = 10
        trainer_spec_args = {'total_iterations': steps_to_take_first, 'ld_chkpt': False, 'save_freq': 1,
                             'learning_decrement': 0.5, 'learning_decrement_rate': 5, 'staircase': False}
        trainer = builder.build_trainer(data_loader_class, network_class, trainer_class,
                                        dataset_dirs_args, dataset_dirs_class, dataset_spec_args,
                                        dataset_spec_class, data_loader_spec_args, data_loader_spec_class,
                                        network_spec_args, network_spec_class, trainer_spec_args,
                                        trainer_spec_class)
        trainer.setup()
        trainer.train()
        if data_loader_class is ExampleTFDataLoader:
            ten_step_lr = trainer.learning_rate.numpy()
            assert trainer._global_step.numpy() == steps_to_take_first
        else:
            ten_step_lr = trainer.learning_rate
            assert trainer._global_step == steps_to_take_first
        assert initial_lr > ten_step_lr

        trainer_spec_args = {'total_iterations': steps_to_take_first, 'ld_chkpt': True,
                             'learning_decrement': 0.5, 'learning_decrement_rate': 5, 'staircase': False}
        trainer = builder.build_trainer(data_loader_class, network_class, trainer_class,
                                        dataset_dirs_args, dataset_dirs_class, dataset_spec_args,
                                        dataset_spec_class, data_loader_spec_args, data_loader_spec_class,
                                        network_spec_args, network_spec_class, trainer_spec_args,
                                        trainer_spec_class)
        trainer.setup()
        trainer.train()
        if data_loader_class is ExampleTFDataLoader:
            loaded_lr = trainer.learning_rate.numpy()
            assert trainer._global_step.numpy() == steps_to_take_first
        else:
            loaded_lr = trainer.learning_rate
            assert trainer._global_step == steps_to_take_first
        assert loaded_lr == ten_step_lr

        steps_to_take_second = 20
        trainer_spec_args = {'total_iterations': steps_to_take_second, 'ld_chkpt': True, 'save_freq': 1,
                             'learning_decrement': 0.5, 'learning_decrement_rate': 5, 'staircase': False}
        trainer = builder.build_trainer(data_loader_class, network_class, trainer_class,
                                        dataset_dirs_args, dataset_dirs_class, dataset_spec_args,
                                        dataset_spec_class, data_loader_spec_args, data_loader_spec_class,
                                        network_spec_args, network_spec_class, trainer_spec_args,
                                        trainer_spec_class)
        trainer.setup()
        trainer.train()
        if data_loader_class is ExampleTFDataLoader:
            twenty_step_lr = trainer.learning_rate.numpy()
            assert trainer._global_step.numpy() == steps_to_take_second
        else:
            twenty_step_lr = trainer.learning_rate
            assert trainer._global_step == steps_to_take_second
        assert ten_step_lr > twenty_step_lr
        builder_helpers.remove_dirs()


def test_checkpoint_save_and_restore_via_public_trainer_methods(dev_str, call):
    if call not in [helpers.tf_call, helpers.tf_graph_call]:
        # ivy builder currently onlu supports tensorflow
        pytest.skip()
    for data_loader_class, network_class, trainer_class in zip(ExampleDataLoadersMin,
                                                               ExampleNetworksMin,
                                                               ExampleTrainersMin):
        builder_helpers.remove_dirs()
        data_loader_spec_args = {'batch_size': 1}
        trainer_spec_args = {'total_iterations': 0, 'ld_chkpt': False}
        trainer = builder.build_trainer(data_loader_class, network_class, trainer_class,
                                        data_loader_spec_args=data_loader_spec_args,
                                        trainer_spec_args=trainer_spec_args)
        trainer.setup()
        chkpt0_path = os.path.join('chkpt/', 'test_chkpt0')
        trainer.save(chkpt0_path)
        if data_loader_class is ExampleTFDataLoaderMin:
            assert os.path.exists(chkpt0_path + '.index')
        trainer.train()
        chkpt1_path = os.path.join('chkpt/', 'test_chkpt1')
        trainer.save(chkpt1_path)
        if data_loader_class is ExampleTFDataLoaderMin:
            assert os.path.exists(chkpt1_path + '.index')

        data_loader_spec_args = {'batch_size': 1}
        trainer_spec_args = {'total_iterations': 10, 'ld_chkpt': False, 'save_freq': 1}
        trainer = builder.build_trainer(data_loader_class, network_class, trainer_class,
                                        data_loader_spec_args=data_loader_spec_args,
                                        trainer_spec_args=trainer_spec_args)
        trainer.setup()
        trainer.restore(chkpt0_path)
        trainer.train()
        chkpt3_path = os.path.join('chkpt/', 'test_chkpt3')
        trainer.save(chkpt3_path)
        if data_loader_class is ExampleTFDataLoaderMin:
            assert os.path.exists(chkpt3_path + '.index')
        builder_helpers.remove_dirs()


def test_saving_serialized_model(dev_str, call):
    if call not in [helpers.tf_call, helpers.tf_graph_call]:
        # ivy builder currently onlu supports tensorflow
        pytest.skip()

    SerializableNetworks = [SerializableTFNetwork]

    for data_loader_class, network_class, trainer_class in zip(ExampleDataLoadersMin,
                                                               SerializableNetworks,
                                                               ExampleTrainersMin):
        builder_helpers.remove_dirs()
        data_loader_spec_args = {'batch_size': 1}
        trainer_spec_args = {'total_iterations': 10, 'ld_chkpt': False, 'save_freq': 1}
        trainer = builder.build_trainer(data_loader_class, network_class, trainer_class,
                                        data_loader_spec_args=data_loader_spec_args,
                                        trainer_spec_args=trainer_spec_args)
        trainer.setup()
        trainer.train()
        trainer.save_model('serialized_model')
        assert os.path.exists('serialized_model')
        builder_helpers.remove_dirs()


def test_saving_and_loading_serialized_model(dev_str, call):
    if call not in [helpers.tf_call, helpers.tf_graph_call]:
        # ivy builder currently onlu supports tensorflow
        pytest.skip()

    SerializableNetworks = [SerializableTFNetwork]

    for data_loader_class, network_class, trainer_class in zip(ExampleDataLoadersMin,
                                                               SerializableNetworks,
                                                               ExampleTrainersMin):

        builder_helpers.remove_dirs()
        data_loader_spec_args = {'batch_size': 1}
        trainer_spec_args = {'total_iterations': 10, 'ld_chkpt': False, 'save_freq': 1}
        trainer = builder.build_trainer(data_loader_class, network_class, trainer_class,
                                        data_loader_spec_args=data_loader_spec_args,
                                        trainer_spec_args=trainer_spec_args)
        trainer.setup()
        trainer.train()
        trainer.save_model('serialized_model')
        if data_loader_class is ExampleTFDataLoaderMin:
            model = tf.keras.models.load_model('serialized_model')
            assert model(tf.ones((1, 1))).shape == [1, 5]
        builder_helpers.remove_dirs()
