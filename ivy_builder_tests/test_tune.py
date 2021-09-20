# global
import ray
import ivy
import pytest

# local
import ivy_tests.helpers as helpers
import ivy_builder.builder as builder
import ivy_builder_tests.helpers as builder_helpers
from demos.simple_example import ExampleDataLoader, ExampleNetwork, ExampleTrainer

ray.init()


def test_tune_numeric_spec(dev_str, call):
    if call is not helpers.torch_call:
        # ToDo: work out why the backend framework is fixed for tune after the first call,
        #  and include other frameworks in test once this is fixed
        pytest.skip()

    builder_helpers.remove_dirs()
    trainer_spec_args = {'total_iterations': 10, 'ld_chkpt': False, 'log_freq': 1, 'log_dir': 'log'}
    tuner_spec_args = {'framework': ivy.current_framework_str(),
                       'train_steps_per_tune_step': 2,
                       'trainer_spec':
                           {'initial_learning_rate':
                                {'min': 10 ** -6,
                                 'max': 10 ** -3,
                                 'uniform': True,
                                 'exponential': True,
                                 'as_int': False
                                 }
                            },
                       'name': 'asynchyperband_test',
                       'num_samples': 5,
                       'parallel_trials': 1,
                       'grace_period': 1,
                       'checkpoint_freq': 0}
    tuner = builder.build_tuner(ExampleDataLoader, ExampleNetwork, ExampleTrainer,
                                trainer_spec_args=trainer_spec_args, tuner_spec_args=tuner_spec_args)
    tuner.tune()
    builder_helpers.remove_dirs()


def test_tune_general_spec(dev_str, call):
    if call is not helpers.torch_call:
        # ToDo: work out why the backend framework is fixed for tune after the first call,
        #  and include other frameworks in test once this is fixed
        pytest.skip()

    builder_helpers.remove_dirs()
    trainer_spec_args = {'total_iterations': 10, 'ld_chkpt': False, 'log_freq': 1, 'log_dir': 'log'}
    tuner_spec_args = {'framework': ivy.current_framework_str(),
                       'train_steps_per_tune_step': 2,
                       'network_spec':
                           {'spec_a':
                               {
                                   "configs": [{'param_0': True}, {'param_1': False}],
                                   "grid": True
                               },
                            'spec_b':
                                {
                                    "configs": [{'param_0': True}, {'param_0': False}]
                                },
                            'spec_c':
                                {
                                    "spec_c_a":
                                        {"configs": [1, 2],
                                         "grid": True},
                                    "spec_c_b":
                                        {"configs": ['100', '200']},
                                },
                            'spec_d_AND_spec_e_AND_spec_f':
                                {
                                    "configs": [(False, False, False), (False, True, False),
                                                (True, False, True), (False, True, True)],
                                    "grid": True
                                }
                            },
                       'name': 'asynchyperband_test',
                       'num_samples': 1,
                       'parallel_trials': 1,
                       'grace_period': 1,
                       'checkpoint_freq': 0}
    tuner = builder.build_tuner(ExampleDataLoader, ExampleNetwork, ExampleTrainer,
                                trainer_spec_args=trainer_spec_args, tuner_spec_args=tuner_spec_args)
    tuner.tune()
    builder_helpers.remove_dirs()
