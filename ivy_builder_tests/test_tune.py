# global
import os
import ray
import ivy
import math
import pytest

# local
from ivy_tests.test_ivy import helpers
import ivy_builder.builder as builder
import ivy_builder_tests.helpers as builder_helpers
from ivy_builder_demos.simple_example import ExampleDataLoader, ExampleNetwork, ExampleTrainer

ray.init()
THIS_DIR = os.path.dirname(os.path.realpath(__file__))


def test_tune_numeric_spec(dev_str, call):
    if call is not helpers.torch_call:
        # ToDo: work out why the backend framework is fixed for tune after the first call,
        #  and include other frameworks in test once this is fixed
        pytest.skip()

    builder_helpers.remove_dirs()
    data_loader_spec_args = {'batch_size': 1}
    trainer_spec_args = {'total_iterations': 10,
                         'ld_chkpt': False,
                         'log_freq': 1,
                         'log_dir': os.path.join(THIS_DIR, 'log')}
    tuner_spec_args = {'framework': ivy.current_backend_str(),
                       'train_steps_per_tune_step': 2,
                       'trainer_spec':
                           {'initial_learning_rate':
                                {'min': 10 ** -6,
                                 'max': 10 ** -3,
                                 'exponent': 10
                                 }
                            },
                       'name': 'tune',
                       'num_samples': 5,
                       'parallel_trials': 1,
                       'grace_period': 1,
                       'checkpoint_freq': 0}
    tuner = builder.build_tuner(ExampleDataLoader, ExampleNetwork, ExampleTrainer,
                                data_loader_spec_args=data_loader_spec_args, trainer_spec_args=trainer_spec_args,
                                tuner_spec_args=tuner_spec_args)
    tuner.tune()
    builder_helpers.remove_dirs()


def test_tune_general_spec(dev_str, call):
    if call is not helpers.torch_call:
        # ToDo: work out why the backend framework is fixed for tune after the first call,
        #  and include other frameworks in test once this is fixed
        pytest.skip()

    builder_helpers.remove_dirs()
    data_loader_spec_args = {'batch_size': 1}
    trainer_spec_args = {'total_iterations': 2,
                         'ld_chkpt': False,
                         'log_freq': 1,
                         'log_dir': os.path.join(THIS_DIR, 'log')}
    tuner_spec_args = {'framework': ivy.current_backend_str(),
                       'train_steps_per_tune_step': 1,
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
                       'name': 'tune',
                       'num_samples': 1,
                       'parallel_trials': 1,
                       'grace_period': 1,
                       'checkpoint_freq': 0}
    tuner = builder.build_tuner(ExampleDataLoader, ExampleNetwork, ExampleTrainer,
                                data_loader_spec_args=data_loader_spec_args, trainer_spec_args=trainer_spec_args,
                                tuner_spec_args=tuner_spec_args)
    tuner.tune()
    builder_helpers.remove_dirs()


def test_tune_resume_training(dev_str, call):
    if call is not helpers.torch_call:
        # ToDo: work out why the backend framework is fixed for tune after the first call,
        #  and include other frameworks in test once this is fixed
        pytest.skip()
    if ivy.wrapped_mode(): # to check!
        # this test fails when running all tests for some reason, need to further investigate
        pytest.skip()

    builder_helpers.remove_dirs()

    # tuner spec args
    train_steps_per_tune_step = 2
    data_loader_spec_args = {'batch_size': 1}
    tuner_spec_args = {'framework': ivy.current_backend_str(),
                       'train_steps_per_tune_step': train_steps_per_tune_step,
                       'trainer_spec':
                           {'initial_learning_rate':
                                {'min': 10 ** -5,
                                 'max': 10 ** -4,
                                 'num_grid_samples': 2,
                                 'grid': True
                                 }
                            },
                       'name': 'tune',
                       'num_samples': 1,
                       'parallel_trials': 1,
                       'grace_period': -1,
                       'checkpoint_freq': 0}

    # first run
    total_iterations = 5
    trainer_spec_args = {'total_iterations': total_iterations,
                         'ld_chkpt': False,
                         'log_freq': 1,
                         'log_dir': os.path.join(THIS_DIR, 'log'),
                         'save_freq': 1}
    tuner = builder.build_tuner(ExampleDataLoader, ExampleNetwork, ExampleTrainer,
                                data_loader_spec_args=data_loader_spec_args, trainer_spec_args=trainer_spec_args,
                                tuner_spec_args=tuner_spec_args)
    first_results = ivy.Container(tuner.tune().results)
    first_losses = first_results.at_keys('cost').to_flat_list()

    # second run
    trainer_spec_args = {'total_iterations': total_iterations*2,
                         'ld_chkpt': True,
                         'log_freq': 1,
                         'log_dir': os.path.join(THIS_DIR, 'log'),
                         'save_freq': 1}
    tuner = builder.build_tuner(ExampleDataLoader, ExampleNetwork, ExampleTrainer,
                                data_loader_spec_args=data_loader_spec_args, trainer_spec_args=trainer_spec_args,
                                tuner_spec_args=tuner_spec_args)
    second_results = ivy.Container(tuner.tune().results)
    second_losses = second_results.at_keys('cost').to_flat_list()

    # assertion

    # first session ends training at ceil(5/2)=3 timesteps
    first_timestep = int(math.ceil(total_iterations/train_steps_per_tune_step))
    assert min([fts == first_timestep for fts in first_results.at_keys('timestep').to_flat_list()])

    # second session ends training at ceil(10/2)=5 timesteps
    second_timestep = int(math.ceil(total_iterations*2/train_steps_per_tune_step))
    assert min([sts == second_timestep for sts in second_results.at_keys('timestep').to_flat_list()])

    # both sessions trained for ceil(5/2)=3 training iterations
    training_iteration = int(math.ceil(total_iterations/train_steps_per_tune_step))
    assert min([fti == sti == training_iteration for fti, sti in
                zip(first_results.at_keys('training_iteration').to_flat_list(),
                    second_results.at_keys('training_iteration').to_flat_list())])

    # the loss is lower for the second session, after the checkpoint load from the first
    assert min([second_loss < first_loss for first_loss, second_loss in zip(first_losses, second_losses)])

    # end
    builder_helpers.remove_dirs()
