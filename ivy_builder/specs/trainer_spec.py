# global
import ivy
import abc
from typing import Union, List

# local
from ivy_builder.specs.spec import Spec
from ivy_builder.abstract.network import Network
from ivy_builder.specs.spec import locals_to_kwargs


class TrainerSpec(Spec, abc.ABC):

    # noinspection PyShadowingBuiltins
    def __init__(self,
                 data_loader: None,
                 network: Network,
                 log_dir: str = 'log',
                 overwrite_log_dir: bool = False,
                 seed: int = 0,
                 ld_chkpt: bool = False,
                 save_freq: int = 1000,
                 save_at_end: bool = True,
                 log_freq: int = 100,
                 log_at_end: bool = True,
                 vis_freq: int = 500,
                 vis_at_end: bool = True,
                 log_validation: bool = True,
                 log_time: bool = True,
                 log_learning_rate: bool = True,
                 starting_iteration: int = None,
                 total_iterations: int = 1e6,
                 initial_learning_rate: float = 1e-4,
                 save_spec: bool = True,
                 custom_train_step: bool = False,
                 auto_detect_weights: bool = True,
                 log_gradients: (tuple, str) = 'all',
                 log_variables: (tuple, str) = 'all',
                 log_optimizer_state: (tuple, str) = 'all',
                 profile_start_step: int = 5,
                 steps_to_profile: int = 0,
                 compile_graph: bool = 'all',
                 dev_strs: Union[str, List[str]] = None,
                 dev_map_fn: str = '_raw_execute_with_grads',
                 tune_device_allocation: bool = True,
                 tune_splitting: bool = True,
                 **kwargs) -> None:
        """
        parameters which define the training procedure
        """
        kw = locals_to_kwargs(locals())
        if log_gradients == 'all' or 'all' in log_gradients:
            log_gradients = ['mean', 'abs_mean', 'var', 'abs_var', 'min', 'abs_min', 'max', 'abs_max', 'vector_norm',
                             'global_vector_norm']
        if log_variables == 'all' or 'all' in log_variables:
            log_variables = ['mean', 'abs_mean', 'var', 'abs_var', 'min', 'abs_min', 'max', 'abs_max', 'vector_norm',
                             'global_vector_norm']
        if log_optimizer_state == 'all' or 'all' in log_optimizer_state:
            log_optimizer_state = ['mean', 'abs_mean', 'var', 'abs_var', 'min', 'abs_min', 'max', 'abs_max',
                                   'vector_norm', 'global_vector_norm']
        super().__init__(data_loader=data_loader,
                         network=network,
                         log_dir=log_dir,
                         overwrite_log_dir=overwrite_log_dir,
                         seed=seed,
                         ld_chkpt=ld_chkpt,
                         save_freq=save_freq,
                         save_at_end=save_at_end,
                         log_freq=log_freq,
                         log_at_end=log_at_end,
                         vis_freq=vis_freq,
                         vis_at_end=vis_at_end,
                         log_validation=log_validation,
                         log_time=log_time,
                         log_learning_rate=log_learning_rate,
                         starting_iteration=starting_iteration,
                         total_iterations=total_iterations,
                         initial_learning_rate=initial_learning_rate,
                         save_spec=save_spec,
                         custom_train_step=custom_train_step,
                         auto_detect_weights=auto_detect_weights,
                         log_gradients=log_gradients,
                         log_variables=log_variables,
                         log_optimizer_state=log_optimizer_state,
                         profile_start_step=profile_start_step,
                         steps_to_profile=steps_to_profile,
                         compile_graph=compile_graph,
                         dev_strs=ivy.default(dev_strs, ['gpu:0'] if ivy.gpu_is_available() else ['cpu']),
                         dev_map_fn=dev_map_fn,
                         tune_device_allocation=tune_device_allocation,
                         tune_splitting=tune_splitting,
                         **kwargs)
        self._kwargs = kw
