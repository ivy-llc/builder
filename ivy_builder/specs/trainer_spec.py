# global
import ivy
import abc

# local
from ivy_builder.specs.spec import Spec
from ivy_builder.abstract.network import Network
from ivy_builder.specs.spec import locals_to_kwargs


class TrainerSpec(Spec, abc.ABC):

    def __init__(self,
                 data_loader: None,
                 network: Network,
                 log_dir: str = 'log',
                 overwrite_log_dir: bool = False,
                 seed: int = 0,
                 ld_chkpt: bool = False,
                 save_freq: int = 1000,
                 log_freq: int = 100,
                 vis_freq: int = 500,
                 log_scalars: bool = True,
                 log_vis: bool = True,
                 log_validation: bool = True,
                 log_time: bool = True,
                 log_learning_rate: bool = True,
                 starting_iteration: int = None,
                 total_iterations: int = 1e6,
                 initial_learning_rate: float = 1e-4,
                 custom_train_step: bool = False,
                 auto_detect_weights: bool = True,
                 log_gradients: (tuple, str) = 'all',
                 device: str = None,
                 **kwargs) -> None:
        """
        parameters which define the training procedure
        """
        kw = locals_to_kwargs(locals())
        if log_gradients == 'all' or 'all' in log_gradients:
            log_gradients = ['mean', 'abs_mean', 'var', 'abs_var', 'min', 'abs_min', 'max', 'abs_max', 'vector_norm',
                             'global_vector_norm']
        super().__init__(data_loader=data_loader,
                         network=network,
                         log_dir=log_dir,
                         overwrite_log_dir=overwrite_log_dir,
                         seed=seed,
                         ld_chkpt=ld_chkpt,
                         save_freq=save_freq,
                         log_freq=log_freq,
                         vis_freq=vis_freq,
                         log_scalars=log_scalars,
                         log_vis=log_vis,
                         log_validation=log_validation,
                         log_time=log_time,
                         log_learning_rate=log_learning_rate,
                         starting_iteration=starting_iteration,
                         total_iterations=total_iterations,
                         initial_learning_rate=initial_learning_rate,
                         custom_train_step=custom_train_step,
                         auto_detect_weights=auto_detect_weights,
                         log_gradients=log_gradients,
                         device=ivy.default(device, 'gpu:0' if ivy.gpu_is_available() else 'cpu'),
                         **kwargs)
        self._kwargs = kw
