# local
from ivy.core.container import Container
from ivy_builder.abstract.network import Network


class TrainerSpec(Container):

    def __init__(self,
                 data_loader: None,
                 network: Network,
                 log_dir: str = 'log',
                 overwrite_log_dir: bool = False,
                 seed: int = 0,
                 ld_chkpt: bool = True,
                 save_freq: int = 1000,
                 log_freq: int = 100,
                 vis_freq: int = 500,
                 log_scalars: bool = True,
                 log_vis: bool = True,
                 log_validation: bool = True,
                 starting_iteration: int = None,
                 total_iterations: int = 1e6,
                 initial_learning_rate: float = 1e-4,
                 custom_train_step: bool = False,
                 save_trace: bool = False,
                 auto_detect_weights: bool = True,
                 **kwargs) -> None:
        """
        parameters which define the training procedure
        """
        super().__init__(kwargs)
        self['data_loader'] = data_loader
        self['network'] = network
        self['log_dir'] = log_dir
        self['overwrite_log_dir'] = overwrite_log_dir
        self['seed'] = seed
        self['ld_chkpt'] = ld_chkpt
        self['save_freq'] = save_freq
        self['log_freq'] = log_freq
        self['vis_freq'] = vis_freq
        self['log_scalars'] = log_scalars
        self['log_vis'] = log_vis
        self['log_validation'] = log_validation
        self['starting_iteration'] = starting_iteration
        self['total_iterations'] = total_iterations
        self['initial_learning_rate'] = initial_learning_rate
        self['custom_train_step'] = custom_train_step
        self['save_trace'] = save_trace
        self['auto_detect_weights'] = auto_detect_weights
