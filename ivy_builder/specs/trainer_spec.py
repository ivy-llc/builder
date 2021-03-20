# local
from ivy.core.container import Container
from ivy_builder.abstract.network import Network
from ivy_builder.abstract.data_loader import DataLoader


class TrainerSpec(Container):

    def __init__(self,
                 data_loader: DataLoader,
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
                 min_learning_rate: float = 6.25e-6,
                 max_learning_rate: float = 1e-4,
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
        self['min_learning_rate'] = min_learning_rate
        self['max_learning_rate'] = max_learning_rate
        self['custom_train_step'] = custom_train_step
        self['save_trace'] = save_trace
        self['auto_detect_weights'] = auto_detect_weights

    # Getters #
    # --------#

    @property
    def data_loader(self):
        return self['data_loader']

    @property
    def network(self):
        return self['network']

    @property
    def log_dir(self):
        return self['log_dir']

    @property
    def overwrite_log_dir(self):
        return self['overwrite_log_dir']

    @property
    def seed(self):
        return self['seed']

    @property
    def ld_chkpt(self):
        return self['ld_chkpt']

    @property
    def save_freq(self):
        return self['save_freq']

    @property
    def log_freq(self):
        return self['log_freq']

    @property
    def vis_freq(self):
        return self['vis_freq']

    @property
    def log_scalars(self):
        return self['log_scalars']

    @property
    def log_vis(self):
        return self['log_vis']

    @property
    def log_validation(self):
        return self['log_validation']

    @property
    def starting_iteration(self):
        return self['starting_iteration']

    @property
    def total_iterations(self):
        return self['total_iterations']

    @property
    def initial_learning_rate(self):
        return self['initial_learning_rate']

    @property
    def min_learning_rate(self):
        return self['min_learning_rate']

    @property
    def max_learning_rate(self):
        return self['max_learning_rate']

    @property
    def custom_train_step(self):
        return self['custom_train_step']

    @property
    def save_trace(self):
        return self['save_trace']

    @property
    def auto_detect_weights(self):
        return self['auto_detect_weights']
