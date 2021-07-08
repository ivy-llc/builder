# local
from ivy.core.container import Container


class TunerSpec(Container):

    def __init__(self,
                 trainer: None,
                 train_steps_per_tune_step: int,
                 framework: str,
                 name: str,
                 num_samples: int,
                 parallel_trials: int,
                 grace_period: int,
                 checkpoint_freq: int,
                 **kwargs) -> None:
        """
        parameters which define the training procedure
        """
        super().__init__(kwargs)
        self['trainer'] = trainer
        self['train_steps_per_tune_step'] = train_steps_per_tune_step
        self['framework'] = framework
        self['name'] = name
        self['num_samples'] = num_samples
        self['parallel_trials'] = parallel_trials
        self['grace_period'] = grace_period
        self['checkpoint_freq'] = checkpoint_freq
