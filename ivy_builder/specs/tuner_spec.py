# global
import abc

# local
from ivy_builder.specs.spec import Spec
from ivy_builder.specs.spec import locals_to_kwargs


class TunerSpec(Spec, abc.ABC):

    def __init__(self,
                 trainer: None,
                 train_steps_per_tune_step: int,
                 framework: str,
                 name: str,
                 num_samples: int,
                 parallel_trials: int,
                 grace_period: int,
                 checkpoint_freq: int,
                 device_priority: str = 'gpu',
                 **kwargs) -> None:
        """
        parameters which define the training procedure
        """
        kw = locals_to_kwargs(locals())
        assert device_priority in ['cpu', 'gpu']
        super().__init__(trainer=trainer,
                         train_steps_per_tune_step=train_steps_per_tune_step,
                         framework=framework,
                         name=name,
                         num_samples=num_samples,
                         parallel_trials=parallel_trials,
                         grace_period=grace_period,
                         checkpoint_freq=checkpoint_freq,
                         device_priority=device_priority,
                         **kwargs)
        self._kwargs = kw
