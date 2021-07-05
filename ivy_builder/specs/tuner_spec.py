# local
from ivy.core.container import Container


class TunerSpec(Container):

    def __init__(self,
                 trainer: None,
                 train_steps_per_tune_step: int,
                 framework: str,
                 **kwargs) -> None:
        """
        parameters which define the training procedure
        """
        super().__init__(kwargs)
        self['trainer'] = trainer
        self['train_steps_per_tune_step'] = train_steps_per_tune_step
        self['framework'] = framework
