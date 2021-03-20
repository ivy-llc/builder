# local
from ivy_builder.abstract.trainers import TFTrainer


class EmptyTFTrainer(TFTrainer):

    def __init__(self, trainer_spec):
        super(EmptyTFTrainer, self).__init__(trainer_spec)

    def _compute_cost(self, batch):
        pass

    def _learning_rate_func(self, global_step):
        pass

    def _write_scalar_summaries(self, data_loader, network, training_batch, training_summary_writer, global_step):
        pass

    def _write_image_summaries(self, data_loader, network, training_batch, training_summary_writer, global_step):
        pass

    @property
    def _optimizer(self):
        return None
