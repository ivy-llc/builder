# global
import abc
import os
import shutil
import pathlib
import logging
import numpy as np
import tensorflow as tf
from tensorflow.profiler.experimental import ProfilerOptions

# local
from ivy_builder.specs.trainer_spec import TrainerSpec
from ivy_builder.abstract.data_loader import DataLoader
from ivy_builder.abstract.network import Network
from ivy_builder.abstract.trainer import Trainer

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(message)s')


class TFTrainer(Trainer):

    def __init__(self, spec: TrainerSpec) -> None:
        super().__init__(spec)

        # trainer variables
        self._global_step = tf.Variable(0, dtype=tf.int64, name='global_step')
        self._learning_rate = tf.Variable(self._spec.initial_learning_rate, dtype=tf.float32, name='learning_rate')
        self._moving_average_loss = tf.Variable(0, dtype=tf.float32, name='moving_average_loss')
        self._ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        self._log_tensors = {}

        # reset graph
        tf.keras.backend.clear_session()

        # set seed
        np.random.seed(self._spec.seed)
        tf.random.set_seed(self._spec.seed)

        # uninitialized variables
        self._log_dir_train = ''
        self._vis_dir_train = ''
        self._chkpt = None
        self._chkpt_manager = None
        self._summary_writers = dict()
        self._log_dirs = list()

        # profiling
        self._save_trace = self._spec.save_trace

    # Abstract #
    # ---------#

    # Private Methods #

    @abc.abstractmethod
    @tf.function
    def _compute_cost(self, batch: tf.Tensor, tape: tf.GradientTape) -> tf.Tensor:
        """
        compute training cost from input batch
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _learning_rate_func(self, global_step: tf.Variable) -> tf.Tensor:
        """
        compute learning rate, given global step
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _write_scalar_summaries(self, data_loader: DataLoader, network: Network, training_batch: tf.Tensor,
                                training_writer: tf.summary.SummaryWriter, global_step: tf.Variable) -> None:
        """
        write scalar summaries to disk, ready for tensorboard viewing
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _write_image_summaries(self, data_loader: DataLoader, network: Network, training_batch: tf.Tensor,
                               training_writer: tf.summary.SummaryWriter, global_step: tf.Variable) -> None:
        """
        write image summaries to disk, ready for tensorboard viewing
        """
        raise NotImplementedError

    # Getters #

    @property
    @abc.abstractmethod
    def _optimizer(self):
        raise NotImplementedError

    @_optimizer.setter
    def _optimizer(self, value):
        self._optimizer = value

    # Initialization #
    # ---------------#

    def _init_checkpoint_manager(self):
        pathlib.Path(os.path.join(self._spec.log_dir, 'chkpts')).mkdir(parents=True, exist_ok=True)
        self._chkpt = tf.train.Checkpoint(step=self._global_step, optimizer=self._optimizer, net=self._spec.network)
        self._chkpt_manager = tf.train.CheckpointManager(self._chkpt, os.path.join(self._spec.log_dir, 'chkpts'), 20,
                                                         step_counter=self._global_step)

    def _make_log_dirs(self):
        for log_dir in self._log_dirs:
            pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

    def _init_log_dirs(self):

        self._log_dir_train = os.path.join(self._spec.log_dir, 'tnsrbrd', 'scalars', 'training')

        self._vis_dir_train = os.path.join(self._spec.log_dir, 'tnsrbrd', 'images', 'training')

        self._log_dirs = [
            self._log_dir_train,
            self._vis_dir_train]

        self._make_log_dirs()

    def _init_log_summary_writers(self):

        sw_log_train = tf.summary.create_file_writer(self._log_dir_train)
        self._summary_writers['log_train'] = sw_log_train

    def _init_vis_summary_writers(self):

        sw_vis_train = tf.summary.create_file_writer(self._vis_dir_train)
        self._summary_writers['vis_train'] = sw_vis_train

    def _init_summary_writers(self):

        if self._spec.log_scalars:
            self._init_log_summary_writers()
        if self._spec.log_vis:
            self._init_vis_summary_writers()

    def _init_logger(self):
        self._init_log_dirs()
        self._init_summary_writers()

    def _log_scalars(self):
        self._write_scalar_summaries(self._spec.data_loader, self._spec.network, self._training_batch,
                                     self._summary_writers['log_train'], self._global_step)

    def _save(self):
        tf.py_function(self._chkpt_manager.save, [], [tf.string])
        logging.info('network checkpoint saved @ step ' + str(self._global_step.numpy()))

    def _initialize_model(self, checkpoint_path=None):
        starting_iteration = 0
        self._init_checkpoint_manager()
        if not checkpoint_path:
            checkpoint_path = self._chkpt_manager.latest_checkpoint
        if self._spec.ld_chkpt is True and checkpoint_path is None:
            raise Exception('Unable to load checkpoint, no checkpoint files found.')
        if self._spec.ld_chkpt is True and checkpoint_path is not None:
            load_status = self._chkpt.restore(checkpoint_path)
            logging.info('loaded checkpoints from {}'.format(checkpoint_path))
            try:
                load_status.assert_consumed()
            except AssertionError:
                pass
            except Exception as e:
                raise e
            # ToDo: add this assertion back normally once this issue is resolved:
            # https://github.com/tensorflow/tensorflow/issues/33150
            # load_status.assert_consumed()
            starting_iteration = int(checkpoint_path.split('-')[-1])
            logging.info('#--------------#\n# MODEL LOADED #\n#--------------#')
        else:
            logging.info('#-------------#\n# MODEL BUILT #\n#-------------#')
        if isinstance(self._spec.starting_iteration, int):
            return self._spec.starting_iteration
        return starting_iteration

    # Training #
    # ---------#

    @tf.function
    def _train_step(self, with_output=False):
        training_batch = self._spec.data_loader.get_next_training_batch()
        with tf.GradientTape(watch_accessed_variables=self._spec.auto_detect_weights) as tape:
            cost = self._compute_cost(training_batch, tape)
            cost = tf.math.reduce_mean(cost)
        self._moving_average_loss.assign(cost)
        self._ema.apply([self._moving_average_loss])
        if not self._spec.custom_train_step:
            self._gradients = tape.gradient(cost, self._spec.network.trainable_variables)
            self._optimizer.apply_gradients(zip(self._gradients, self._spec.network.trainable_variables))
        if with_output:
            return training_batch, cost
        return cost

    def _data_load_and_train_step(self, vis_mode, log_scalars_on_this_it, log_viz_on_this_it):
        if vis_mode:
            self._training_batch = self._spec.data_loader.get_next_training_batch()
        else:
            if log_scalars_on_this_it or log_viz_on_this_it:
                self._training_batch, self._total_cost = \
                    self._train_step(with_output=True)
            else:
                self._total_cost = self._train_step()

            logging.info('step ' + str(self._global_step.numpy()) + ': cost = ' + str(self._total_cost.numpy()))
            logging.info('step ' + str(self._global_step.numpy()) + ': smoothed cost = ' +
                         str(self.moving_average_loss.numpy()))

    def _train(self, vis_mode=False, starting_iteration=None, repeat_run=False):

        if starting_iteration:
            self._starting_iteration = starting_iteration

        if repeat_run:
            self._total_iterations = self._spec.total_iterations + self._starting_iteration
        else:
            self._total_iterations = self._spec.total_iterations

        self._global_step.assign(self._starting_iteration)
        self._learning_rate.assign(tf.minimum(tf.maximum(self._learning_rate_func(self._global_step),
                                                         self._spec.min_learning_rate), self._spec.max_learning_rate))

        if self._starting_iteration == self._total_iterations:
            return self._starting_iteration

        if vis_mode:
            vis_freq = 1
        else:
            vis_freq = self._spec.vis_freq

        local_counter = 0
        tracing = False

        while self._global_step < int(self._total_iterations) or int(self._total_iterations) == -1:

            if local_counter == 11 and self._save_trace:
                tf.profiler.experimental.start(os.path.join(self._spec.log_dir, 'tnsrbrd', 'profile'),
                                               ProfilerOptions(host_tracer_level=3, python_tracer_level=1,
                                                               device_tracer_level=1))
                tracing = True

            log_scalars_on_this_it = self._spec.log_scalars and self._global_step % self._spec.log_freq == 0 \
                                     and self._spec.log_freq > 0 and not vis_mode
            log_viz_on_this_it = self._spec.log_vis and self._global_step % vis_freq == 0 and self._spec.vis_freq > 0

            if tracing:
                with tf.profiler.experimental.Trace('train', step_num=local_counter, _r=1):
                    self._data_load_and_train_step(vis_mode, log_scalars_on_this_it, log_viz_on_this_it)
            else:
                self._data_load_and_train_step(vis_mode, log_scalars_on_this_it, log_viz_on_this_it)

            if local_counter == 19 and self._save_trace:
                tf.profiler.experimental.stop()
                tracing = False

            if log_scalars_on_this_it:
                self._log_scalars()
                logging.info('tensorboard scalar log saved @ step ' + str(self._global_step.numpy()))
            if log_viz_on_this_it or vis_mode:
                self._write_image_summaries(self._spec.data_loader, self._spec.network, self._training_batch,
                                            self._summary_writers['vis_train'], self._global_step)
                logging.info('tensorboard image log saved @ step ' + str(self._global_step.numpy()))
            if self._global_step % self._spec.save_freq == 0 and self._spec.save_freq > 0 and not vis_mode:
                self._save()

            self._global_step.assign_add(1)
            local_counter += 1
            self._learning_rate.assign(tf.minimum(tf.maximum(self._learning_rate_func(self._global_step),
                                                             self._spec.min_learning_rate),
                                                  self._spec.max_learning_rate))

            if vis_mode:
                input('press enter to visualise another pwc')

        return self._global_step.numpy()

    # Public Methods #
    # ---------------#

    def save(self, checkpoint_path: str) -> None:
        """
        save the network weights in checkpoint file
        :param checkpoint_path: path of the checkpoint file for saving the weights
        """
        self._spec.network.save_weights(checkpoint_path)

    def restore(self, checkpoint_path: str, global_step: int = None) -> None:
        """
        restore the network weights from checkpoint file
        :param checkpoint_path: path of the checkpoint file for loading the weights
        :param global_step: training step to start at for continued training
        """
        self._chkpt.restore(checkpoint_path)
        if global_step is not None:
            self._global_step.assign(global_step)

    def save_model(self, saved_model_path: str, checkpoint_path: str = None) -> None:
        """
        saved the model in saved model format, from the specified checkpoint
        :param saved_model_path: path to save the new saved model
        :param checkpoint_path: path of the network weights in checkpoint files
        """
        if checkpoint_path:
            self._chkpt.restore(checkpoint_path)
        else:
            self._init_checkpoint_manager()
            self._chkpt.restore(self._chkpt_manager.latest_checkpoint)
        serialized_model = self._spec.network.get_serializable_model(self._spec)
        self._spec.network.test_serializable_model(serialized_model, self._spec)
        tf.saved_model.save(serialized_model, saved_model_path)
        try:
            loaded_model = tf.saved_model.load(saved_model_path)
            self._spec.network.test_serializable_model(loaded_model, self._spec)
        except Exception as e:
            logging.exception(e)
            logging.exception('model was saved, but could not be loaded. Now deleting.')
            shutil.rmtree(saved_model_path)

    # Getters #
    # --------#

    @property
    def moving_average_loss(self):
        return self._ema.average(self._moving_average_loss)

    @property
    def learning_rate(self):
        return self._learning_rate
