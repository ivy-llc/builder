# global
import os
import ivy
import git
import abc
import shutil
import pathlib
import logging
import datetime
import numpy as np
from datetime import datetime

# local
from ivy_builder.abstract.network import Network
from ivy_builder.specs.trainer_spec import TrainerSpec
from ivy_builder.abstract.data_loader import DataLoader
from ivy_builder.builder import spec_to_dict, save_dict_as_json

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(message)s')


def _get_valid_filepath(base_dir, base_filename, file_type):
    i = 0
    while True:
        filename = base_filename + str(i) if i != 0 else base_filename
        filepath = os.path.join(base_dir, filename + file_type)
        if os.path.exists(filepath):
            i += 1
            continue
        return filepath


class Checkpoint:

    def __init__(self, optimizer, net):
        self._optimizer = optimizer
        self._net = net

    def restore(self, checkpoint_path):
        checkpoint = ivy.Container.from_disk_as_hdf5(checkpoint_path)
        self._net.v = checkpoint.network.map(lambda x, kc: ivy.variable(ivy.to_dev(x, self._net.spec.device)))
        self._optimizer.set_state(checkpoint.optimizer.map(lambda x, kc: ivy.to_dev(x, self._net.spec.device)))

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def net(self):
        return self._net


class CheckpointManager:

    def __init__(self, checkpoint, directory, max_to_keep, step_counter):
        self._checkpoint = checkpoint
        self._directory = directory
        self._max_to_keep = max_to_keep
        self._step_counter = step_counter
        self._get_latest_checkpoint_fpath()

    def _get_latest_checkpoint_fpath(self):
        if os.path.exists(self._directory):
            contents = os.listdir(self._directory)
            if len(contents) == 0:
                self._latest_checkpoint_fpath = None
            else:
                contents.sort(key=lambda x: int(x.split('-')[-1].split('.hdf5')[0]))
                self._latest_checkpoint_fpath = os.path.join(self._directory, contents[-1])
        else:
            self._latest_checkpoint_fpath = None

    @property
    def latest_checkpoint_fpath(self):
        return self._latest_checkpoint_fpath

    def save(self, step):
        checkpoint = ivy.Container({'network': self._checkpoint.net.v,
                                    'optimizer': self._checkpoint.optimizer.state})
        self._latest_checkpoint_fpath = os.path.join(self._directory, 'chkpt-{}.hdf5'.format(step))
        checkpoint.to_disk_as_hdf5(self._latest_checkpoint_fpath)


class Trainer:

    def __init__(self, spec: TrainerSpec) -> None:

        # specification
        self._spec = spec

        # uninitialized variables
        if spec.starting_iteration is not None:
            self._starting_iteration = spec.starting_iteration
        else:
            self._starting_iteration = 0
        self._total_iterations = None

        # trainer variables
        self._global_step = 0
        self._moving_average_loss = 0

        # set seed
        np.random.seed(self._spec.seed)
        ivy.seed(self._spec.seed)

        # uninitialized variables
        self._chkpt = None
        self._chkpt_manager = None

        # profiling
        self._save_trace = self._spec.save_trace

    # Abstract #
    # ---------#

    # Private Methods #

    @abc.abstractmethod
    def _compute_cost(self, batch: ivy.Array, v: ivy.Container) -> ivy.Array:
        """
        compute training cost from input batch
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _learning_rate_func(self, global_step: ivy.Variable) -> ivy.Array:
        """
        compute learning rate, given global step
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _write_scalar_summaries(self, data_loader: DataLoader, network: Network, training_batch: ivy.Array,
                                global_step: ivy.Variable) -> None:
        """
        write scalar summaries to disk, ready for tensorboard viewing
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _write_image_summaries(self, data_loader: DataLoader, network: Network, training_batch: ivy.Array,
                               global_step: ivy.Variable) -> None:
        """
        write image summaries to disk, ready for tensorboard viewing
        """
        raise NotImplementedError

    def _init(self) -> None:
        """
        Initialize the model
        """
        pass

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
        self._chkpt = Checkpoint(optimizer=self._optimizer, net=self._spec.network)
        self._chkpt_manager = CheckpointManager(self._chkpt, os.path.join(self._spec.log_dir, 'chkpts'), 20,
                                                step_counter=self._global_step)

    def _log_scalars(self):
        self._write_scalar_summaries(self._spec.data_loader, self._spec.network, self._training_batch,
                                     self._global_step)

    def _save(self):
        self._chkpt_manager.save(self._global_step)
        logging.info('network checkpoint saved @ step ' + str(self._global_step))

    def _save_spec_to_disk(self):

        # remove/create log dir
        if self._spec.overwrite_log_dir:
            shutil.rmtree(self._spec.log_dir, ignore_errors=True)
        os.makedirs(self._spec.log_dir, exist_ok=True)

        # write spec json
        spec_dir = os.path.join(self._spec.log_dir, 'spec')
        os.makedirs(spec_dir, exist_ok=True)
        complete_spec_filepath = _get_valid_filepath(spec_dir, 'complete_spec', '.json')
        spec_dict = spec_to_dict(self._spec)
        save_dict_as_json(spec_dict, complete_spec_filepath)

    def _save_info_to_disk(self):
        info_dir = os.path.join(self._spec.log_dir, 'info')
        os.makedirs(info_dir, exist_ok=True)
        info_filepath = _get_valid_filepath(info_dir, 'info', '.txt')
        try:
            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha
        except (git.exc.InvalidGitRepositoryError, ValueError):
            sha = 'NOT A GIT REPO'
        with open(info_filepath, 'w+') as info_file:
            info_file.writelines(['time of execution:\n',
                                  str(datetime.now()) + '\n\n',
                                  'git commit hash at time of execution:\n',
                                  sha + '\n'])

    def _initialize_model(self, checkpoint_path=None):
        self._init()
        self._save_spec_to_disk()
        self._save_info_to_disk()
        starting_iteration = 0
        self._init_checkpoint_manager()
        if not checkpoint_path:
            checkpoint_path = self._chkpt_manager.latest_checkpoint_fpath
        if self._spec.ld_chkpt is True and checkpoint_path is None:
            raise Exception('Unable to load checkpoint, no checkpoint files found.')
        if self._spec.ld_chkpt is True and checkpoint_path is not None:
            load_status = self._chkpt.restore(checkpoint_path)
            logging.info('loaded checkpoints from {}'.format(checkpoint_path))
            starting_iteration = int(checkpoint_path.split('-')[-1].split('.')[0])
            logging.info('#--------------#\n# MODEL LOADED #\n#--------------#')
            return starting_iteration
        else:
            logging.info('#-------------#\n# MODEL BUILT #\n#-------------#')
        if isinstance(self._spec.starting_iteration, int):
            return self._spec.starting_iteration
        return starting_iteration

    # Training #
    # ---------#

    def _train_step(self, with_output=False):
        training_batch = self._spec.data_loader.get_next_training_batch()
        cost, grads = ivy.execute_with_gradients(
            lambda v: self._compute_cost(training_batch, v=v), self._spec.network.v)
        self._moving_average_loss = (cost + self._global_step * self._moving_average_loss) / (self._global_step + 1)
        self._spec.network.v = self._optimizer.step(self._spec.network.v, grads)
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

    def _train(self, vis_mode=False, starting_iteration=None, total_iterations=None):

        if starting_iteration:
            self._starting_iteration = starting_iteration
        if total_iterations:
            self._total_iterations = total_iterations
        else:
            self._total_iterations = self._spec.total_iterations

        self._global_step = self._starting_iteration
        self._learning_rate = self._learning_rate_func(self._global_step)

        if self._starting_iteration == self._total_iterations:
            return self._starting_iteration

        if vis_mode:
            vis_freq = 1
        else:
            vis_freq = self._spec.vis_freq

        local_counter = 0
        tracing = False

        while self._global_step < int(self._total_iterations) or int(self._total_iterations) == -1:

            log_scalars_on_this_it = self._spec.log_scalars and self._global_step % self._spec.log_freq == 0 \
                                     and self._spec.log_freq > 0 and not vis_mode
            log_viz_on_this_it = self._spec.log_vis and self._global_step % vis_freq == 0 and self._spec.vis_freq > 0

            self._data_load_and_train_step(vis_mode, log_scalars_on_this_it, log_viz_on_this_it)

            if log_scalars_on_this_it:
                self._log_scalars()
            if log_viz_on_this_it or vis_mode:
                self._write_image_summaries(self._spec.data_loader, self._spec.network, self._training_batch,
                                            self._global_step)
            if self._global_step % self._spec.save_freq == 0 and self._spec.save_freq > 0 and not vis_mode:
                self._save()

            self._global_step += 1
            local_counter += 1
            self._learning_rate = self._learning_rate_func(self._global_step)

            if vis_mode:
                input('press enter to visualise another example')

        return self._global_step

    # Public Methods #
    # ---------------#

    def save(self, checkpoint_path: str) -> None:
        """
        save the network weights and optimizer state in checkpoint file
        :param checkpoint_path: path of the checkpoint file for saving the weights and optimizer state
        """
        checkpoint = ivy.Container({'network': self._spec.network.v,
                                    'optimizer': self._optimizer.state})
        os.makedirs('/'.join(checkpoint_path.split('/')[:-1]), exist_ok=True)
        checkpoint.to_disk_as_hdf5(checkpoint_path)

    def restore(self, checkpoint_path: str, global_step: int = None) -> None:
        """
        restore the network weights from checkpoint file
        :param checkpoint_path: path of the checkpoint file for loading the weights
        :param global_step: training step to start at for continued training
        """
        self._chkpt.restore(checkpoint_path)
        if global_step is not None:
            self._global_step = global_step

    def setup(self) -> None:
        """
        setup the trainer, ready for training
        """
        starting_iteration = self._initialize_model()
        if self._spec.starting_iteration is None:
            self._starting_iteration = starting_iteration

    def train(self, starting_iteration: int = None, total_iterations: int = None) -> None:
        """
        run the trainer, returning the iteration step reached
        """
        return self._train(False, starting_iteration, total_iterations)

    def visualize(self) -> None:
        """
        run the trainer, but without weight optimization. Only used for tensorboard visualization
        """
        self._train(True)

    # Getters #
    # --------#

    @property
    def learning_rate(self):
        return self._learning_rate_func(self._global_step)

    @property
    def spec(self):
        return self._spec

    @property
    def moving_average_loss(self):
        return self._moving_average_loss
