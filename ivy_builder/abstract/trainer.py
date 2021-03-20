# global
import os
import abc
import git
import shutil
import logging
from datetime import datetime

# local
from ivy_builder.specs.trainer_spec import TrainerSpec
from ivy_builder.abstract.data_loader import DataLoader
from ivy_builder.abstract.network import Network
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


class Trainer(abc.ABC):

    def __init__(self, spec: TrainerSpec) -> None:

        # specification
        self._spec = spec

        # remove/create log dir
        if self._spec.overwrite_log_dir:
            shutil.rmtree(self._spec.log_dir, ignore_errors=True)
        os.makedirs(self._spec.log_dir, exist_ok=True)

        # write spec json
        spec_dir = os.path.join(self._spec.log_dir, 'spec')
        os.makedirs(spec_dir, exist_ok=True)
        complete_spec_filepath = _get_valid_filepath(spec_dir, 'complete_spec', '.json')
        spec_dict = spec_to_dict(spec)
        save_dict_as_json(spec_dict, complete_spec_filepath)

        # write general info
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

        # uninitialized variables
        self._starting_iteration = None

    # Abstract #
    # ---------#

    # Private Methods #

    @abc.abstractmethod
    def _compute_cost(self, batch):
        """
        compute training cost from input batch
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _learning_rate_func(self, global_step):
        """
        compute learning rate, given global step
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _write_scalar_summaries(self, data_loader: DataLoader, network: Network, training_batch,
                                training_writer, global_step) -> None:
        """
        write scalar summaries to disk, ready for tensorboard viewing
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _write_image_summaries(self, data_loader: DataLoader, network: Network, training_batch,
                               training_writer, global_step) -> None:
        """
        write image summaries to disk, ready for tensorboard viewing
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _train(self, vis_mode=False, starting_iteration=None, repeat_run=False):
        """
        Run the training
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _initialize_model(self):
        """
        Initialize model, possibly loading from checkpoint.
        :return starting iteration
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _init_logger(self):
        """
        Initialize logger
        """
        raise NotImplementedError

    # Public Methods #

    @abc.abstractmethod
    def save_model(self, saved_model_path: str, checkpoint_path: str = None) -> None:
        """
        saved the model in saved model format, from the specified checkpoint
        :param saved_model_path: path to save the new saved model
        :param checkpoint_path: path of the network weights in checkpoint files
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, checkpoint_path: str) -> None:
        """
        save the network weights in checkpoint file
        :param checkpoint_path: path of the checkpoint file for saving the weights
        """
        raise NotImplementedError

    @abc.abstractmethod
    def restore(self, checkpoint_path: str) -> None:
        """
        restore the network weights from checkpoint file
        :param checkpoint_path: path of the checkpoint file for loading the weights
        """
        raise NotImplementedError

    # Getters #

    @property
    @abc.abstractmethod
    def _optimizer(self):
        """
        get scheduler
        """
        raise NotImplementedError

    # Setters #
    # --------#

    @_optimizer.setter
    def _optimizer(self, value):
        """
        set scheduler
        """
        self._optimizer = value

    # Public Methods #
    # ---------------#

    def setup(self) -> None:
        """
        setup the trainer, ready for training
        """
        self._starting_iteration = self._initialize_model()
        self._init_logger()

    def train(self, starting_iteration: int = 0, repeat_run: bool = False) -> None:
        """
        run the trainer, returning the iteration step reached
        """
        return self._train(False, starting_iteration, repeat_run)

    def visualize(self) -> None:
        """
        run the trainer, but without weight optimization. Only used for tensorboard visualization
        """
        self._train(True)
