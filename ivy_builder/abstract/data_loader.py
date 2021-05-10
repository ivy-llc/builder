# global
import abc

# local
from ivy_builder.specs.data_loader_spec import DataLoaderSpec


class DataLoader(abc.ABC):

    def __init__(self, data_loader_spec: DataLoaderSpec):
        """
        base class for loading data from disk for training
        """
        self._spec = data_loader_spec

    @abc.abstractmethod
    def get_next_training_batch(self):
        """
        get next sample from the training dataset, as a tuple of loaded tensors
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_next_batch(self, dataset_key: str):
        """
        get next sample from the data, as specified by the key, as a tuple of loaded tensors
        """
        raise NotImplementedError

    # Private #
    # --------#

    def set_base(self, base):
        self._base = base

    # Getters #
    # --------#

    @property
    def spec(self):
        return self._spec
