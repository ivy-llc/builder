# global
import abc
import ivy

# local
from ivy_builder.specs.data_loader_spec import DataLoaderSpec


class DataLoader(abc.ABC):

    def __init__(self, data_loader_spec: DataLoaderSpec):
        """
        base class for loading data from disk for training
        """
        self._spec = data_loader_spec
        self._dev_str = ivy.default(lambda: data_loader_spec.dev_strs[0], ivy.default_device(), True)

    @abc.abstractmethod
    def get_next_batch(self, dataset_key: str = None):
        """
        get next sample from the data, as specified by the key, as an ivy.Container
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_first_batch(self, dataset_key: str = None):
        """
        get the first batch, as specified by the key, as an ivy.Container
        """
        raise NotImplementedError

    def close(self):
        """
        Close this dataset, and destroy all child objects or processes which may not be garbage collected.
        """
        pass

    # Getters #
    # --------#

    @property
    def spec(self):
        return self._spec
