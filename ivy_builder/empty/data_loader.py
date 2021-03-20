# local
from ivy_builder.specs import DataLoaderSpec
from ivy_builder.abstract.data_loader import DataLoader


class EmptyDataLoader(DataLoader):

    def __init__(self, data_loader_spec: DataLoaderSpec) -> None:
        """
        base class for loading data from disk for training
        """
        super(EmptyDataLoader, self).__init__(data_loader_spec)

    def get_next_training_batch(self) -> tuple:
        """
        get next sample from the training dataset, as a tuple of loaded tensors
        """
        pass

    def get_next_batch(self, dataset_key: str) -> tuple:
        """
        get next sample from the data, as specified by the key, as a tuple of loaded tensors
        """
        pass
