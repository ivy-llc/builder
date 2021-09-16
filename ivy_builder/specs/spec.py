# global
import ivy
import abc


class Spec(ivy.Container):

    def __init__(self,
                 **kwargs) -> None:
        """
        base class for storing general properties of the dataset which is saved on disk
        """
        super().__init__(**kwargs)

    @property
    @abc.abstractmethod
    def _kwargs(self):
        return self._locals
