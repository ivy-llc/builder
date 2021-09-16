# global
import ivy
import abc
import copy
import inspect


def locals_to_kwargs(locals_in):
    locals_cp = copy.deepcopy(locals_in)
    for k, v in locals_in.items():
        if inspect.isclass(v) or k[0:2] == '__' or k == 'self':
            del locals_cp[k]
    return {**locals_cp, **(locals_cp['kwargs'] if 'kwargs' in locals_cp else {})}


class Spec(ivy.Container):

    def __init__(self,
                 **kwargs) -> None:
        """
        base class for storing general properties of the dataset which is saved on disk
        """
        super().__init__(**kwargs)

    @property
    @abc.abstractmethod
    def kwargs(self):
        return self._kwargs
