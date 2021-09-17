# global
import ivy
import abc
import copy
import inspect


def locals_to_kwargs(locals_in):
    keys_to_del = list()
    for k, v in locals_in.items():
        if inspect.isclass(v) or k[0:2] == '__' or k in\
                ['self', 'dataset_dirs', 'dataset_spec', 'data_loader_spec', 'data_loader', 'network_spec', 'network',
                 'trainer_spec', 'trainer', 'tuner_spec', 'tuner']:
            keys_to_del.append(k)
    for key_to_del in keys_to_del:
        del locals_in[key_to_del]
    if 'kwargs' in locals_in:
        kwargs_dict = locals_in['kwargs']
        del locals_in['kwargs']
    else:
        kwargs_dict = {}
    return copy.deepcopy({**locals_in, **kwargs_dict})


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
