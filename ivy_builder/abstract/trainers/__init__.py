try:
    import tensorflow as _tf
    from . import tf_trainer
    from .tf_trainer import *
except ImportError:
    pass
