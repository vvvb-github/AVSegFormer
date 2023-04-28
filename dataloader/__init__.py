from .v2_dataset import V2Dataset, get_v2_pallete
from .s4_dataset import S4Dataset
from .ms3_dataset import MS3Dataset
from mmcv import Config


def build_dataset(type, split, **kwargs):
    if type == 'V2Dataset':
        return V2Dataset(split=split, cfg=Config(kwargs))
    elif type == 'S4Dataset':
        return S4Dataset(split=split, cfg=Config(kwargs))
    elif type == 'MS3Dataset':
        return MS3Dataset(split=split, cfg=Config(kwargs))
    else:
        raise ValueError


__all__ = ['build_dataset', 'get_v2_pallete']
