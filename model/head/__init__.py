from .AVSegHead import AVSegHead
from .UMHead import UMHead


def build_head(type, **kwargs):
    if type == 'AVSegHead':
        return AVSegHead(**kwargs)
    elif type=='UMHead':
        return UMHead(**kwargs)
    else:
        raise ValueError


__all__ = ['build_head']
