from .AVSegHead import AVSegHead
from .AVSegHeadStar import AVSegHeadStar


def build_head(type, **kwargs):
    if type == 'AVSegHead':
        return AVSegHead(**kwargs)
    elif type=='AVSegHeadStar':
        return AVSegHeadStar(**kwargs)
    else:
        raise ValueError


__all__ = ['build_head']
