from .AVSegHead import AVSegHead


def build_head(type, **kwargs):
    if type == 'AVSegHead':
        return AVSegHead(**kwargs)
    else:
        raise ValueError


__all__ = ['build_head']
