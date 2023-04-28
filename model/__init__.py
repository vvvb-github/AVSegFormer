from .AVSegFormer import AVSegFormer


def build_model(type, **kwargs):
    if type == 'AVSegFormer':
        return AVSegFormer(**kwargs)
    else:
        raise ValueError
