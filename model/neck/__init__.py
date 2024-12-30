from .channel_mapper import ChannelMapperWithPooling


def build_neck(type, **kwargs):
    if type == 'ChannelMapper':
        return ChannelMapperWithPooling(**kwargs)
    else:
        raise ValueError


__all__ = ['build_neck']
