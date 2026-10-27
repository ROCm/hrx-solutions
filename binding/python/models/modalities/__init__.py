"""Select one modality policy without importing unrelated implementations."""
from importlib import import_module

from ...hem_common import InvalidInput


NAMES = ('chat', 'asr')


def get(modality):
    if type(modality) is not str or modality not in NAMES:
        raise InvalidInput(f'Unsupported modality {modality!r}; select chat or asr')
    return import_module(f'{__name__}.{modality}')
