"""Select one modality policy without importing unrelated implementations."""
from importlib import import_module

from ...hem_common import InvalidInput


def get(modality):
    if type(modality) is not str or modality not in ('chat', 'asr'):
        raise InvalidInput(f'Unsupported modality {modality!r}; select chat or asr')
    return import_module(f'{__name__}.{modality}')
