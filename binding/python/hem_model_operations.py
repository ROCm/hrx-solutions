"""Finite inference protocol facts, independent of preparation and placement."""
from dataclasses import dataclass


@dataclass(frozen=True)
class OperationProfile:
    operation: str
    native_kind: str
    endpoint: str
    input_modalities: tuple[str, ...]
    output_modalities: tuple[str, ...]
    request_policy: str


CHAT = OperationProfile('chat', 'text', '/v1/chat/completions',
                        ('text',), ('text',), 'json')
TRANSCRIPTION = OperationProfile('transcription', 'recognition', '/v1/audio/transcriptions',
                                 ('audio',), ('text',), 'multipart')


def operation_profile(operation: str) -> OperationProfile:
    """Select protocol constraints only; this supplies no deployment authority."""
    if operation == CHAT.operation:
        return CHAT
    if operation == TRANSCRIPTION.operation:
        return TRANSCRIPTION
    raise ValueError('unsupported operation')
