"""Current WebUI STT prerequisites and owned projections; no effects or recovery.

The WebUI owner composes this policy with its single audio/picker implementation.
Future TTS policy must use that owner rather than copying an effect sequence.
"""
from .hem_openwebui_wire import (validate_audio, strict_equal as same,
    TTS_STRINGS, STT_STRINGS, OPTIONAL_TTS, OPTIONAL_STT)
from .hem_openwebui_receipts import (ROUTING, CLEAR, GENERATED, META_DEFAULTS, BODY_KEYS,
    fail, projection, desired, create_body, comparable, created_row, validate_record)


def pristine(audio):
    row = audio['stt']; routing = projection(audio)
    default = dict(CLEAR, ENGINE='', OPENAI_API_BASE_URL='https://api.openai.com/v1')
    return ((same(routing, CLEAR) or same(routing, default) or same(routing, dict(default, OPENAI_API_BASE_URL='')))
            and row['WHISPER_MODEL'] == 'base'
            and all(row[key] == '' for key in ('DEEPGRAM_API_KEY', 'AZURE_API_KEY', 'MISTRAL_API_KEY')))


def upload_prerequisites(audio):
    row = audio['stt']; mimes = row.get('SUPPORTED_CONTENT_TYPES', [])
    extensions = row.get('ALLOWED_EXTENSIONS', [])
    if ((mimes and not all(any(candidate in mimes for candidate in choices)
            for choices in (('audio/*', 'audio/mpeg'), ('audio/*', 'audio/wav')))) or
            extensions and not {'mp3', 'wav'} <= set(extensions)):
        fail('WebUI upload policy must permit MP3 and WAV; adjust it explicitly before configure')
