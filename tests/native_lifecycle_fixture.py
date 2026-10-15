"""Reusable native HTTP gate assertions (no Hem implementation dependency).

The external installed-wheel gate supplies transport and a bounded observer.
Generic numerical inference validation remains in HRX.
"""


def incarnation(document, name):
    matches = [item for item in document['data'] if item['id'] == name]
    assert len(matches) == 1, matches
    value = matches[0]['loom']['incarnation']
    assert type(value) is int and value > 0, value
    return value


def exact(document, name, original):
    return next((item for item in document['data'] if item['id'] == name
                 and item['loom']['incarnation'] == original), None)


def activated(document, name, original):
    item = exact(document, name, original)
    assert item is not None, 'original registration disappeared'
    loom = item['loom']
    assert not loom.get('error'), loom
    assert loom['registration'] not in ('failed', 'release_failed'), loom
    return (loom['registration'] == 'ready'
            and loom['parameter_operation'] == 'idle'
            and loom['parameters']['active'] is True
            and loom['status']['admission'] == 'drained')


def eager_start(request, observe, deployment):
    """One externally bounded sequence; never undrain after a failed wait."""
    name = deployment['name']
    deployment = dict(deployment, admission='drained')
    code, _ = request('/v1/models/register', deployment)
    assert code == 202, code
    state = request('/v1/status')[1]
    original = incarnation(state, name)
    metadata = exact(state, name, original)['loom']
    assert metadata['registration'] == 'registered', metadata
    # Metadata-only registrations have no residency object yet. Native status
    # may omit parameters entirely until explicit activation prepares the model.
    assert metadata.get('parameters', {}).get('active', False) is False, metadata
    assert metadata['status']['admission'] == 'drained', metadata
    assert request('/v1/models/activate', {'model': name})[0] == 202
    ready = observe(lambda state: activated(state, name, original))
    assert request('/v1/models/undrain', {'model': name})[0] == 200
    opened = request('/v1/status')[1]
    assert exact(opened, name, original)['loom']['status']['admission'] == 'open'
    return original, ready


def retire(request, observe, name, original, mode='drain'):
    assert request('/v1/models/unregister', {'model': name, 'mode': mode})[0] == 202
    return observe(lambda state: exact(state, name, original) is None)
