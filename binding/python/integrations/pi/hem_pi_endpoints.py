"""Derived Pi connection targets; no live model cache or native operations."""
from __future__ import annotations

import ipaddress
import json
from pathlib import Path
from urllib.parse import urlsplit

from ... import hem_display
from ...hem_common import HemError, InvalidInput
from .hem_pi_contracts import ENDPOINT_PATH, ENDPOINT_SCHEMA_VERSION, NATIVE_PROFILE, provider_id
from .hem_pi_state import get_target, integration_lock, target_lock


def _encoded(document):
    # Match the installer's canonical JSON bytes, including newline and ASCII
    # escapes. Pi's existing bounded reader accepts at most one MiB.
    encoded = (json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
    if len(encoded) > 1024 * 1024:
        raise InvalidInput('Pi endpoint projection exceeds the 1 MiB reader limit; reduce configured endpoint data')
    return encoded


def _origin(value):
    try:
        parsed = urlsplit(value)
        host = ipaddress.ip_address(parsed.hostname)
        port = parsed.port
        if (parsed.scheme != 'http' or not host.is_loopback or port is None or
                not 0 < port < 65536 or parsed.username is not None or
                parsed.password is not None or parsed.path not in ('', '/') or
                parsed.query or parsed.fragment):
            raise ValueError()
    except (ValueError, TypeError, AttributeError):
        raise InvalidInput('Pi endpoint requires a numeric loopback HTTP origin and nonzero port') from None
    return f'http://[{host}]:{port}' if host.version == 6 else f'http://{host}:{port}'


def project_endpoints(root, target, servers):
    """Project immutable saved listens, retaining stopped and never-started names."""
    endpoints = []
    for server in sorted(servers, key=lambda row: row['id']):
        listen = server['saved_settings']['listen']
        origin = _origin('http://' + listen)
        if server.get('launch_token') and server.get('endpoint'):
            origin = _origin(server['endpoint'])
        endpoints.append(dict(server=server['id'], provider=provider_id(server['id']),
                              base_url=origin + '/v1', native_profile=NATIVE_PROFILE))
    if len({row['server'] for row in endpoints}) != len(endpoints):
        raise InvalidInput('duplicate saved Pi server definition')
    document = dict(schema_version=ENDPOINT_SCHEMA_VERSION,
                    integration_id=target['integration_id'], hem_root=str(Path(root).resolve()),
                    servers=endpoints)
    _encoded(document)  # Reject before a caller can persist an unreadable map.
    return document


def publish_endpoints(registry, deadline):
    """Short root→target locked compare/replace, with no native I/O or waits."""
    unchanged = dict(changed=False, new_providers=[], pi_update='unchanged')
    # Avoid even creating target directories for roots with no integration.
    if get_target(registry) is None:
        return unchanged
    from .hem_pi_files import inspect_target, read_owned, atomic_replace

    with integration_lock(registry.root, deadline):
        target = get_target(registry)
        if target is None:
            return unchanged
        # A durable reservation alone must not create a Pi target directory.
        preliminary = inspect_target(target['target'], registry.root)
        if ((preliminary.get('manifest') or {}).get('enabled') is False or
                preliminary.get('manifest') is None and not preliminary.get('pending')):
            return unchanged
        with target_lock(target['target'], deadline):
            inspected = inspect_target(target['target'], registry.root)
            manifest = inspected.get('manifest')
            if manifest is not None and manifest.get('enabled') is False:
                return unchanged
            if inspected.get('pending'):
                raise HemError('Pi update is pending; run loom-control pi configure to recover')
            if manifest is None:
                return unchanged
            if (not inspected.get('configured') or
                    manifest.get('integration_id') != target['integration_id'] or
                    manifest.get('hem_root') != str(registry.root) or
                    manifest.get('target') != target['target']):
                raise HemError('Pi integration is incomplete or changed; run loom-control pi configure')
            document = project_endpoints(registry.root, target, registry.list_servers())
            path = registry.root / ENDPOINT_PATH
            previous = read_owned(path)
            prior_providers = set()
            if previous is not None:
                try:
                    old = json.loads(previous)
                    if (not isinstance(old, dict) or set(old) != set(document) or
                            type(old['schema_version']) is not int or
                            old['schema_version'] != ENDPOINT_SCHEMA_VERSION or
                            old['integration_id'] != target['integration_id'] or
                            old['hem_root'] != str(registry.root) or
                            not isinstance(old['servers'], list)):
                        raise ValueError()
                    for row in old['servers']:
                        if (set(row) != {'server', 'provider', 'base_url', 'native_profile'} or
                                row['provider'] != provider_id(row['server']) or
                                row['native_profile'] != NATIVE_PROFILE or
                                not row['base_url'].endswith('/v1') or
                                _origin(row['base_url'][:-3]) + '/v1' != row['base_url']):
                            raise ValueError()
                        if row['provider'] in prior_providers:
                            raise ValueError()
                        prior_providers.add(row['provider'])
                except (ValueError, TypeError, KeyError, AttributeError, InvalidInput):
                    raise HemError('Pi endpoint projection ownership/schema is invalid; run loom-control pi configure') from None
            encoded = _encoded(document)
            changed = previous != encoded
            if changed:
                atomic_replace(path, encoded, expected=previous, deadline=deadline)
            new = [row['provider'] for row in document['servers'] if row['provider'] not in prior_providers]
            if new:
                hem_display.emit('Pi: new server provider; use /reload or restart Pi to discover it')
            return dict(changed=changed, new_providers=new,
                        pi_update='updated' if changed else 'unchanged')


def update_result(registry, deadline, payload):
    """A secondary Pi failure cannot turn completed native work into a retry."""
    try:
        update = publish_endpoints(registry, deadline)
        if get_target(registry) is not None or update['changed']:
            payload.update(update)
    except (HemError, OSError) as error:
        payload.update(pi_update='failed', pi_error=str(error),
                       pi_repair='loom-control pi configure')
        hem_display.emit(f'Pi endpoint update failed: {error}; native result unchanged. Repair: loom-control pi configure')
    from ...hem_clients_endpoints import update_result as update_clients
    return update_clients(registry, deadline, payload)
