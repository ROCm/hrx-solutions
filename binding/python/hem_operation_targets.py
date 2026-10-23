"""Fresh tracked operation observations, never persisted authority or a lease.

Initial selection takes the server lock. Revalidation observes only the original
identity without ranked locks, so it can run inside an integration lock. Neither
path reconciles records or performs native lifecycle mutations.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import ipaddress
import json
import re
from urllib.parse import urlsplit

from .hem_common import DeadlineExpired, HemError, InvalidInput, UncertainOutcome
from .hem_http import primary_parameters, registration
from .hem_model_operations import OperationProfile, operation_profile
from .hem_process import ProcessManager, observe_server, require_build
from .hem_state import _name


@dataclass(frozen=True)
class OperationTarget:
    server_id: str
    origin: str
    instance_name: str
    launch_token: str
    incarnation: int
    server_revision: int
    instance_revision: int
    profile: OperationProfile
    build_identity: tuple[tuple[str, str], ...]
    registration: str
    parameter_operation: str
    primary_active: bool
    error: None
    admission: str
    # Current records contain no checked pipeline/profile identity. Do not read
    # deployment or model paths to manufacture one.
    tracked_profile_identity: None
    _server_evidence: str = field(repr=False)


@contextmanager
def _budget(registry, deadline):
    prior = registry.deadline
    registry.deadline = deadline
    try:
        yield
    finally:
        registry.deadline = prior


def _unknown(message):
    return UncertainOutcome(message, reason='unknown')


def _origin(endpoint):
    try:
        parsed = urlsplit(endpoint)
        host = ipaddress.ip_address(parsed.hostname)
        port = parsed.port
        if (parsed.scheme != 'http' or not host.is_loopback or port is None
                or not 0 < port < 65536 or parsed.username is not None
                or parsed.password is not None or parsed.path not in ('', '/')
                or parsed.query or parsed.fragment):
            raise ValueError()
        return f'http://[{host}]:{port}' if host.version == 6 else f'http://{host}:{port}'
    except (ValueError, TypeError, AttributeError) as error:
        raise _unknown('tracked native origin is unresolved') from error


def _records(registry, server_id, instance_name, deadline):
    """List APIs prove absence without interpreting generic missing-row errors."""
    deadline.remaining()
    try:
        servers = [row for row in registry.list_servers() if row['id'] == server_id]
        if not servers:
            raise HemError('saved server is missing', reason='missing')
        if len(servers) != 1:
            raise _unknown('saved server identity is ambiguous')
        server = servers[0]
        if server['pending'] or server['phase'] not in ('running', 'stopped'):
            raise _unknown('server lifecycle is unresolved')
        if server['phase'] == 'stopped' and server['launch_token'] is None:
            raise HemError('saved server is stopped', reason='unavailable')
        if server['phase'] != 'running' or not server['launch_token']:
            raise _unknown('server launch is unresolved')
        _name(server['launch_token'])
        rows = [row for row in registry.list_instances(server_id=server_id)
                if row['name'] == instance_name]
        current = [row for row in rows if row['launch_token'] == server['launch_token']]
        if not current:
            if rows:
                raise _unknown('tracked instance belongs to another launch')
            raise HemError('saved instance is missing', reason='missing')
        if len(current) != 1:
            raise _unknown('tracked instance identity is ambiguous')
        instance = current[0]
        if (instance['server_id'] != server_id or instance['pending']
                or type(instance['incarnation']) is not int or instance['incarnation'] <= 0
                or any(type(row['revision']) is not int or row['revision'] <= 0
                       for row in (server, instance))):
            raise _unknown('tracked incarnation is unresolved')
        origin = _origin(server['endpoint'])
        # Only lifecycle identity evidence is retained; no model/store facts.
        evidence = json.dumps({key: server[key] for key in
            ('id', 'launch_token', 'revision', 'endpoint', 'process', 'build', 'phase', 'pending')},
            sort_keys=True, allow_nan=False, separators=(',', ':'))
        deadline.remaining()
        return server, instance, origin, evidence
    except DeadlineExpired:
        raise
    except HemError as error:
        if error.reason in ('missing', 'unavailable', 'unknown'):
            raise
        raise _unknown('tracked operation evidence could not be read') from error
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise _unknown('tracked operation evidence is malformed or unreadable') from error


def _identity(records):
    _, instance, origin, evidence = records
    return (evidence, origin, instance['incarnation'], instance['revision'],
            instance['launch_token'], instance['name'])


def _reread(registry, server_id, instance_name, expected, deadline):
    try:
        records = _records(registry, server_id, instance_name, deadline)
    except HemError as error:
        if error.reason in ('missing', 'unavailable'):
            raise _unknown('original operation target disappeared or changed') from error
        raise
    if _identity(records) != expected:
        raise _unknown('original operation target changed; resolve explicitly before retrying')
    return records


def _build(server, manager, deadline):
    build = server.get('build')
    keys = ('native_sha256', 'payload_sha256', 'catalog_sha256')
    if not isinstance(build, dict) or any(
            not isinstance(build.get(key), str) or re.fullmatch(r'[0-9a-fA-F]{64}', build[key]) is None
            for key in keys):
        raise _unknown('recorded build digest evidence is incomplete or malformed')
    try:
        require_build(server, manager.package_root, deadline)
    except DeadlineExpired:
        raise
    except HemError as error:
        if error.reason == 'build_mismatch':
            raise HemError(str(error), reason='incompatible') from error
        raise _unknown('operation build identity could not be established') from error
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise _unknown('operation build identity could not be read') from error
    return tuple((key, build[key]) for key in keys)


def _native(document, instance_name, incarnation, profile, *, revalidating):
    try:
        row = registration(document, instance_name)
    except HemError as error:
        raise _unknown('native operation status is malformed') from error
    if row is None:
        if revalidating:
            raise _unknown('original native registration disappeared')
        raise HemError('native operation registration is unavailable', reason='unavailable')
    loom = row['loom']
    if loom['incarnation'] != incarnation:
        raise _unknown('native registration incarnation differs from tracked identity')
    facts = (loom.get('kind'), loom.get('endpoint'), row.get('input_modalities'), row.get('output_modalities'))
    if (not all(isinstance(value, str) for value in facts[:2])
            or not all(isinstance(value, list) and all(isinstance(item, str) for item in value)
                       for value in facts[2:])):
        raise _unknown('native operation profile evidence is incomplete')
    if facts != (profile.native_kind, profile.endpoint, list(profile.input_modalities), list(profile.output_modalities)):
        raise HemError('native registration is incompatible with the selected operation', reason='incompatible')
    primary = primary_parameters(loom)
    if (not isinstance(loom.get('registration'), str) or not isinstance(loom.get('parameter_operation'), str)
            or type(primary.get('active')) is not bool
            or not isinstance(loom.get('status', {}).get('admission'), str)):
        raise _unknown('native readiness evidence is incomplete')
    if (loom['registration'] != 'ready' or loom['parameter_operation'] != 'idle'
            or primary['active'] is not True or loom.get('error') is not None
            or loom['status']['admission'] != 'open'):
        raise HemError('native operation is not ready with open admission', reason='unavailable')


def _observe(server, instance, profile, deadline, manager, *, revalidating):
    identity = _build(server, manager, deadline)
    deadline.remaining()
    try:
        observation = observe_server(server, deadline, pin=manager.pin, client=manager.client)
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise _unknown('native ownership observation failed') from error
    deadline.remaining()
    if observation.process_health != 'verified' or observation.http_health != 'healthy':
        raise _unknown('native process/socket/status ownership could not be established')
    _native(observation.native, instance['name'], instance['incarnation'], profile,
            revalidating=revalidating)
    return identity


def _observe_checked(registry, records, profile, deadline, *, revalidating):
    server, instance, _, _ = records
    expected = _identity(records)
    manager = ProcessManager(registry)
    try:
        return _observe(server, instance, profile, deadline, manager, revalidating=revalidating)
    finally:
        # A failed native admission is confirmed only while its original tracked
        # identity still holds. Replacement/unreadable evidence takes precedence.
        _reread(registry, server['id'], instance['name'], expected, deadline)


def resolve_operation_target(registry, server_id, instance_name, operation, deadline):
    """Select exactly one fresh tracked target while holding its server lock."""
    _name(server_id)
    _name(instance_name)
    try:
        profile = operation_profile(operation)
    except ValueError as error:
        raise InvalidInput('unsupported operation; select chat or transcription') from error
    with _budget(registry, deadline):
        try:
            with registry.lock('server', server_id, deadline=deadline):
                records = _records(registry, server_id, instance_name, deadline)
                identity = _observe_checked(registry, records, profile, deadline, revalidating=False)
                server, instance, origin, evidence = records
                target = OperationTarget(server_id=server_id, origin=origin, instance_name=instance_name,
                    launch_token=server['launch_token'], incarnation=instance['incarnation'],
                    server_revision=server['revision'], instance_revision=instance['revision'],
                    profile=profile, build_identity=identity, registration='ready', parameter_operation='idle',
                    primary_active=True, error=None, admission='open', tracked_profile_identity=None,
                    _server_evidence=evidence)
                return target
        except DeadlineExpired:
            raise
        except HemError as error:
            if error.reason in ('missing', 'unavailable', 'incompatible', 'unknown'):
                raise
            raise _unknown('operation target locking or observation could not be established') from error
        except OSError as error:
            raise _unknown('operation target lock could not be read') from error


def revalidate_operation_target(registry, target, deadline):
    """Freshly check the same original target, without locks or replacement."""
    if not isinstance(target, OperationTarget):
        raise InvalidInput('revalidation requires an original OperationTarget')
    with _budget(registry, deadline):
        expected = (target._server_evidence, target.origin, target.incarnation, target.instance_revision,
                    target.launch_token, target.instance_name)
        records = _reread(registry, target.server_id, target.instance_name, expected, deadline)
        _observe_checked(registry, records, target.profile, deadline, revalidating=True)
