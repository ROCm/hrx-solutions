"""Trusted preparation admission and execution, independent of modality callers.

Declarations select a finite reviewed implementation; they never name code.
Admission loads only the selected contract, and execution dependencies are lazy.
"""
from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath

from ...hem_common import HemError


def _moonshine_contract():
    from .moonshine_contract import load_operation
    return load_operation()


def _moonshine_executor():
    from . import moonshine
    return moonshine


_IMPLEMENTATIONS = {'moonshine-assets-v1': (_moonshine_contract, _moonshine_executor)}


def _implementation(operation):
    if type(operation) is not str or operation not in _IMPLEMENTATIONS:
        raise HemError(f'Unsupported trusted preparation operation: {operation}')
    return _IMPLEMENTATIONS[operation]


def load_operation(operation):
    """Read the selected implementation's current independently reviewed contract."""
    return _implementation(operation)[0]()


def _relative(value):
    if (type(value) is not str or not value or '\\' in value or '\0' in value
            or PurePosixPath(value).is_absolute() or PureWindowsPath(value).drive
            or any(part in ('', '.', '..') or part.startswith('-') for part in value.split('/'))):
        raise HemError('Unsafe preparation binding path')


def validate_binding(preparation, assets):
    """Admit executable bindings without a recipe or conversion dependencies."""
    if preparation is None:
        return None
    if (type(preparation) is not dict or set(preparation) != {'operation', 'inputs', 'outputs'}
            or any(type(preparation[role]) is not dict or not preparation[role]
                   for role in ('inputs', 'outputs'))):
        raise HemError('Invalid preparation binding; expected operation, inputs and outputs')
    purposes = set()
    for role in ('inputs', 'outputs'):
        for name, purpose in preparation[role].items():
            _relative(name)
            if type(purpose) is not str or purpose not in assets or purpose in purposes:
                raise HemError('Preparation bindings require distinct known asset purposes')
            purposes.add(purpose)
    operation = load_operation(preparation['operation'])
    if (set(preparation['inputs']) != set(operation['inputs']) or
            set(preparation['outputs']) != set(operation['outputs'])):
        raise HemError('Preparation bindings differ from trusted operation inventory')
    for name, expected in operation['outputs'].items():
        purpose = preparation['outputs'][name]
        if (purpose != expected['purpose'] or
                assets[purpose] != {'format': expected['format'], 'path': name}):
            raise HemError('Preparation generated output differs from its reviewed contract')
    return operation


def validate_preparation(preparation, files, assets):
    """Associate recipe sources and all generated assets with the selected contract."""
    generated = {purpose for purpose, item in files.items() if item['source']['kind'] == 'generated'}
    operation = validate_binding(preparation, assets)
    if operation is None:
        if generated:
            raise HemError('Generated recipe assets require a trusted preparation operation')
        return None
    for name, purpose in preparation['inputs'].items():
        if purpose not in files or files[purpose]['source'] != operation['inputs'][name]:
            raise HemError('Preparation input source differs from its reviewed pinned contract')
    for name, purpose in preparation['outputs'].items():
        if purpose not in files or files[purpose]['path'] != name or purpose not in generated:
            raise HemError('Preparation generated output differs from its reviewed contract')
    if set(preparation['outputs'].values()) != generated:
        raise HemError('Preparation must bind every generated recipe purpose exactly once')
    return operation


def preflight(operation, deadline):
    """Check the selected converter under the caller's original time budget."""
    return _implementation(operation)[1]().preflight(deadline)


def prepare(operation, inputs, attempt, deadline):
    """Run the selected converter in caller-owned scratch and return generated paths."""
    return _implementation(operation)[1]().prepare(inputs, attempt, deadline)
