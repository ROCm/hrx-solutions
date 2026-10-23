"""Conservative OMP activation grammar, shared with activation.ts fixtures.

Offsets retain original bytes; this is deliberately not a YAML serializer.
"""
from __future__ import annotations

import base64
import hashlib
import re
from pathlib import Path
from ...hem_common import InvalidInput
from ..pi.hem_pi_files import read_owned


def scan(text):
    lines = re.findall(r'[^\n]*\n|[^\n]+$', text)
    offsets = [0]
    for line in lines: offsets.append(offsets[-1] + len(line))
    found, roots = {}, set()
    inside, provider, start, valid, auth, marker = False, None, 0, False, False, None

    def finish(end):
        nonlocal provider, valid, auth
        if provider is None: return
        if provider in found: raise InvalidInput("duplicate OMP YAML provider")
        found[provider] = dict(text=''.join(lines[start:end]), valid=valid and auth,
                               start=offsets[start], end=offsets[end], owner=None)
        provider, valid, auth = None, False, False

    for i, raw in enumerate(lines):
        line = re.sub(r'\r?\n$', '', raw)
        begin = re.fullmatch(r'  # loom-control omp begin ([0-9a-f-]{36}) ([A-Za-z0-9_.-]+)', line)
        end = re.fullmatch(r'  # loom-control omp end ([0-9a-f-]{36}) ([A-Za-z0-9_.-]+)', line)
        if begin:
            if not inside or marker: raise InvalidInput('invalid OMP activation marker')
            finish(i); marker = (begin[1], begin[2], i); continue
        if end:
            if not marker or not provider or end[1] != marker[0] or end[2] != provider or marker[1] != provider:
                raise InvalidInput('mismatched OMP activation marker')
            key, owner, first = provider, marker[0], marker[2]
            finish(i + 1)
            found[key].update(text=''.join(lines[first:i + 1]), start=offsets[first], owner=owner)
            marker = None; continue
        if 'loom-control omp begin' in line or 'loom-control omp end' in line:
            raise InvalidInput('damaged OMP activation marker')
        if not line.strip() or re.match(r'^\s*#', line): continue
        if re.search(r'\t|[&*!|>]|(?:^|\s)<<\s*:', line): raise InvalidInput('unsupported OMP YAML syntax; manually simplify activation layout')
        if not line.startswith(' '):
            if marker: raise InvalidInput('unterminated OMP activation marker')
            finish(i); inside = False
            top = re.fullmatch(r'([A-Za-z_][A-Za-z0-9_-]*):(?:\s+(.*))?', line)
            if not top or (top[2] and re.search(r'''[\[\]{}"']''', top[2]) and top[1] != 'providers') or top[1] in roots:
                raise InvalidInput('unsupported or duplicate OMP YAML root')
            roots.add(top[1])
            if top[1] == 'providers':
                if top[2] and not re.fullmatch(r'\{\}(?:\s*#.*)?', top[2]) and not top[2].startswith('#'):
                    raise InvalidInput('unsupported OMP providers mapping')
                inside = not top[2] or top[2].startswith('#')
            continue
        if not inside:
            if not re.fullmatch(r''' +[A-Za-z0-9_.-]+:(?:\s+[^\[\]{}"']*)?''', line):
                raise InvalidInput('unsupported foreign OMP YAML layout')
            continue
        key = re.fullmatch(r'  ([A-Za-z0-9][A-Za-z0-9_.-]*):\s*(?:#.*)?', line)
        if key:
            if marker and provider: raise InvalidInput('nested OMP activation marker')
            finish(i); provider, start, valid, auth = key[1], i, True, False
            if marker and marker[1] != provider: raise InvalidInput('OMP marker provider mismatch')
            continue
        if not provider or not re.match(r'^    [A-Za-z_][A-Za-z0-9_-]*:', line): raise InvalidInput('unsupported OMP provider layout')
        if re.fullmatch(r'    auth: none\s*(?:#.*)?', line):
            if auth: raise InvalidInput('duplicate OMP auth key')
            auth = True
        else: valid = False
    if marker: raise InvalidInput('unterminated OMP activation marker')
    finish(len(lines))
    return found


def selected_file(target):
    if (target / 'models.json').exists() or (target / 'models.json').is_symlink():
        raise InvalidInput('legacy models.json requires explicit user migration')
    for name in ('models.yml', 'models.yaml'):
        path = target / name
        if path.exists() or path.is_symlink(): return path
    return target / 'models.yml'


def owned_stanza(identity, provider):
    return (f'  # loom-control omp begin {identity} {provider}\n'
            f'  {provider}:\n    auth: none\n'
            f'  # loom-control omp end {identity} {provider}\n')


def edit_plan(data, identity, providers, previous, *, disable=False, heading_state=None):
    """Return replacement and minimal exact patches, excluding foreign content."""
    text = '' if data is None else data.decode('utf8')
    rows = scan(text)
    for name, record in previous.items():
        row = rows.get(name)
        if not row or not row['valid'] or hashlib.sha256(row['text'].encode()).hexdigest() != record['sha256'] or row['owner'] != (identity if record['ownership'] == 'owned' else None):
            raise InvalidInput('edited/missing OMP activation retained: ' + name)
    patches = []
    heading_state = heading_state or None
    if disable:
        for name, record in previous.items():
            if record['ownership'] == 'owned':
                row = rows[name]; patches.append(dict(start=row['start'], old=row['text'], new=''))
    else:
        missing = []
        for name in providers:
            row = rows.get(name)
            if row and (not row['valid'] or row['owner'] not in (None, identity)):
                raise InvalidInput('conflicting OMP provider activation: ' + name + '; resolve manually')
            if row and row['owner'] == identity and name not in previous:
                raise InvalidInput('untracked OMP owned marker retained: ' + name)
            if row is None: missing.append(owned_stanza(identity, name))
        if missing:
            heading = re.search(r'^providers:(?:[^\n]*)\n?', text, re.M)
            if heading:
                old = heading[0]
                if re.match(r'providers:\s*\{\}', old):
                    new = re.sub(r'\{\} *', '', old, count=1)
                    if not new.endswith('\n'): new += '\n'
                    heading_state = dict(original=old, replacement=new, created_file=False)
                    patches.append(dict(start=heading.start(), old=old, new=new))
                    at = heading.end()
                else: at = heading.end()
                if not old.endswith('\n'):
                    if not patches:
                        heading_state = dict(original=old, replacement=old+'\n', created_file=False)
                        patches.append(dict(start=heading.start(), old=old, new=old+'\n'))
                patches.append(dict(start=at, old='', new=''.join(missing)))
            else:
                prefix = '' if not text or text.endswith('\n') else '\n'
                new = prefix + 'providers:\n' + ''.join(missing)
                heading_state = dict(original='', replacement=prefix+'providers:\n', created_file=data is None)
                patches.append(dict(start=len(text), old='', new=new))
    for patch in sorted(patches, key=lambda p:p['start'], reverse=True):
        if text[patch['start']:patch['start']+len(patch['old'])] != patch['old']: raise InvalidInput('OMP patch anchor changed')
        text = text[:patch['start']] + patch['new'] + text[patch['start']+len(patch['old']):]
    if disable:
        # Restore only our exact heading when no foreign provider needs it.
        if heading_state and not scan(text):
            replacement = heading_state['replacement']
            at = text.find(replacement)
            if at >= 0 and text.count(replacement) == 1:
                patches.append(dict(start=at, old=replacement, new=heading_state['original'], final=True))
                text = text[:at] + heading_state['original'] + text[at+len(replacement):]
                if heading_state['created_file'] and not text: return None, patches, heading_state
    scan(text)
    return text.encode(), patches, heading_state
