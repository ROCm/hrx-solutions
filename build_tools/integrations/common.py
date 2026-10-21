"""Shared file snapshot and provider identity helpers for client qualifications."""
import base64
import hashlib


def snapshot(root):
    """File content + mtime make passive checks/idempotence falsifiable."""
    return {str(path.relative_to(root)): dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                             mtime_ns=path.stat().st_mtime_ns)
            for path in sorted(root.rglob('*')) if path.is_file()}


def provider(name):
    return 'loom-hem-' + base64.urlsafe_b64encode(name.encode()).decode().rstrip('=')


