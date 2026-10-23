"""Fixed built-in setup dispatch; each integration owns its prerequisites."""
from .hem_common import InvalidInput


def configure_integration(integration, root, deadline, *, mode="configure", pi_dir=None, omp_dir=None, opencode_dir=None,
                          url=None, token=None, stt_server=None, stt_instance=None):
    """Apply one explicit intent without inspecting unrelated integrations."""
    if mode not in ("configure", "check", "disable"):
        raise InvalidInput("unsupported integration configuration mode")
    if integration == "pi":
        from .hem_pi import configure_pi
        return configure_pi(root, pi_dir, deadline, check=mode == "check", disable=mode == "disable")
    if integration == "omp":
        from .integrations.omp.hem_omp import configure_omp
        return configure_omp(root, omp_dir, deadline, check=mode == "check", disable=mode == "disable")
    if integration == "opencode":
        from .integrations.opencode.hem_opencode import configure_opencode
        return configure_opencode(root, opencode_dir, deadline, check=mode == "check", disable=mode == "disable")
    if integration == "openwebui":
        from .integrations.openwebui.hem_openwebui import configure_openwebui
        return configure_openwebui(root, url, token, deadline, check=mode == "check", disable=mode == "disable",
                                   stt_server=stt_server, stt_instance=stt_instance)
    raise InvalidInput("unsupported integration")
