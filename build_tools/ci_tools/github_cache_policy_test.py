# Copyright 2026 The HRX Authors
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT
"""Exercise actual workflow boolean gates with representative GitHub contexts."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]


def condition(file, marker):
    text = (ROOT / ".github/workflows" / file).read_text().split(marker, 1)[1]
    return re.search(r"^    +if: (.+)$", text, re.MULTILINE).group(1)


def evaluate(expression, values):
    # Gates intentionally use only comparisons, parentheses, && and ||.
    expression = re.sub(r"\b(?:github|inputs|steps|needs)\.[A-Za-z0-9_.]+",
                        lambda match: repr(values.get(match.group(), "")), expression)
    expression = expression.replace("&&", " and ").replace("||", " or ")
    return bool(eval(expression, {"__builtins__": {}}, {}))


class WorkflowPolicyTest(unittest.TestCase):
    def test_writer_and_cleanup_trust_and_branch_matrix(self):
        writer = condition("build_python_wheel.yml", "id: cache_admit")
        cleanup = condition("ci_python_wheel.yml", "  wheel_cache_cleanup:")
        baseline = {"inputs.use_bazel_cache": True, "inputs.cache_write_pilot": True,
                    "steps.cache_prepare.outputs.enabled": "true",
                    "github.repository": "ROCm/hrx-solutions", "github.ref": "refs/heads/main",
                    "github.event_name": "push", "needs.python_wheel.result": "success",
                    "needs.python_wheel.outputs.cache_attempted_key": "attempted"}
        cases = [({}, True), ({"github.event_name": "workflow_dispatch"}, False),
                 ({"github.ref": "refs/heads/pilot/wheel-bazel-cache", "github.event_name": "workflow_dispatch"}, True),
                 ({"github.ref": "refs/heads/pilot/wheel-bazel-cache"}, False),
                 ({"github.event_name": "pull_request"}, False),
                 ({"github.repository": "fork/hrx-solutions"}, False),
                 ({"github.ref": "refs/heads/other", "github.event_name": "workflow_dispatch"}, False),
                 ({"github.ref": "refs/heads/pilot/wheel-bazel-cache", "github.event_name": "workflow_dispatch", "inputs.cache_write_pilot": False}, False),
                 ({"github.ref": "refs/heads/pilot/wheel-bazel-cache", "github.event_name": "workflow_dispatch", "inputs.use_bazel_cache": False}, False)]
        for changes, expected in cases:
            with self.subTest(changes=changes):
                values = baseline | changes
                self.assertEqual(evaluate(writer, values), expected)
                self.assertEqual(evaluate(cleanup, values), expected)

    def test_model_job_runs_for_normal_dispatch_and_push(self):
        # Preserve the default dispatch selection that previously regressed by
        # silently skipping gfx1151.
        ci = (ROOT / '.github/workflows/ci_python_wheel.yml').read_text()
        dispatch = ci.split('      test_model:', 1)[1].split('      release_type:', 1)[0]
        self.assertIn('default: true', dispatch)
        model = condition('ci_python_wheel.yml', '  model:')
        for event, selected in [('push', True), ('pull_request', True),
                                 ('workflow_dispatch', True)]:
            self.assertEqual(evaluate(model, {'github.ref': 'refs/heads/main',
                'github.event_name': event, 'inputs.test_model': True}), selected)
        self.assertFalse(evaluate(model, {'github.ref': 'refs/heads/main',
            'github.event_name': 'workflow_dispatch', 'inputs.test_model': False}))
        self.assertIn('uses: ./.github/workflows/check_model.yml', ci.split('  model:', 1)[1])
        workflow = (ROOT / '.github/workflows/check_model.yml').read_text()
        self.assertIn('  model:', workflow)
        self.assertIn('runs-on: linux-gfx1151-gpu-rocm', workflow)
        self.assertEqual(workflow.count('name: hrx-solutions-wheel-linux-x86_64'), 1)
        self.assertNotIn('  openwebui:', workflow)
        self.assertNotIn('model-smoke.sh --openwebui', workflow)

    def test_pilot_model_is_skipped_even_when_explicitly_requested(self):
        model = condition("ci_python_wheel.yml", "  model:")
        for event in ("push", "workflow_dispatch", "pull_request"):
            self.assertFalse(evaluate(model, {"github.ref": "refs/heads/pilot/wheel-bazel-cache", "github.event_name": event, "inputs.test_model": True}))
        for repository in ("ROCm/hrx-solutions", "fork/hrx-solutions"):
            with self.subTest(repository=repository):
                self.assertTrue(evaluate(model, {"github.repository": repository, "github.ref": "refs/pull/1/merge", "github.head_ref": "pilot/wheel-bazel-cache", "github.event_name": "pull_request", "inputs.test_model": True}))
        self.assertTrue(evaluate(model, {"github.ref": "refs/heads/main", "github.event_name": "push", "inputs.test_model": False}))


if __name__ == "__main__":
    unittest.main()
