"""CLI dispatch and help checks for Loom Control TUI."""

import contextlib
import io
import unittest
from unittest.mock import patch

from binding.python import hem_cli


class TuiCliTest(unittest.TestCase):
    def test_cli_dispatch_and_help(self):
        with patch("binding.python.hem_tui.run", return_value=0) as run:
            self.assertEqual(hem_cli.main(["tui"]), 0)
            run.assert_called_once()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with self.assertRaises(SystemExit) as raised:
                hem_cli.main(["tui", "--help"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("loom-control tui", output.getvalue())
        args = hem_cli.parse_args(["tui"])
        self.assertEqual((args.resource, args.command), ("tui", "run"))


if __name__ == "__main__":
    unittest.main()
