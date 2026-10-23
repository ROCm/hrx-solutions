"""Fixed helper packaging contracts with synthetic tool bytes, never model data."""
import hashlib
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import zipfile

from build_tools.model_tools import moonshine_preparer as preparer
from binding.python.models import moonshine_preparation


class MoonshinePreparerTest(unittest.TestCase):
    def test_reviewed_tool_identity_shared_with_runtime(self):
        self.assertEqual(preparer.FILES, moonshine_preparation.TOOL_FILES)
        self.assertEqual(preparer.REVISION, moonshine_preparation.REVISION)

    def test_staged_tool_license_provenance_and_archive_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'hrx'
            tool = source / preparer.UPSTREAM
            tool.mkdir(parents=True)
            contents = {'prepare_assets.py': b'# trusted fixture', 'model.json': b'{}'}
            checksums = {name: hashlib.sha256(value).hexdigest() for name,value in contents.items()}
            for name,value in contents.items():
                (tool/name).write_bytes(value)
            (source/'LICENSE').write_text('notice')
            with patch.object(preparer, 'FILES', checksums):
                native = root/'package/_native'
                outputs = preparer.stage(native, source)
                destination = native / preparer.RELATIVE
                self.assertEqual(outputs, [destination / name for name in (*contents, 'LICENSE', 'provenance.json')])
                self.assertEqual(set(outputs), set(destination.iterdir()))
                runtime_license = native/'share/loom-serve/LICENSE'
                runtime_license.write_text('notice')
                archive_path = root/'wheel.zip'
                def archive():
                    with zipfile.ZipFile(archive_path, 'w') as wheel:
                        for file in native.rglob('*'):
                            if file.is_file():
                                wheel.write(file, 'hrx_solutions/'+file.relative_to(root/'package').as_posix())
                    return zipfile.ZipFile(archive_path)
                with archive() as wheel:
                    preparer.validate_archive(wheel)
                (native/preparer.RELATIVE/'prepare_assets.py').write_text('tampered')
                with archive() as wheel, self.assertRaisesRegex(ValueError, 'bytes differ'):
                    preparer.validate_archive(wheel)
                (tool/'prepare_assets.py').write_text('changed upstream input')
                with self.assertRaisesRegex(ValueError, 'immutable'):
                    preparer.stage(root/'other-native', source)


if __name__ == '__main__':
    unittest.main()
