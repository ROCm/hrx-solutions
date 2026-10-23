import io
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from binding.python.models.gguf import ArrayInfo, GGUFError, inspect_gguf, inspect_gguf_set
from tests.gguf_fixtures import write_gguf


class GGUFReaderTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / 'model.gguf'

    def test_mixed_types_arrays_and_header_only_reads(self):
        header, start = write_gguf(self.path, metadata=[
            ('general.architecture',8,'qwen35'), ('sections',9,(4,[11,11,10,0])),
            ('tokens',9,(8,['token']*4097)), ('numbers',9,(5,[0]*4097))],
            tensors=[('a',(256,2),13,0),('b',(8,),0,352)])
        original_open = Path.open
        reads = []
        class Tracked(io.FileIO):
            def read(self, count=-1):
                reads.append((self.tell(),count))
                return super().read(count)
        def tracked_open(path, *args, **kwargs):
            return Tracked(path, 'rb')
        with patch.object(Path, 'open', tracked_open):
            result = inspect_gguf(self.path)
        self.assertEqual(result.header_bytes, header)
        self.assertEqual(result.data_offset, start)
        self.assertEqual(result.metadata['sections'].value.values,(11,11,10,0))
        self.assertEqual(result.metadata['tokens'].value,ArrayInfo(8,4097))
        self.assertEqual(result.metadata['numbers'].value,ArrayInfo(5,4097))
        self.assertEqual(result.tensors['a'].size_bytes,352)
        self.assertEqual(result.tensors['a'].type_name,'Q5_K')
        self.assertTrue(all(count >= 0 and offset + count <= header for offset,count in reads))
        self.assertLess(sum(count for _,count in reads), header)

    def test_versions_and_symlink(self):
        for version in (2,3):
            write_gguf(self.path,version=version,tensors=[('a',(1,),0,0)])
            self.assertEqual(inspect_gguf(self.path).version,version)
        alias = self.path.with_name('alias.gguf')
        alias.symlink_to(self.path)
        self.assertEqual(inspect_gguf(alias).tensors,inspect_gguf(self.path).tensors)
        for version in (1,4,0x03000000):
            write_gguf(self.path,version=version)
            with self.assertRaisesRegex(GGUFError,'version or byte order'):
                inspect_gguf(self.path)

    def test_malformed_descriptors(self):
        cases = [
            ([('a',(0,),0,0)], 'dimensions'),
            ([('a',(1,1,1,1,1),0,0)], 'dimension count'),
            ([('a',(16,16),13,0)], 'row width'),
            ([('a',(256,),99,0)], 'type 99'),
            ([('a',(1,),0,1)], 'not aligned'),
            ([('a',(1,),0,0),('a',(1,),0,32)], 'duplicate.*tensor'),
            ([('a',(16,),0,0),('b',(16,),0,32)], 'overlaps'),
            ([('a',(1<<63,256),13,0)], 'overflows'),
        ]
        for tensors, error in cases:
            with self.subTest(error=error):
                write_gguf(self.path,tensors=tensors)
                with self.assertRaisesRegex(GGUFError,error):
                    inspect_gguf(self.path)

    def test_bad_metadata_and_split(self):
        cases = [
            ([('k',4,0),('k',4,1)],'duplicate.*metadata'),
            ([('general.alignment',4,0)],'alignment'),
            ([('general.alignment',5,32)],'alignment'),
            ([('split.count',4,2)],'Split GGUF'),
            ([('bool',7,2)],'boolean'),
        ]
        for metadata,error in cases:
            with self.subTest(error=error):
                write_gguf(self.path,metadata=metadata)
                with self.assertRaisesRegex(GGUFError,error):
                    inspect_gguf(self.path)
        self.path.write_bytes(b'GGUF'+struct.pack('<IQQQ',3,0,1,1<<63))
        with self.assertRaisesRegex(GGUFError,'string exceeds'):
            inspect_gguf(self.path)

    def test_truncated_headers_and_payload(self):
        header,start=write_gguf(self.path,metadata=[('arch',8,'qwen35')],tensors=[('a',(256,),13,0)])
        original=self.path.read_bytes()
        for cutoff in (0,4,12,23,35,header-1,start+175):
            with self.subTest(cutoff=cutoff):
                self.path.write_bytes(original[:cutoff])
                with self.assertRaises(GGUFError):
                    inspect_gguf(self.path)

    def test_fragment_metadata_mismatch_and_duplicate_tensors_rejected(self):
        first = self.path.parent / 'first.gguf'
        second = self.path.parent / 'second.gguf'
        write_gguf(first, metadata=[('general.architecture', 8, 'qwen35')], tensors=[('a', [32], 0, 0)])
        write_gguf(second, metadata=[('general.architecture', 8, 'qwen35')], tensors=[('a', [32], 0, 0)])
        with self.assertRaisesRegex(GGUFError, 'Duplicate tensor'):
            inspect_gguf_set([first, second])
        write_gguf(second, metadata=[('general.architecture', 8, 'other')], tensors=[('b', [32], 0, 0)])
        with self.assertRaisesRegex(GGUFError, 'metadata'):
            inspect_gguf_set([first, second])


if __name__ == '__main__':
    unittest.main()
