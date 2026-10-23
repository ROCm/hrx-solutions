"""Existing narrow ownership admission, including its nonfinite/type policy."""
import json
import math
import subprocess
import sys
import unittest
import uuid

from binding.python import hem_client_documents, hem_omp, hem_opencode
from binding.python.hem_common import InvalidInput


class ClientDocumentTest(unittest.TestCase):
    def test_each_entry_preserves_context_safe_diagnostics_and_primitive_cause(self):
        invalid = [None, {}, 0, memoryview(b'{}'), b'[]', b'null', b'false',
                   b'{"secret":"PRIVATE","secret":2}', b'{"nested":{"x":1,"x":2}}',
                   b'{"x":', b'{} extra', b'\xff', b' ' * (1024 * 1024 + 1),
                   ('{"x":' * 65 + '0' + '}' * 65).encode()]
        for owner,label in ((hem_omp,'OMP'),(hem_opencode,'OpenCode')):
            for data in invalid:
                with self.subTest(client=label,kind=type(data).__name__), self.assertRaises(InvalidInput) as caught:
                    owner.document(data)
                self.assertEqual(str(caught.exception),'malformed '+label+' ownership data; '+owner.REPAIR)
                self.assertIsInstance(caught.exception.__cause__,(ValueError,TypeError,UnicodeError,RecursionError))
                self.assertNotIn('PRIVATE',str(caught.exception))

    def test_supported_json_types_order_and_nonfinite_constants_remain_admitted(self):
        text='{"z":[true,null,1,1.0,"text"],"a":{"x":-2}}'
        for owner in (hem_omp,hem_opencode):
            for data in (text,text.encode(),bytearray(text.encode())):
                with self.subTest(client=owner.__name__,kind=type(data).__name__):
                    value=owner.document(data)
                    self.assertEqual(value,json.loads(text))
                    self.assertEqual(list(value),['z','a'])
                    self.assertIs(type(value['z'][0]),bool)
                    self.assertIs(type(value['z'][2]),int)
                    self.assertIs(type(value['z'][3]),float)
            value=owner.document(b'{"nan":NaN,"pos":Infinity,"neg":-Infinity,"nested":[NaN]}')
            self.assertTrue(math.isnan(value['nan']) and math.isnan(value['nested'][0]))
            self.assertEqual((value['pos'],value['neg']),(math.inf,-math.inf))

    def test_exact_byte_and_depth_limits_and_existing_str_length_behavior(self):
        maximum=1024*1024
        data=b'{"x":"'+b'a'*(maximum-len(b'{"x":""}'))+b'"}'
        deep='{"x":'*64+'0'+'}'*64
        # json.loads accepts str; the original bound counts its characters.
        unicode_text='{"x":"'+'é'*(maximum//2)+'"}'
        for owner in (hem_omp,hem_opencode):
            with self.subTest(client=owner.__name__):
                self.assertEqual(len(owner.document(data)['x']),maximum-len(b'{"x":""}'))
                with self.assertRaises(InvalidInput):owner.document(data+b' ')
                self.assertIsInstance(owner.document(deep),dict)
                with self.assertRaises(InvalidInput):owner.document('{"x":'+deep+'}')
                self.assertEqual(owner.document(unicode_text)['x'],'é'*(maximum//2))
                with self.assertRaises(InvalidInput):owner.document(unicode_text.encode())

    def test_uuid_v4_canonical_spelling_is_unchanged(self):
        canonical='4f2d5064-bc08-48d2-b309-ed8912abbb1b'
        for owner in (hem_omp,hem_opencode):
            self.assertTrue(owner._uuid(canonical))
            for value in (canonical.upper(),canonical.replace('-',''),'{'+canonical+'}',
                          'urn:uuid:'+canonical,' '+canonical,
                          '4f2d5064-bc08-18d2-b309-ed8912abbb1b',None,True,4,b'uuid',uuid.UUID(canonical)):
                with self.subTest(client=owner.__name__,value=value):
                    self.assertFalse(owner._uuid(value))

    def test_pure_decoder_import_does_not_load_operational_owners(self):
        code=('import sys; from binding.python import hem_client_documents as module; '
              'prefix=module.__package__; '
              'assert not any(prefix+"."+name in sys.modules for name in '
              '("hem_state","hem_omp","hem_opencode","hem_common","hem_http","hem_clients_endpoints")); '
              'assert module.decode_document(b"{}") == {}')
        subprocess.run([sys.executable,'-c',code],check=True,timeout=5)
