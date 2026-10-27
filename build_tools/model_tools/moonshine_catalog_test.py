"""Existing recognition native framing ABI assertions, independent of packaging selectors."""
from pathlib import Path
import unittest

class RecognitionSourceTest(unittest.TestCase):
    def setUp(self):
        self.source = Path(__file__).resolve().parents[2] / 'loom-pipeline/recognition/moonshine/streaming_tiny'

    def test_recognition_source_abi_and_framing_bounds(self):
        prepare=(self.source/'prepare.loom').read_text(); control=(self.source/'control.loom').read_text()
        signature='-> ('+', '.join(['i64']*14+['i32']+['buffer']*4)+')'
        self.assertIn(signature,prepare)
        self.assertIn('%event_capacity = scalar.constant 1 : i64',prepare)
        self.assertIn('%outputs = scalar.constant 1 : i32',prepare)
        self.assertIn('%event_bytes = scalar.constant 16444 : i64',prepare)
        self.assertIn('%temperature: f32, %outputs: i32) -> (buffer, buffer)',control)
        self.assertIn('%outputs_ok = scalar.cmpi eq, %outputs, %transcript_output : i32',control)
        self.assertIn('func.call @transcript(%text, %length, %revision, %end, %final)',control)
        self.assertIn('%prefix_length = index.constant 12 : offset',control)
        self.assertIn('view.store %one, %header[0]',control)
        self.assertIn('view.store %zero, %header[1]',control)
        self.assertIn('view.store %length32, %header[2]',control)
        self.assertIn('func.return %count, %packet, %event_bytes, %complete : i64, buffer, i64, i32',control)
        # Empty reset/no-event paths remain genuinely empty, not a count-zero packet.
        self.assertIn('scf.yield %zero64, %empty, %zero : i64, buffer, i32',control)
        self.assertIn('%packet_length = buffer.length %packet',control)


if __name__ == '__main__': unittest.main()
