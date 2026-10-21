"""Ephemeral privileged observer: retain only one owned native request summary."""
import json
from pathlib import Path
import socket
import struct
import sys
import time


def capture(port, output, seconds=30):
    buffers = {}
    with socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_TCP) as sock:
        sock.settimeout(.2)
        output.with_suffix('.ready').touch()
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            try:
                packet = sock.recv(65535)
            except socket.timeout:
                continue
            ip = (packet[0] & 15) * 4
            if packet[12:20] != b'\x7f\x00\x00\x01' * 2:
                continue
            source, destination, sequence = struct.unpack('!HHI', packet[ip:ip+8])
            if destination != port:
                continue
            payload = packet[ip + (packet[ip+12] >> 4)*4:]
            if not payload:
                continue
            if payload.startswith(b'POST /v1/chat/completions '):
                buffers[source] = (sequence, payload)
            elif source in buffers:
                previous, data = buffers[source]
                if sequence == previous + len(data):
                    buffers[source] = (previous, data + payload)
            if source not in buffers:
                continue
            data = buffers[source][1]
            if len(data) > 1048576:
                raise RuntimeError('owned capture budget exceeded')
            if b'\r\n\r\n' not in data:
                continue
            head, body = data.split(b'\r\n\r\n', 1)
            names = dict(line.split(':', 1) for line in head.decode().split('\r\n')[1:])
            headers = {name.lower(): value.strip() for name, value in names.items()}
            length = int(headers.get('content-length', '0'))
            if len(body) < length:
                continue
            request = json.loads(body[:length])
            output.write_text(json.dumps(dict(port=port, path='/v1/chat/completions',
                header_names=sorted(headers), model=request.get('model'),
                max_tokens=request.get('max_tokens'), stream=request.get('stream'))))
            return
    raise TimeoutError('no owned native request observed')


if __name__ == '__main__':
    capture(int(sys.argv[1]), Path(sys.argv[2]))
