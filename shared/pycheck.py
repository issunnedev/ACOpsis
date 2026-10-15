import platform
import struct
import sys

bits = struct.calcsize("P") * 8
machine = platform.machine().lower()
ok = (3, 10) <= sys.version_info[:2] <= (3, 14) and bits == 64 and machine in ("amd64", "x86_64")
print(f"Python {platform.python_version()}, {bits}-bit, {machine}: {'supported' if ok else 'not supported'}")
sys.exit(0 if ok else 1)
