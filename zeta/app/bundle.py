import itertools
import json
import struct
import threading

import torch
from safetensors.torch import save_file

_FP8 = (torch.float8_e4m3fn, torch.float8_e5m2)
_DTYPES = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
           "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2, "I64": torch.int64,
           "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}


class SafetensorsFile:
    def __init__(self, path):
        self.path = str(path)
        self._file = open(self.path, "rb")
        size = struct.unpack("<Q", self._file.read(8))[0]
        self.header = json.loads(self._file.read(size))
        self.header.pop("__metadata__", None)
        self._base = 8 + size
        self._lock = threading.Lock()

    def keys(self):
        return sorted(self.header, key=lambda k: self.header[k]["data_offsets"][0])

    def get(self, name, device="cpu"):
        info = self.header[name]
        start, end = info["data_offsets"]
        dtype = _DTYPES[info["dtype"]]
        if end == start:
            return torch.empty(info["shape"], dtype=dtype, device=device)
        buf = bytearray(end - start)
        with self._lock:
            self._file.seek(self._base + start)
            self._file.readinto(buf)
        return torch.frombuffer(buf, dtype=dtype).reshape(info["shape"]).to(device)

    def close(self):
        self._file.close()


def save_bundle(module, path, dtype=None):
    tensors, seen = {}, set()
    named = itertools.chain((("p", n, t) for n, t in module.named_parameters(remove_duplicate=False)),
                            (("b", n, t) for n, t in module.named_buffers(remove_duplicate=False)))
    for kind, name, t in named:
        t = t.detach()
        if dtype is not None and t.is_floating_point() and t.dtype not in _FP8:
            t = t.to(dtype)
        if t.is_complex():
            kind, t = "c", torch.view_as_real(t)
        t = t.contiguous().cpu()
        if t.data_ptr() in seen:
            t = t.clone()
        seen.add(t.data_ptr())
        tensors[f"{kind}.{name}"] = t
    save_file(tensors, str(path))


def load_bundle(module, path, device="cuda"):
    src = SafetensorsFile(path)
    try:
        for key in src.keys():
            t = src.get(key, device)
            kind, name = key.split(".", 1)
            owner_name, _, attr = name.rpartition(".")
            owner = module.get_submodule(owner_name) if owner_name else module
            if kind == "p":
                owner._parameters[attr] = torch.nn.Parameter(t, requires_grad=False)
            elif kind == "c":
                owner._buffers[attr] = torch.view_as_complex(t)
            else:
                owner._buffers[attr] = t
    finally:
        src.close()
    missing = [n for n, t in itertools.chain(module.named_parameters(remove_duplicate=False),
                                              module.named_buffers(remove_duplicate=False)) if t.is_meta]
    if missing:
        raise RuntimeError(f"{path} doesn't cover {len(missing)} tensors, e.g. {missing[:3]}")
    return module.eval()
