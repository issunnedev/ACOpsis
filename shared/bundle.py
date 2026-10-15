import itertools
import json
import math
import struct
import threading
import types

import torch
from safetensors.torch import save_file

_FP8 = (torch.float8_e4m3fn, torch.float8_e5m2)
_DTYPES = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
           "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2, "I64": torch.int64,
           "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}
_CODES = {v: k for k, v in _DTYPES.items()}


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

    def get_rows(self, name, rows, device="cpu"):
        info = self.header[name]
        dtype = _DTYPES[info["dtype"]]
        width = info["shape"][1]
        row_bytes = width * torch.tensor([], dtype=dtype).element_size()
        buf = bytearray(row_bytes * len(rows))
        view = memoryview(buf)
        with self._lock:
            for i, row in enumerate(rows):
                self._file.seek(self._base + info["data_offsets"][0] + row * row_bytes)
                self._file.readinto(view[i * row_bytes:(i + 1) * row_bytes])
        return torch.frombuffer(buf, dtype=dtype).reshape(len(rows), width).to(device)

    def close(self):
        self._file.close()


def write_safetensors(path, entries):
    header, offset = {}, 0
    for name, dtype, shape, _ in entries:
        size = math.prod(shape) * torch.tensor([], dtype=dtype).element_size()
        header[name] = {"dtype": _CODES[dtype], "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for name, dtype, shape, make in entries:
            t = make().to(dtype).contiguous().cpu().reshape(-1)
            f.write(t.view(torch.uint8).numpy().tobytes())


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


def _assign(module, key, t):
    kind, name = key.split(".", 1)
    owner_name, _, attr = name.rpartition(".")
    owner = module.get_submodule(owner_name) if owner_name else module
    if kind == "p":
        owner._parameters[attr] = torch.nn.Parameter(t, requires_grad=False)
    elif kind == "c":
        owner._buffers[attr] = torch.view_as_complex(t)
    else:
        owner._buffers[attr] = t


def _check(module, path):
    missing = [n for n, t in itertools.chain(module.named_parameters(remove_duplicate=False),
                                              module.named_buffers(remove_duplicate=False)) if t.is_meta]
    if missing:
        raise RuntimeError(f"{path} doesn't cover {len(missing)} tensors, e.g. {missing[:3]}")


def load_bundle(module, path, device="cuda"):
    src = SafetensorsFile(path)
    try:
        for key in src.keys():
            _assign(module, key, src.get(key, device))
    finally:
        src.close()
    _check(module, path)
    return module.eval()


def _streamed_linear(self, x):
    src, key = self._weight_src
    return torch.nn.functional.linear(x, src.get(key, x.device).to(x.dtype), self.bias)


def _upcast_linear(self, x):
    return torch.nn.functional.linear(x, self.weight.to(x.dtype), self.bias)


def _streamed_embedding(self, ids):
    src, key = self._weight_src
    unique, inverse = torch.unique(ids.reshape(-1), return_inverse=True)
    rows = src.get_rows(key, unique.tolist(), ids.device)
    return rows[inverse].reshape(*ids.shape, rows.shape[-1])


def attach(module, path, device="cuda", stream_linear=True, stream_embeddings=()):
    src = SafetensorsFile(path)
    streamed = set()
    for name, m in module.named_modules():
        linear = isinstance(m, torch.nn.Linear)
        if (linear and stream_linear) or (isinstance(m, torch.nn.Embedding) and name in stream_embeddings):
            m.weight = None
            m._weight_src = (src, f"p.{name}.weight")
            m.forward = types.MethodType(_streamed_linear if linear else _streamed_embedding, m)
            streamed.add(id(m))
        elif linear:
            m.forward = types.MethodType(_upcast_linear, m)
    for key in src.keys():
        owner_name, _, attr = key.split(".", 1)[1].rpartition(".")
        owner = module.get_submodule(owner_name) if owner_name else module
        if id(owner) in streamed and attr == "weight":
            continue
        _assign(module, key, src.get(key, device))
    _check(module, path)
    module._bundle_src = src
    return module.eval()


def convert_to_bundle(sources, model, out_path, extra=None):
    linear = {f"{n}.weight" for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    files = [SafetensorsFile(p) for p in sources]
    entries = []
    for f in files:
        for key in f.keys():
            src_dtype = _DTYPES[f.header[key]["dtype"]]
            if key in linear:
                dtype = torch.float8_e4m3fn
            elif src_dtype.is_floating_point:
                dtype = torch.bfloat16
            else:
                dtype = src_dtype
            entries.append((f"p.{key}", dtype, f.header[key]["shape"], _reader(f, key, dtype)))
    for key, (dtype, shape, make) in (extra or {}).items():
        entries.append((key, dtype, shape, make))
    write_safetensors(out_path, entries)
    for f in files:
        f.close()


def _reader(f, key, dtype):
    if dtype in _FP8:
        return lambda: f.get(key).float().clamp(-448, 448)
    return lambda: f.get(key)
