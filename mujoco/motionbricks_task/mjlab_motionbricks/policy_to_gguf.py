#!/usr/bin/env python3
"""Export a trained rsl-rl actor to GGUF for ggml inference (no ONNX).

The deployed policy is deterministic: action = mlp(normalize(obs)), where
normalize(x) = (x - mean) / (std + eps) and mlp is the actor Linear stack with
ELU between hidden layers. This writes the normalizer and the Linear weights to
a GGUF v3 file (same hand-rolled layout as scripts/convert_to_gguf.py) so ggml
can run the policy, and records an osquery host snapshot for provenance.

    pixi run python -m mjlab_motionbricks.policy_to_gguf --checkpoint <model.pt> --out policies/smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
ALIGNMENT = 32
GGML_TYPE_F32 = 0
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8
GGUF_TYPE_UINT64 = 10
NORM_EPS = 1e-2  # rsl-rl EmpiricalNormalization default.


def encoded(value: str) -> bytes:
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def kv_string(key: str, value: str) -> bytes:
    return encoded(key) + struct.pack("<I", GGUF_TYPE_STRING) + encoded(value)


def kv_u32(key: str, value: int) -> bytes:
    return encoded(key) + struct.pack("<II", GGUF_TYPE_UINT32, value)


def kv_f32(key: str, value: float) -> bytes:
    return encoded(key) + struct.pack("<If", GGUF_TYPE_FLOAT32, value)


def kv_u64(key: str, value: int) -> bytes:
    return encoded(key) + struct.pack("<IQ", GGUF_TYPE_UINT64, value)


def aligned(offset: int) -> int:
    return (offset + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


@dataclass
class Tensor:
    name: str
    value: np.ndarray
    dimensions: tuple[int, ...]
    offset: int

    @property
    def size(self) -> int:
        return int(self.value.size) * 4

    def payload(self) -> bytes:
        return np.asarray(self.value, dtype="<f4", order="C").tobytes(order="C")


def osquery_snapshot() -> dict:
    """Host state for the policy's provenance manifest. Empty on any failure."""
    queries = {
        "system_info": "SELECT computer_name, cpu_brand, physical_memory FROM system_info;",
        "os_version": "SELECT name, version, platform FROM os_version;",
        "kernel_info": "SELECT version FROM kernel_info;",
    }
    out: dict = {}
    for key, sql in queries.items():
        try:
            proc = subprocess.run(["osqueryi", "--json", sql], capture_output=True, text=True, timeout=30)
            out[key] = json.loads(proc.stdout) if proc.returncode == 0 and proc.stdout.strip() else []
        except (OSError, ValueError):
            out[key] = []
    return out


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return x @ weight.T + bias


def elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0.0, x, np.expm1(x))


def forward_numpy(obs: np.ndarray, tensors: dict[str, np.ndarray]) -> np.ndarray:
    x = (obs - tensors["norm.mean"]) / (tensors["norm.std"] + NORM_EPS)
    for i, layer in enumerate((0, 2, 4, 6)):
        x = linear(x, tensors[f"mlp.{layer}.weight"], tensors[f"mlp.{layer}.bias"])
        if layer != 6:
            x = elu(x)
    return x


def export(checkpoint: Path, out_dir: Path) -> dict:
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)["actor_state_dict"]
    tensors: dict[str, np.ndarray] = {
        "norm.mean": state["obs_normalizer._mean"].reshape(-1).float().numpy(),
        "norm.std": state["obs_normalizer._std"].reshape(-1).float().numpy(),
    }
    for layer in (0, 2, 4, 6):
        tensors[f"mlp.{layer}.weight"] = state[f"mlp.{layer}.weight"].float().numpy()
        tensors[f"mlp.{layer}.bias"] = state[f"mlp.{layer}.bias"].float().numpy()
    obs_dim = int(tensors["norm.mean"].shape[0])
    action_dim = int(tensors["mlp.6.bias"].shape[0])

    # Parity: the numpy reimplementation must match a torch run of the same math.
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((8, obs_dim)).astype(np.float32)
    ref = obs.copy()
    ref = (ref - tensors["norm.mean"]) / (tensors["norm.std"] + NORM_EPS)
    with torch.no_grad():
        t = torch.from_numpy(ref)
        for layer in (0, 2, 4, 6):
            t = torch.nn.functional.linear(t, state[f"mlp.{layer}.weight"].float(), state[f"mlp.{layer}.bias"].float())
            if layer != 6:
                t = torch.nn.functional.elu(t)
        torch_action = t.numpy()
    numpy_action = forward_numpy(obs, tensors)
    max_abs_diff = float(np.max(np.abs(numpy_action - torch_action)))

    # Write the GGUF (F32, dims reversed, aligned 32).
    order = sorted(tensors)
    infos: list[Tensor] = []
    offset = 0
    param_count = 0
    for name in order:
        value = tensors[name]
        dims = tuple(reversed(tuple(int(d) for d in value.shape))) or (1,)
        infos.append(Tensor(name, value, dims, offset))
        offset = aligned(offset + int(value.size) * 4)
        param_count += int(value.size)

    metadata = [
        kv_string("general.architecture", "mlp-policy"),
        kv_string("general.name", "MotionBricks G1 PPO actor"),
        kv_u32("general.alignment", ALIGNMENT),
        kv_u32("policy.obs_dim", obs_dim),
        kv_u32("policy.action_dim", action_dim),
        kv_string("policy.activation", "elu"),
        kv_string("policy.hidden_dims", "512,256,128"),
        kv_f32("policy.norm_eps", NORM_EPS),
        kv_string("policy.checkpoint_sha256", hashlib.sha256(checkpoint.read_bytes()).hexdigest()),
        kv_u64("policy.parameter_count", param_count),
    ]
    header = bytearray(GGUF_MAGIC)
    header += struct.pack("<IQQ", GGUF_VERSION, len(infos), len(metadata))
    for item in metadata:
        header += item
    for tensor in infos:
        header += encoded(tensor.name)
        header += struct.pack("<I", len(tensor.dimensions))
        header += struct.pack("<" + "Q" * len(tensor.dimensions), *tensor.dimensions)
        header += struct.pack("<IQ", GGML_TYPE_F32, tensor.offset)
    header += bytes(aligned(len(header)) - len(header))

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "policy.gguf"
    with path.open("wb") as stream:
        stream.write(header)
        position = 0
        for tensor in infos:
            stream.write(bytes(tensor.offset - position))
            stream.write(tensor.payload())
            position = tensor.offset + tensor.size
        stream.write(bytes(offset - position))
        stream.flush()
        os.fsync(stream.fileno())

    manifest = {
        "checkpoint": str(checkpoint),
        "gguf": str(path),
        "gguf_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "activation": "elu",
        "norm_eps": NORM_EPS,
        "parity_numpy_vs_torch_max_abs_diff": max_abs_diff,
        "host_provenance": osquery_snapshot(),
    }
    (out_dir / "policy_gguf_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "host_provenance"}, indent=2))
    print(f"wrote {path}: {len(infos)} tensors, {param_count:,} values, parity {max_abs_diff:.2e}")
    if max_abs_diff > 1e-4:
        raise SystemExit(f"FAIL: numpy/torch parity {max_abs_diff:.2e} exceeds 1e-4")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("policies/smoke"))
    args = ap.parse_args()
    export(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
