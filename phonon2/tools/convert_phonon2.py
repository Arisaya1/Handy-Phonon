"""Convert Phonon-2 (model.fermion) into Handy's Parakeet ONNX layout.

Takes the istupakov parakeet-tdt-0.6b-v3 fp32 ONNX graphs (same architecture as
Phonon-2's teacher) and swaps every weight for Phonon-2's dequantized one.
Then writes fp32 + int8 (per-channel) models.

Usage: python convert_phonon2.py  (paths are relative to the phonon2 folder)
"""
import os
import re
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "phonon2_hf"))
from fermion_container import read_container  # noqa: E402

BASE = ROOT / "base_onnx"
OUT = ROOT / "build" / "phonon-2-fp32"
BN_EPS = 1e-5


def f32(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float32))


def lstm_reorder(w):
    # torch gate order i,f,g,o -> onnx i,o,f,c
    i, f, g, o = np.split(w, 4, axis=0)
    return np.concatenate([i, o, f, g], axis=0)


def build_maps(sd):
    """Returns {onnx_init_name_or_node_path: array} keyed by node path for anonymous inits."""
    named = {}   # direct initializer names
    by_node = {}  # (node_name, input_index) -> array
    # --- subsampling
    for k in (0, 2, 3, 5, 6):
        named[f"pre_encode.conv.{k}.weight"] = f32(sd[f"encoder.subsampling.layers.{k}.weight"])
        named[f"pre_encode.conv.{k}.bias"] = f32(sd[f"encoder.subsampling.layers.{k}.bias"])
    named["pre_encode.out.bias"] = f32(sd["encoder.subsampling.linear.bias"])
    by_node[("/pre_encode/out/MatMul", 1)] = f32(sd["encoder.subsampling.linear.weight"]).T
    n_layers = 1 + max(int(m.group(1)) for k in sd for m in [re.match(r"encoder\.layers\.(\d+)\.", k)] if m)
    for L in range(n_layers):
        p = f"encoder.layers.{L}."
        for nm in ("norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2", "norm_out"):
            named[f"layers.{L}.{nm}.weight"] = f32(sd[p + nm + ".weight"])
            named[f"layers.{L}.{nm}.bias"] = f32(sd[p + nm + ".bias"])
        named[f"layers.{L}.self_attn.pos_bias_u"] = f32(sd[p + "self_attn.bias_u"])
        named[f"layers.{L}.self_attn.pos_bias_v"] = f32(sd[p + "self_attn.bias_v"])
        named[f"layers.{L}.conv.pointwise_conv1.weight"] = f32(sd[p + "conv.pointwise_conv1.weight"])[:, :, None]
        named[f"layers.{L}.conv.pointwise_conv2.weight"] = f32(sd[p + "conv.pointwise_conv2.weight"])[:, :, None]
        lin = {
            "feed_forward1/linear1": "feed_forward1.linear1", "feed_forward1/linear2": "feed_forward1.linear2",
            "feed_forward2/linear1": "feed_forward2.linear1", "feed_forward2/linear2": "feed_forward2.linear2",
            "self_attn/linear_q": "self_attn.q_proj", "self_attn/linear_k": "self_attn.k_proj",
            "self_attn/linear_v": "self_attn.v_proj", "self_attn/linear_out": "self_attn.o_proj",
            "self_attn/linear_pos": "self_attn.relative_k_proj",
        }
        for onnx_path, hf in lin.items():
            by_node[(f"/layers.{L}/{onnx_path}/MatMul", 1)] = f32(sd[p + hf + ".weight"]).T
        # depthwise conv with BatchNorm folded in (that's how the base export stores it)
        w = f32(sd[p + "conv.depthwise_conv.weight"])
        g, b = f32(sd[p + "conv.norm.weight"]), f32(sd[p + "conv.norm.bias"])
        mu, var = f32(sd[p + "conv.norm.running_mean"]), f32(sd[p + "conv.norm.running_var"])
        s = g / np.sqrt(var + BN_EPS)
        dw_bias = sd.get(p + "conv.depthwise_conv.bias")
        b0 = f32(dw_bias) if dw_bias is not None else np.zeros_like(mu)
        by_node[(f"/layers.{L}/conv/depthwise_conv/Conv", 1)] = w * s[:, None, None]
        by_node[(f"/layers.{L}/conv/depthwise_conv/Conv", 2)] = (b0 - mu) * s + b
    # --- decoder / joint
    named["decoder.prediction.embed.weight"] = f32(sd["decoder.embedding.weight"])
    named["joint.enc.bias"] = f32(sd["encoder_projector.bias"])
    named["joint.pred.bias"] = f32(sd["decoder.decoder_projector.bias"])
    named["joint.joint_net.2.bias"] = f32(sd["joint.head.bias"])
    by_node[("/joint/enc/MatMul", 1)] = f32(sd["encoder_projector.weight"]).T
    by_node[("/joint/pred/MatMul", 1)] = f32(sd["decoder.decoder_projector.weight"]).T
    by_node[("/joint/joint_net/joint_net.2/MatMul", 1)] = f32(sd["joint.head.weight"]).T
    for li, node in ((0, "/decoder/dec_rnn/lstm/LSTM"), (1, "/decoder/dec_rnn/lstm/LSTM_1")):
        W = lstm_reorder(f32(sd[f"decoder.lstm.weight_ih_l{li}"]))[None]
        R = lstm_reorder(f32(sd[f"decoder.lstm.weight_hh_l{li}"]))[None]
        B = np.concatenate([lstm_reorder(f32(sd[f"decoder.lstm.bias_ih_l{li}"])),
                            lstm_reorder(f32(sd[f"decoder.lstm.bias_hh_l{li}"]))])[None]
        by_node[(node, 1)], by_node[(node, 2)], by_node[(node, 3)] = W, R, B
    return named, by_node


def swap(model_path, named, by_node, used):
    m = onnx.load(str(model_path), load_external_data=True)
    inits = {t.name: t for t in m.graph.initializer}
    targets = dict(named)
    for node in m.graph.node:
        for idx, inp in enumerate(node.input):
            if (node.name, idx) in by_node:
                targets[inp] = by_node[(node.name, idx)]
                used.add((node.name, idx))
    replaced = 0
    for name, arr in targets.items():
        if name not in inits:
            continue
        t = inits[name]
        old = numpy_helper.to_array(t)
        if old.shape != arr.shape:
            raise ValueError(f"{name}: shape {old.shape} vs {arr.shape}")
        t.CopyFrom(numpy_helper.from_array(arr.astype(old.dtype), name))
        used.add(name)
        replaced += 1
    left = [n for n in inits if n not in targets]
    return m, replaced, left


def main():
    tensors, _ = read_container(str(ROOT / "phonon2_hf" / "model.fermion"))
    named, by_node = build_maps(tensors)
    OUT.mkdir(parents=True, exist_ok=True)
    used = set()
    enc, n_enc, left_enc = swap(BASE / "encoder-model.onnx", named, by_node, used)
    dec, n_dec, left_dec = swap(BASE / "decoder_joint-model.onnx", named, by_node, used)
    print(f"encoder: replaced {n_enc}, untouched {left_enc}")
    print(f"decoder: replaced {n_dec}, untouched {left_dec}")
    missing = [k for k in list(named) + list(by_node) if k not in used]
    if missing or left_enc or left_dec:
        raise SystemExit(f"unmapped: {missing[:10]} / {left_enc[:10]} / {left_dec[:10]}")
    for f in OUT.glob("*"):
        f.unlink()
    onnx.save(enc, str(OUT / "encoder-model.onnx"), save_as_external_data=True,
              all_tensors_to_one_file=True, location="encoder-model.onnx.data")
    onnx.save(dec, str(OUT / "decoder_joint-model.onnx"))
    for f in ("nemo128.onnx", "vocab.txt", "config.json"):
        (OUT / f).write_bytes((BASE / f).read_bytes())
    print("wrote", OUT)


if __name__ == "__main__":
    main()
