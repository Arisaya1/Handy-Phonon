"""Quantize the converted Phonon-2 ONNX to int8, compare transcripts, and build
Handy's download archive.

  python package_phonon2.py verify          # transcribe test_audio/*.wav with base + phonon fp32 + phonon int8
  python package_phonon2.py pack            # build build/phonon-2-int8.tar.gz and print sha256/size
"""
import hashlib
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FP32 = ROOT / "build" / "phonon-2-fp32"
INT8 = ROOT / "build" / "phonon-2-int8"
BASE = ROOT / "base_onnx"
ARCHIVE = ROOT / "build" / "phonon-2-int8.tar.gz"


def quantize():
    from onnxruntime.quantization import QuantType, quantize_dynamic
    INT8.mkdir(parents=True, exist_ok=True)
    for name in ("encoder-model", "decoder_joint-model"):
        quantize_dynamic(
            model_input=str(FP32 / f"{name}.onnx"),
            model_output=str(INT8 / f"{name}.int8.onnx"),
            weight_type=QuantType.QUInt8,  # signed int8 produced empty transcripts
            per_channel=False,
            op_types_to_quantize=["MatMul"],  # depthwise/pointwise convs stay fp32
        )
    for f in ("nemo128.onnx", "vocab.txt", "config.json"):
        shutil.copy(FP32 / f, INT8 / f)


def transcribe(model_dir, quant, wavs):
    import onnx_asr
    m = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", str(model_dir), quantization=quant)
    return [m.recognize(str(w)) for w in wavs]


def verify():
    wavs = sorted((ROOT / "test_audio").glob("*.wav"))
    if not (INT8 / "encoder-model.int8.onnx").exists():
        quantize()
    for label, d, q in (("parakeet-v3 fp32 (teacher)", BASE, None),
                        ("phonon-2 fp32", FP32, None),
                        ("phonon-2 int8", INT8, "int8")):
        for w, t in zip(wavs, transcribe(d, q, wavs)):
            print(f"[{label}] {w.name}: {t}")


def pack():
    if not (INT8 / "encoder-model.int8.onnx").exists():
        quantize()
    keep = ["encoder-model.int8.onnx", "decoder_joint-model.int8.onnx", "nemo128.onnx", "vocab.txt", "config.json"]
    with tarfile.open(ARCHIVE, "w:gz", compresslevel=9) as tar:
        for f in keep:
            tar.add(INT8 / f, arcname=f"phonon-2-int8/{f}")
    sha = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    size_mb = round(ARCHIVE.stat().st_size / 1e6)
    print(f"archive={ARCHIVE}\nsha256={sha}\nsize_mb={size_mb}")
    for f in keep:
        print(f"  {f}: {(INT8 / f).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    {"verify": verify, "pack": pack, "quantize": quantize}[sys.argv[1]]()
