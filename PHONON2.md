# Handy Phonon

A fork of [Handy](https://github.com/cjpais/Handy) that ships one speech-to-text model:
[Phonon-2](https://huggingface.co/FermionResearch/Phonon-2) by Fermion Research.

Phonon-2 is NVIDIA's Parakeet TDT 0.6B v3, retrained so each encoder weight is one of five
values. Fermion reports a 5.21% average word error rate on the Open ASR Leaderboard's seven
English sets, against 4.96% for the full-size Parakeet v3 it came from. Their release is 164 MB.

This fork exists because I wanted that model in Handy and nothing else.

## Download

Grab `Handy-Phonon-setup.exe` from the [releases page](https://github.com/Arisaya1/Handy-Phonon/releases).
On first run Handy downloads the model (410 MB, about 930 MB once unpacked) from the same
releases page, then you're set.

Windows x64 is the only build I've tested.

## How it works

Handy can't read Fermion's packed 2-bit file, but it can already run Parakeet v3 through
ONNX. Phonon-2 has exactly the same architecture as Parakeet v3, so the conversion swaps
weights instead of writing a new engine:

1. Unpack Phonon-2's `model.fermion` with Fermion's own reader. Each weight comes back as its
   real value (0, ±small or ±big per row, times a per-row scale).
2. Load [istupakov's fp32 ONNX export of Parakeet v3](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx),
   which is the layout Handy's Parakeet engine expects.
3. Replace all 625 weight tensors with Phonon-2's. Most map by name. The anonymous ones map
   through the ONNX node path (`/layers.3/self_attn/linear_q/MatMul` and so on). Two need
   real conversion work:
   - The export folds each conv block's BatchNorm into the depthwise conv, so Phonon-2's
     BatchNorm stats are folded in the same way.
   - PyTorch orders LSTM gates i,f,g,o and ONNX orders them i,o,f,c, so the decoder LSTM
     weights get reordered.
4. Quantize the MatMuls to 8-bit for Handy. Signed int8 produced empty transcripts, so this
   uses unsigned 8-bit, per tensor. Its output matched the fp32 model word for word on my test clips.

Going from 2-bit to 8-bit doesn't add information; the same five values per row are
written out in a roomier format. That's also why the model is 930 MB on disk instead of 164 MB.

The scripts are in [`phonon2/tools`](phonon2/tools).

## What changed from upstream Handy

- `src-tauri/src/managers/model.rs`: the model list is just Phonon-2. Catalog models,
  Hugging Face cache discovery and custom model discovery are off.
- `src/components/...`: Phonon-2 isn't hidden as a "legacy" ONNX download.
- `src-tauri/Cargo.toml`: the Whisper Vulkan backend is gone on Windows x64, since no
  Whisper model can be selected. This also means you don't need the Vulkan SDK to build.
- `src-tauri/src/settings.rs`: update checks are always off, so an official Handy release
  can't install over this fork.
- `src-tauri/tauri.phonon.conf.json`: builds as "Handy Phonon" (`com.phonon2.handy`), unsigned,
  NSIS installer only. It installs alongside regular Handy without touching its settings.

## Speed

Measured on the CPU of the PC I built it on, one clip at a time, with the model already loaded:

| Clip | Audio | Time | Speed |
|---|---|---|---|
| TTS sentence | 9.8 s | 0.42 s | 23× realtime |
| Short voice clip | 14.8 s | 0.62 s | 24× realtime |

That's the speed of Handy's normal ONNX Parakeet path. Fermion's own runtimes (MLX, their C
engine, CUDA) are faster, but they aren't used here.

## Rebuilding the model

You need Python 3.12 and about 6 GB of free space.

```bash
cd phonon2
python -m venv .venv && .venv/Scripts/pip install onnx onnxruntime numpy zstandard soundfile onnx-asr

# Phonon-2 (Fermion Research)
mkdir phonon2_hf && cd phonon2_hf
curl -LO https://huggingface.co/FermionResearch/Phonon-2/resolve/main/fermion_container.py
curl -LO https://huggingface.co/FermionResearch/Phonon-2/resolve/main/phonon-2.bps.tar.zst
python -c "import zstandard,tarfile;tarfile.open(fileobj=zstandard.ZstdDecompressor().stream_reader(open('phonon-2.bps.tar.zst','rb')),mode='r|').extractall('.')"
cd ..

# Parakeet v3 fp32 ONNX graph (istupakov)
mkdir base_onnx && cd base_onnx
for f in encoder-model.onnx encoder-model.onnx.data decoder_joint-model.onnx nemo128.onnx vocab.txt config.json; do
  curl -LO https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/$f
done
cd ..

.venv/Scripts/python tools/convert_phonon2.py     # -> build/phonon-2-fp32
.venv/Scripts/python tools/package_phonon2.py pack  # -> build/phonon-2-int8.tar.gz + sha256
.venv/Scripts/python tools/quick_test.py some.wav  # transcribe + timing
```

The expected sha256 of `phonon-2.bps.tar.zst` is
`98125795b6dda72f5c6eee9ba33d19815df65dcb18b50a357bf9f73c9935309e`.

If you host your own archive, update `PHONON2_URL`, `PHONON2_SHA256` and `PHONON2_SIZE_MB`
in `src-tauri/src/managers/model.rs`.

## Building the app

Same as Handy (see [BUILD.md](BUILD.md)), minus the Vulkan SDK:

```bash
bun install
bun run tauri build --config src-tauri/tauri.phonon.conf.json
```

## Limitations

- English only. Parakeet v3 handles 25 languages; Phonon-2 was trained for English.
- No Whisper or other models, on purpose.
- No auto-updates. Pull upstream changes into the fork yourself.

## Credits and licences

- [Handy](https://github.com/cjpais/Handy) by CJ Pais, MIT. This fork keeps that licence.
- [Phonon-2](https://huggingface.co/FermionResearch/Phonon-2) by Fermion Research. Weights are
  CC-BY-4.0, and `fermion_container.py` is Apache 2.0.
- [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) by NVIDIA, CC-BY-4.0.
- The ONNX graph comes from [istupakov/parakeet-tdt-0.6b-v3-onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx).

The model archive in the releases is Phonon-2 converted to ONNX and quantized to 8-bit. That
conversion is the only change made to Fermion's weights.
