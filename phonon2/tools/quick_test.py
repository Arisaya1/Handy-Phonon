"""Quick Phonon-2 int8 test: transcribe each audio file given and report speed."""
import sys
import time
from pathlib import Path

import onnx_asr
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "build" / "phonon-2-int8"

t0 = time.perf_counter()
model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", str(MODEL), quantization="int8")
print(f"load: {time.perf_counter() - t0:.1f}s")

for f in sys.argv[1:] or [str(ROOT / "test_audio" / "sample.wav")]:
    dur = sf.info(f).duration
    model.recognize(f)  # warm-up
    t = time.perf_counter()
    text = model.recognize(f)
    dt = time.perf_counter() - t
    print(f"\n{Path(f).name}  ({dur:.1f}s audio, {dt:.2f}s to transcribe, {dur / dt:.0f}x realtime)")
    print(f"  {text}")
