# Checkpoints

The released weights are LoRA adapters on `Qwen/Qwen3.5-4B`. Merge them with `scripts/merge_lora.py`, serve the inspectors with `scripts/serve_full.sh`, and run the investigator with the default engine backend. See `docs/deployment.md`.

The main investigator adapter is at the root of the release. The inspector adapters are in `physical_consistency/`, `visual_appearance/`, `common_sense/`, `ocr_consistency/`.

`asr_analysis` uses the unmodified base model as its judge. `voice_profile` uses the audio models named in `configs/multi_server.json` (pyannote and audEERING). The ASR acoustic model is `Systran/faster-whisper-small`.

```bash
hf download Qwen/Qwen3.5-4B --local-dir weights/Qwen3.5-4B
```
