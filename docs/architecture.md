# Architecture

The **Investigator** lives in `main_agent`. The **Inspectors** live in `sub_agents`.

`main_agent` sees up to 16 overview frames at about 1 FPS, then emits one JSON action per step (at most five steps). A `run_skill` action calls one inspector. The inspector's text observation is appended, and the investigator decides the next step. The trajectory ends with `answer: real` or `answer: fake`.

Inspectors:

| Module | Inspector | Transport |
| --- | --- | --- |
| `sub_agents/physical.py` | physical_consistency | HTTP vLLM |
| `sub_agents/appearance.py` | visual_appearance | HTTP vLLM |
| `sub_agents/common_sense.py` | common_sense | HTTP vLLM |
| `sub_agents/ocr.py` | ocr_consistency | HTTP vLLM, plus local PP-OCR |
| `sub_agents/audio.py` | voice_profile, asr_analysis | local audio tools; ASR judgment is HTTP vLLM on the base model |

The default main backend (`--main-backend engine`) keeps the investigator's token prefix by appending generated token ids. It does not paste the assistant string back into a chat template. `--main-backend http` is a convenience path and is not token-exact.

Shared decoding, frame caching, the OpenAI client, and metrics live in `core/`. `main_agent.build_agent` assembles the investigator and all inspectors from a config file.
