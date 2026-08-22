# AI Video Localizer

Local-first MVP for translating Chinese videos into Vietnamese or English while preserving character relationships, forms of address, subtitle timing, and editability.

## Implemented in this MVP

- FastAPI backend with JSON project persistence.
- Video import plus `ffprobe` metadata inspection.
- FFmpeg audio extraction and final subtitle/sticker rendering.
- FunASR adapter with optional VAD, punctuation and speaker diarization output.
- PaddleOCR hard-subtitle extraction by sampled/cropped frames.
- ASR + OCR temporal/confidence fusion.
- LLM context analysis that maps speaker IDs to characters, infers addressees, directional relationships and Vietnamese pronouns.
- OpenAI-compatible translation with stable cue-ID validation, glossary and neighboring-dialogue context.
- SRT import/export and editable subtitle cues.
- Optional intro/outro and image sticker render configuration.
- React/Vite editor UI for import, analyze, infer roles, translate, review and render.
- Backend unit tests and GitHub Actions CI.

## Architecture

```text
React/Vite editor
      |
      v
FastAPI project API
      |
      +-- FFmpeg / ffprobe
      +-- FunASR adapter ---------+
      +-- PaddleOCR adapter ------+--> ASR/OCR fusion
      |                                |
      |                                v
      +-- LLM context analyzer --> character/relationship graph
      |                                |
      +-- translation engine <---------+
      +-- subtitle/SRT store
      +-- FFmpeg renderer -> MP4
```

The heavy AI engines are adapters on purpose. The core project can run with `ASR_ENGINE=mock` and `OCR_ENGINE=none`, while FunASR and PaddleOCR can be installed independently for a machine's CPU/CUDA setup.

## Requirements

- Python 3.11+
- Node.js 20+
- FFmpeg + ffprobe on `PATH`

## Quick start

### 1. Backend

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

Create the environment file:

```bash
# macOS/Linux
cp .env.example .env

# Windows PowerShell
# Copy-Item .env.example .env
```

Run:

```bash
uvicorn app.main:app --reload --port 8000
```

Health check: `GET http://localhost:8000/api/health`

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Enable real Chinese ASR

Install the PyTorch/torchaudio build suitable for the machine first, then:

```bash
cd backend
pip install -r requirements-asr.txt
```

Set in `backend/.env`:

```env
ASR_ENGINE=funasr
FUNASR_MODEL=paraformer-zh
FUNASR_VAD_MODEL=fsmn-vad
FUNASR_PUNC_MODEL=ct-punc
FUNASR_SPK_MODEL=cam++
FUNASR_DEVICE=cpu
```

For CUDA, change the device according to the installed FunASR/PyTorch environment.

## Enable hard-subtitle OCR

Install the PaddlePaddle wheel suitable for the machine first, then:

```bash
cd backend
pip install -r requirements-ocr.txt
```

Set:

```env
OCR_ENGINE=paddle
OCR_FPS=2.0
OCR_CROP_TOP_RATIO=0.62
```

`OCR_CROP_TOP_RATIO` controls how much of the top of each frame is ignored. The default focuses OCR on the lower subtitle region to reduce logos and unrelated scene text.

## Translation / relationship inference provider

The backend uses an OpenAI-compatible `/chat/completions` interface so the translation layer is provider-swappable.

Configure `backend/.env`:

```env
LLM_BASE_URL=https://your-provider.example/v1
LLM_API_KEY=your-key
LLM_MODEL=your-model-name
```

The LLM is used twice for different jobs:

1. **Infer roles**: analyze the transcript, keep existing diarized speaker IDs, infer characters/addressees/relationships and Vietnamese forms of address.
2. **Translate**: translate immutable subtitle cue IDs with speaker, addressee, relationship, preferred pronouns, glossary and neighboring dialogue as context.

A translation batch is rejected when the provider drops or changes cue IDs, protecting subtitle timing from accidental LLM merging/splitting.

## Current workflow

```text
Video
  -> FFmpeg audio extraction
  -> FunASR -------------------+
  -> PaddleOCR hard subtitles -+-> fusion
                                  -> infer roles/relationships
                                  -> VI/EN context translation
                                  -> review/edit cues
                                  -> FFmpeg render
                                  -> MP4
```

The default `.env.example` keeps heavy engines disabled (`ASR_ENGINE=mock`, `OCR_ENGINE=none`) so a fresh clone can start without downloading AI models. Enable the real adapters when the machine is ready.

## Tests

```bash
cd backend
pytest -q
```

Frontend production build:

```bash
cd frontend
npm install
npm run build
```

## Docker

Create `backend/.env`, then:

```bash
docker compose up --build
```

The base backend image contains FFmpeg and the core API dependencies. GPU-specific FunASR/PaddleOCR dependencies are intentionally not baked into the base image yet.

## Known MVP limitations

- Automatic relationship inference depends on the configured LLM and is reviewable rather than treated as ground truth.
- PaddleOCR currently uses a configurable bottom-frame crop rather than a learned subtitle-region detector.
- Intro/outro inputs should already have compatible resolution/FPS/audio layout; normalization is a follow-up.
- Hard Chinese subtitle removal/inpainting is not implemented yet.
- Background/object editing is not implemented yet.
- Voice dubbing is not implemented yet.

## Next milestones

- [x] FunASR adapter and diarized cue support.
- [x] PaddleOCR hard-subtitle extraction.
- [x] ASR + OCR fusion.
- [x] Character/addressee/relationship inference pass.
- [x] Relationship-aware VI/EN translation.
- [x] Subtitle editor and FFmpeg render path.
- [ ] Translation critic / QA retry pass.
- [ ] Better subtitle-region detection + OCR review UI.
- [ ] Hard subtitle removal/inpainting.
- [ ] Timeline controls for intro/outro/stickers in the UI.
- [ ] SAM2 background/object workflows.
- [ ] Tauri desktop shell with direct local file paths.
- [ ] Voice dubbing pipeline.

## Licensing note

Review the licenses of enabled models, external engines and the exact FFmpeg build/codecs before distributing a commercial binary. Optional integrations are kept isolated to make those decisions replaceable.
