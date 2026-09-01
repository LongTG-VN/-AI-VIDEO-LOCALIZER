# AI Video Localizer

Local-first MVP for translating Chinese videos into Vietnamese or English while preserving character relationships, forms of address, subtitle timing, and editability.

The current branch also includes an end-to-end Vietnamese publishable render path: ASR/OCR fusion, relationship-aware translation, utterance-aware subtitles, Chinese hard-sub cleanup, ASS burn-in, and NVENC export.

## Implemented

- FastAPI backend with JSON project persistence.
- Video import plus `ffprobe` metadata inspection.
- FFmpeg audio extraction.
- FunASR adapter with optional VAD, punctuation and speaker diarization output.
- PaddleOCR hard-subtitle extraction with subtitle ROI, frame-change skipping, multiline merging and noise filtering.
- ASR + OCR temporal/confidence fusion.
- Preserved OCR visual timing (`ocr_start` / `ocr_end`) separate from ASR dialogue timing for safe hard-sub cleanup.
- LLM context analysis that maps speaker IDs to characters, infers addressees, directional relationships and Vietnamese pronouns.
- OpenAI-compatible translation with stable cue-ID validation, glossary and neighboring-dialogue context.
- Translation critic / targeted retry support.
- Utterance-aware render cues to suppress duplicate fragments, merge sentence continuations, calculate CPS and prevent stacked subtitles.
- SRT import/export and ASS generation.
- Chinese hard-sub removal with conservative text masks, OCR-gated timing, temporal donor reconstruction and Telea fallback.
- Lossless FFV1 cleaned intermediate before final encode.
- NVENC H.264 final export with original audio preserved.
- Optional intro/outro and image sticker render configuration.
- React/Vite editor UI for import, analyze, infer roles, translate, review and render.
- Backend unit/regression tests and GitHub Actions CI.

## Architecture

```text
React/Vite editor
      |
      v
FastAPI project API
      |
      +-- FFmpeg / ffprobe
      +-- FunASR ------------------+
      +-- PaddleOCR ---------------+--> ASR/OCR fusion
      |                                  |
      |                                  +--> dialogue timing
      |                                  +--> visual OCR timing
      |                                  |
      |                                  v
      +-- LLM context analyzer --> character/relationship graph
      |                                  |
      +-- translation + critic <----------+
      |                                  |
      +-- utterance subtitle engine ------+
      |                                  |
      +-- hard-sub cleaner <--- OCR visual timing
      |                                  |
      +-- ASS + NVENC renderer ---------> MP4
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

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create the environment file:

```bash
# Windows PowerShell
Copy-Item .env.example .env

# macOS/Linux
# cp .env.example .env
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

Install a matching PyTorch + torchaudio CUDA/CPU build for the machine first, then:

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
FUNASR_DEVICE=cuda
```

Use `FUNASR_DEVICE=cpu` when CUDA is unavailable.

## Enable hard-subtitle OCR

Install the PaddlePaddle runtime suitable for the machine first, then:

```bash
cd backend
pip install -r requirements-ocr.txt
```

Set:

```env
OCR_ENGINE=paddle
OCR_FPS=2.0
OCR_CROP_TOP_RATIO=0.65
OCR_CROP_BOTTOM_RATIO=0.95
OCR_CROP_LEFT_RATIO=0.06
OCR_CROP_RIGHT_RATIO=0.94
OCR_CHANGE_DIFF_THRESHOLD=16.0
```

The optimized OCR path samples the subtitle ROI, skips visually unchanged frames, merges multiline subtitle text and stores visual OCR timing independently from spoken dialogue timing.

## Important after upgrading from an older project

Hard-sub quality mode now relies on `ocr_start` / `ocr_end` stored during Analyze. Projects analyzed before this change may only have ASR-backed cue timestamps.

For the cleanest render after pulling this branch, run **Analyze again** on the source video before Context / Translate / Render. This regenerates precise visual hard-sub timing and avoids stale cleanup after the Chinese subtitle disappears.

## Translation / relationship inference provider

The backend uses an OpenAI-compatible `/chat/completions` interface.

Configure `backend/.env`:

```env
LLM_BASE_URL=https://your-provider.example/v1
LLM_API_KEY=your-key
LLM_MODEL=your-model-name
```

The LLM is used for:

1. **Infer roles** — characters, addressees, directional relationships and Vietnamese forms of address.
2. **Translate** — stable cue-ID translation with speaker/addressee/relationship/glossary context.
3. **Critic** — review translation consistency and retry only cues that need correction.

## Current end-to-end workflow

```text
Video
  -> FFmpeg audio extraction
  -> FunASR ----------------------+ 
  -> PaddleOCR hard subtitles ----+-> fusion
                                     -> preserve OCR visual timing
                                     -> infer roles/relationships
                                     -> context-aware VI/EN translation
                                     -> translation critic
                                     -> utterance-aware subtitle render cues
                                     -> Chinese hard-sub cleanup
                                     -> ASS burn-in
                                     -> NVENC/libx264 render
                                     -> MP4
```

## Hard-sub cleanup modes

`RenderOptions.hardsub_removal_mode` supports:

- `none` — do not remove Chinese hard subtitles.
- `inpaint` — fast spatial Telea inpainting.
- `quality` — use guarded temporal donor reconstruction with Telea fallback.
- `auto` — current default; selects the quality path.
- `cover` — dark subtitle-band fallback.

Quality mode also:

- uses OCR visual timing instead of ASR timing when available;
- rejects large bright background regions;
- keeps a tight subtitle-shaped mask;
- refuses temporal donors across detected active subtitle ranges;
- rejects donors when local motion around the subtitle area is too high;
- refines the mask against the aligned clean donor;
- writes a lossless FFV1 intermediate before the final H.264 encode.

The final visual result should still be reviewed on difficult moving/texture-heavy scenes.

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

## Current limitations

- Automatic relationship inference remains reviewable rather than absolute ground truth.
- OCR subtitle ROI is configurable rather than learned automatically.
- Hard-sub removal is much safer than the original Telea-only version, but moving texture behind text can still require visual QA.
- Intro/outro assets should have compatible resolution/FPS/audio layout.
- Voice dubbing is not implemented yet.
- Long-video checkpoint/resume mode for 2–3 hour videos is not implemented yet.
- Background/object editing and SAM2 workflows are not implemented yet.

## Milestones

- [x] FunASR + speaker diarization.
- [x] PaddleOCR extraction + frame skipping.
- [x] ASR/OCR fusion.
- [x] Directional Character Graph and Vietnamese pronouns.
- [x] Context-aware translation.
- [x] Translation critic and regression QA.
- [x] Utterance-aware subtitle rendering.
- [x] ASS subtitle styling and NVENC export.
- [x] Hard-sub cleanup MVP.
- [x] Hard-sub temporal quality pipeline.
- [ ] Visual QA / tuning of the temporal hard-sub cleaner on the golden 90s video.
- [ ] Vietnamese dubbing with character voice mapping and music/SFX preservation.
- [ ] Long-video chunking + checkpoint/resume + global character memory.
- [ ] Timeline controls for intro/outro/stickers.
- [ ] SAM2 background/object workflows.
- [ ] Tauri desktop shell.

## Licensing note

Review the licenses of enabled models, external engines and the exact FFmpeg build/codecs before distributing a commercial binary. Optional integrations are kept isolated to make those decisions replaceable.
