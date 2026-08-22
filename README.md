# AI Video Localizer

Local-first MVP for translating Chinese videos into Vietnamese or English while preserving character relationships and forms of address.

## What is implemented

- FastAPI backend with project-based workflow.
- Video upload/import and `ffprobe` metadata inspection.
- ASR adapter interface with a working `mock` engine and an optional FunASR adapter hook.
- OCR adapter interface ready for PaddleOCR/VSE integration.
- Character + relationship graph stored per project.
- Translation through any OpenAI-compatible `/chat/completions` endpoint.
- Context-aware prompt builder that includes speaker, addressee, relationship, preferred pronouns, glossary, previous/next cues.
- Subtitle editing APIs and SRT import/export.
- FFmpeg rendering with burned subtitles plus optional intro/outro and image stickers.
- React/Vite control panel for the MVP workflow.
- Unit tests for SRT handling, relationship resolution, prompt construction, and render filter generation.

## Architecture

```text
frontend (React/Vite)
        |
        v
FastAPI project API
        |
        +-- media / ffprobe / ffmpeg
        +-- ASR adapter (mock -> FunASR)
        +-- OCR adapter (null -> PaddleOCR/VSE)
        +-- character relationship graph
        +-- translation engine (OpenAI-compatible)
        +-- subtitle store + SRT
        +-- renderer
```

## Quick start

### Requirements

- Python 3.11+
- Node.js 20+
- FFmpeg + ffprobe available on `PATH`

### Backend

```bash
cd backend
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

Health check: `GET http://localhost:8000/api/health`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Translation provider

The backend deliberately uses an OpenAI-compatible HTTP interface so you can point it at OpenAI, compatible gateways, LM Studio, Ollama-compatible gateways, or your own proxy.

Configure `backend/.env`:

```env
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your-key
LLM_MODEL=gpt-5-mini
```

If these values are missing, the API returns a clear configuration error instead of silently producing fake translations.

## Current MVP workflow

1. Upload a Chinese video.
2. Add/import subtitle cues or run the mock ASR sample.
3. Define characters and their relationship/pronoun rules.
4. Translate cues to Vietnamese or English.
5. Review/edit low-confidence cues.
6. Render burned subtitles, optional intro/outro and image stickers with FFmpeg.

## Why ASR/OCR are adapters

FunASR, PaddleOCR, WhisperX and SAM2 are large, GPU-sensitive dependencies. They are intentionally isolated behind adapters so the core app can run and be tested without forcing several GB of model installs. The next milestone is to wire the real engines behind the same interfaces.

## Safety / licensing notes

Before distributing a commercial closed-source build, review the licenses of every optional external engine/model you enable. FFmpeg build flags/codecs also affect redistribution obligations.

## Roadmap

- [ ] FunASR real inference service + word/segment timestamps.
- [ ] PaddleOCR/VSE hard-subtitle extraction.
- [ ] ASR + OCR confidence fusion.
- [ ] Speaker diarization and automatic addressee inference.
- [ ] Character graph extraction from whole-video transcript.
- [ ] Translation critic/QA pass.
- [ ] Hard subtitle removal/inpainting.
- [ ] Tauri desktop shell with direct local file paths.
- [ ] SAM2 background/object workflows.
- [ ] Voice dubbing pipeline.
