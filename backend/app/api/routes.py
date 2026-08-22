from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

from app.models.project import DubbingMetrics, DubbingOptions, Project, ProjectPatch, RenderOptions
from app.services.asr.factory import create_asr_engine
from app.services.context_analyzer import ContextAnalyzer, ContextAnalysisError
from app.services.dubbing import DubbingService
from app.services.fusion import fuse_cues
from app.services.media import MediaError, MediaService
from app.services.ocr.factory import create_ocr_engine
from app.services.ocr.visual_tracker import VisualBoundaryTracker
from app.services.renderer import RenderError, Renderer
from app.services.store import ProjectStore
from app.services.subtitles import parse_srt, to_ass, to_srt
from app.services.translation import OpenAICompatibleTranslator, TranslationError

router = APIRouter(prefix="/api")
settings = get_settings()
store = ProjectStore(settings.data_dir / "projects")
media = MediaService(settings.ffprobe_bin, settings.ffmpeg_bin)


def _create_project_ocr_engine():
    return create_ocr_engine(
        settings.ocr_engine,
        ffmpeg_bin=settings.ffmpeg_bin,
        fps=settings.ocr_fps,
        crop_top_ratio=settings.ocr_crop_top_ratio,
        crop_bottom_ratio=settings.ocr_crop_bottom_ratio,
        crop_left_ratio=settings.ocr_crop_left_ratio,
        crop_right_ratio=settings.ocr_crop_right_ratio,
        change_threshold=settings.ocr_change_diff_threshold,
    )


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "app": settings.app_name}


@router.get("/projects", response_model=list[Project])
def list_projects() -> list[Project]:
    return store.list()


@router.get("/projects/{project_id}", response_model=Project)
def get_project(project_id: str) -> Project:
    try:
        return store.get(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc


@router.post("/projects/import", response_model=Project)
def import_video(
    file: UploadFile = File(...),
    name: str | None = Form(default=None),
    target_language: str = Form(default="vi"),
) -> Project:
    if target_language not in {"vi", "en"}:
        raise HTTPException(status_code=400, detail="target_language must be vi or en")
    project_id = str(uuid4())
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    upload_path = settings.data_dir / "uploads" / f"{project_id}{suffix}"
    with upload_path.open("wb") as destination:
        shutil.copyfileobj(file.file, destination)

    try:
        metadata = media.probe(upload_path)
    except MediaError as exc:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    project = Project(
        id=project_id,
        name=name or Path(file.filename or "Untitled").stem,
        source_video_path=str(upload_path.resolve()),
        target_language=target_language,
        duration=metadata["duration"],
        width=metadata["width"],
        height=metadata["height"],
    )
    return store.create(project)


@router.patch("/projects/{project_id}", response_model=Project)
def patch_project(project_id: str, patch: ProjectPatch) -> Project:
    try:
        return store.patch(project_id, patch)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc


@router.post("/projects/{project_id}/transcribe", response_model=Project)
def transcribe_project(project_id: str) -> Project:
    try:
        project = store.get(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc

    audio_path = settings.data_dir / "uploads" / f"{project.id}.asr.wav"
    try:
        media.extract_audio(project.source_video_path, audio_path)
        engine = create_asr_engine(
            settings.asr_engine,
            model_name=settings.funasr_model,
            vad_model=settings.funasr_vad_model,
            punc_model=settings.funasr_punc_model,
            spk_model=settings.funasr_spk_model,
            device=settings.funasr_device,
        )
        project.cues = engine.transcribe(audio_path, project.source_language)
    except (MediaError, RuntimeError, NotImplementedError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return store.save(project)


@router.post("/projects/{project_id}/ocr", response_model=Project)
def ocr_project(project_id: str) -> Project:
    try:
        project = store.get(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    try:
        engine = _create_project_ocr_engine()
        project.cues = engine.extract_subtitles(Path(project.source_video_path))
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return store.save(project)


@router.post("/projects/{project_id}/analyze", response_model=Project)
def analyze_project(project_id: str) -> Project:
    try:
        project = store.get(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc

    audio_path = settings.data_dir / "uploads" / f"{project.id}.asr.wav"
    try:
        media.extract_audio(project.source_video_path, audio_path)
        asr_engine = create_asr_engine(
            settings.asr_engine,
            model_name=settings.funasr_model,
            vad_model=settings.funasr_vad_model,
            punc_model=settings.funasr_punc_model,
            spk_model=settings.funasr_spk_model,
            device=settings.funasr_device,
        )
        ocr_engine = _create_project_ocr_engine()
        asr_cues = asr_engine.transcribe(audio_path, project.source_language)
        ocr_cues = ocr_engine.extract_subtitles(Path(project.source_video_path))
        if ocr_cues:
            tracker = VisualBoundaryTracker()
            ocr_cues = tracker.refine_cues(Path(project.source_video_path), ocr_cues)
        project.cues = fuse_cues(asr_cues, ocr_cues) if ocr_cues else asr_cues
    except (MediaError, RuntimeError, NotImplementedError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return store.save(project)


@router.post("/projects/{project_id}/subtitles/import-srt", response_model=Project)
def import_srt(project_id: str, file: UploadFile = File(...)) -> Project:
    try:
        project = store.get(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    content = file.file.read().decode("utf-8-sig", errors="replace")
    try:
        project.cues = parse_srt(content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return store.save(project)


@router.get("/projects/{project_id}/subtitles.srt")
def export_srt(project_id: str) -> PlainTextResponse:
    try:
        project = store.get(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    return PlainTextResponse(
        to_srt(project.cues, translated=True),
        media_type="application/x-subrip; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{project.name}.srt"'},
    )


@router.get("/projects/{project_id}/subtitles.ass")
def export_ass(project_id: str) -> PlainTextResponse:
    try:
        project = store.get(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    return PlainTextResponse(
        to_ass(
            project.cues,
            RenderOptions(),
            width=project.width or 852,
            height=project.height or 480,
            translated=True,
        ),
        media_type="text/x-ssa; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{project.name}.ass"'},
    )


@router.post("/projects/{project_id}/context", response_model=Project)
def analyze_context(project_id: str) -> Project:
    try:
        project = store.get(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    analyzer = ContextAnalyzer(
        settings.llm_base_url, settings.llm_api_key, settings.llm_model
    )
    try:
        project = analyzer.analyze(project)
    except ContextAnalysisError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return store.save(project)


@router.post("/projects/{project_id}/translate", response_model=Project)
def translate_project(project_id: str) -> Project:
    try:
        project = store.get(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    if not project.cues:
        raise HTTPException(status_code=400, detail="Project has no subtitle cues")

    translator = OpenAICompatibleTranslator(
        settings.llm_base_url, settings.llm_api_key, settings.llm_model
    )
    try:
        project.cues = translator.translate_project(project)
    except TranslationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return store.save(project)


@router.post("/projects/{project_id}/render")
def render_project(project_id: str, options: RenderOptions) -> dict:
    try:
        project = store.get(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    if not project.cues:
        raise HTTPException(status_code=400, detail="Project has no subtitle cues")

    output = settings.data_dir / "renders" / f"{project.id}.mp4"
    try:
        renderer = Renderer(settings.ffmpeg_bin, settings.ffprobe_bin)
        renderer.render(project, output, options)
    except RenderError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "project_id": project.id,
        "output": str(output.resolve()),
        "download_url": f"/api/renders/{project.id}.mp4",
        "render_metrics": renderer.last_render_metrics,
    }


@router.post("/projects/{project_id}/dub")
async def dub_project_endpoint(project_id: str, options: DubbingOptions | None = None) -> dict:
    try:
        project = store.get(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    if not project.cues:
        raise HTTPException(status_code=400, detail="Project has no subtitle cues")

    source_path = Path(project.source_video_path)
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Source video file not found")

    output = settings.data_dir / "renders" / f"{project.id}_dubbed.mp4"
    if options is None:
        options = DubbingOptions()

    dubber = DubbingService(ffmpeg_bin=settings.ffmpeg_bin)
    try:
        out_path, metrics = await dubber.dub_project(
            project=project,
            source_video_path=source_path,
            output_video_path=output,
            options=options,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Dubbing failed: {exc}") from exc

    return {
        "project_id": project.id,
        "output": str(out_path.resolve()),
        "download_url": f"/api/renders/{out_path.name}",
        "dubbing_metrics": metrics.model_dump(),
    }


@router.get("/renders/{filename}")
def download_render(filename: str) -> FileResponse:
    safe_name = Path(filename).name
    path = settings.data_dir / "renders" / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Render not found")
    return FileResponse(path, media_type="video/mp4", filename=safe_name)
