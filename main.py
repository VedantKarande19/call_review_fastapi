import uuid
import httpx
import tempfile
import os
from pathlib import Path
from fastapi import FastAPI, Depends, BackgroundTasks, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import List, Optional

from database import engine, Base, get_db, SessionLocal, ensure_audio_records_schema
import models
from call_audio_pipeline import process_call_recording

Base.metadata.create_all(bind=engine)
ensure_audio_records_schema()

app = FastAPI(title="Audio Transcription API")

# ---------------------------------------------------------------------------
# Status registry
# Every valid phase lives here. Add new phases here — nowhere else.
# ---------------------------------------------------------------------------
STATUS_INFO: dict[str, dict] = {
    "pending_download": {
        "terminal": False,
        "summary": "Job created; the audio URL has not been downloaded yet.",
    },
    "downloading": {
        "terminal": False,
        "summary": "Fetching audio from the URL.",
    },
    "downloaded": {
        "terminal": False,
        "summary": "Audio is stored; AI pipeline starting.",
    },
    # kept for rows written by older code
    "uploaded": {
        "terminal": False,
        "summary": "Legacy: same as downloaded — audio stored, waiting for or running AI.",
    },
    "processing_diarization": {
        "terminal": False,
        "summary": "Speaker diarization is running (Pyannote).",
    },
    "processing_transcription": {
        "terminal": False,
        "summary": "Speech transcription / translation is running (Groq).",
    },
    "processing_alignment": {
        "terminal": False,
        "summary": "Aligning transcript with diarization.",
    },
    "completed": {
        "terminal": True,
        "summary": "Processing finished; transcript is available.",
    },
    "failed": {
        "terminal": True,
        "summary": "Processing failed; see error_detail if present.",
    },
    "failed_to_download": {
        "terminal": True,
        "summary": "Legacy: URL fetch failed.",
    },
    "failed_download": {
        "terminal": True,
        "summary": "The audio URL could not be fetched.",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_db_status(raw: str | None) -> tuple[str, str | None]:
    """
    Map legacy / free-form status strings to a known phase + optional error snippet.

    Handles:
      - None / empty          → "unknown"
      - "failed: <message>"   → ("failed", "<message>")   [your old format]
      - anything else         → returned as-is
    """
    if not raw:
        return "unknown", None
    if raw.startswith("failed:"):
        return "failed", raw[len("failed:"):].strip() or None
    return raw, None


def _status_payload(phase: str, error: str | None = None) -> dict:
    """Build the standard status sub-object that every response includes."""
    info = STATUS_INFO.get(phase, {})
    out = {
        "phase": phase,
        "terminal": info.get(
            "terminal",
            phase in ("completed", "failed", "failed_to_download", "failed_download"),
        ),
        "summary": info.get("summary", "Processing update."),
    }
    if error:
        out["error_detail"] = error
    return out


def _persist_stage(audio_id: str, stage: str) -> None:
    """
    Called by the pipeline's on_stage callback to write granular progress to the DB.
    Opens its own session so it is safe to call from any thread.
    """
    s = SessionLocal()
    try:
        r = s.query(models.AudioRecord).filter(models.AudioRecord.audio_id == audio_id).first()
        if r:
            r.status = stage
            s.commit()
    finally:
        s.close()


def _notify_job_terminal(
    notify_url: Optional[str],
    audio_id: str,
    prompt_id: str,
    phase: str,
    error_detail: Optional[str] = None,
) -> None:
    """
    Fire-and-forget POST to the caller's webhook URL.
    Failures are swallowed so a bad webhook never fails a job.
    """
    if not notify_url or not str(notify_url).strip():
        return
    url = str(notify_url).strip()
    body = {
        "event": "call_review.job_finished",
        "audio_id": audio_id,
        "prompt_id": prompt_id,
        "phase": phase,
        "results_ready": phase == "completed",
        "message": (
            "Results are ready; use GET /status/{audio_id} to fetch the transcript."
            if phase == "completed"
            else "Job finished with an error for this audio_id."
        ),
    }
    if error_detail:
        body["error_detail"] = error_detail
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            client.post(url, json=body, headers={"Content-Type": "application/json"})
    except Exception:
        pass  # webhook errors must never surface to the caller


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class AudioRequest(BaseModel):
    audio_links: List[str]
    prompt_id: str
    notify_url: Optional[str] = Field(
        default=None,
        description=(
            "If set, this URL receives a POST when each audio_id finishes "
            "(completed or failed)."
        ),
    )


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def _download_and_process(audio_id: str) -> None:
    """
    Single background task per audio_id:
      1. Download the audio (if not already stored).
      2. Write to a temp file.
      3. Run the AI pipeline — passing BOTH source_url and tmp_path so
         Pyannote's hosted API can use the URL while Groq uses the local file.
      4. Persist results and fire the webhook.

    Session handling
    ----------------
    After any db.rollback() the ORM objects are detached — we always
    re-query the record before writing to it to avoid DetachedInstanceError.
    """
    db = SessionLocal()
    tmp_path: Path | None = None

    try:
        record = (
            db.query(models.AudioRecord)
            .filter(models.AudioRecord.audio_id == audio_id)
            .first()
        )
        if not record:
            return

        # Clear any leftover error from a previous attempt
        record.error_detail = None
        db.commit()

        # ------------------------------------------------------------------
        # Step 1 — Download (skip if audio bytes are already stored)
        # ------------------------------------------------------------------
        if record.audio_file is None:
            if not record.source_url:
                record.status = "failed_download"
                record.error_detail = "No source_url on record"
                db.commit()
                _notify_job_terminal(
                    record.notify_url, audio_id, record.prompt_id,
                    "failed_download", record.error_detail,
                )
                return

            record.status = "downloading"
            db.commit()

            try:
                with httpx.Client(timeout=120.0, follow_redirects=True) as client:
                    response = client.get(record.source_url)
                    response.raise_for_status()
                    record.audio_file = response.content
                    record.status = "downloaded"
                    db.commit()

            except Exception as exc:
                db.rollback()
                # Re-query — object is detached after rollback
                record = (
                    db.query(models.AudioRecord)
                    .filter(models.AudioRecord.audio_id == audio_id)
                    .first()
                )
                if record:
                    record.status = "failed_download"
                    record.error_detail = str(exc)
                    db.commit()
                    _notify_job_terminal(
                        record.notify_url, audio_id, record.prompt_id,
                        "failed_download", record.error_detail,
                    )
                return

        db.refresh(record)
        if not record.audio_file:
            return

        # ------------------------------------------------------------------
        # Step 2 — Write audio bytes to a temp file for local processing
        # ------------------------------------------------------------------
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(record.audio_file)
            tmp_path = Path(tmp_file.name)

        # Snapshot these before the long-running pipeline so we don't hold
        # a reference to `record` across thread boundaries.
        source_url = record.source_url
        notify_url = record.notify_url
        prompt_id  = record.prompt_id

        # ------------------------------------------------------------------
        # Step 3 — AI pipeline
        # Pass BOTH the original URL (for Pyannote hosted API) and the local
        # temp path (for Groq), so the pipeline can choose what it needs.
        # ------------------------------------------------------------------
        try:
            ai_results = process_call_recording(
                audio_url=source_url,
                audio_path=tmp_path,
                on_stage=lambda st: _persist_stage(audio_id, st),
            )

            new_transcript = models.TranscriptResult(
                audio_id=audio_id,
                transcript_text=ai_results.get("text", ""),
                english_translation=ai_results.get("english_translation", ""),
                transcript_json=ai_results.get("combined_json", {}),
            )
            db.add(new_transcript)

            # Re-query before the final write (on_stage callbacks may have
            # touched the row from a different session)
            record = (
                db.query(models.AudioRecord)
                .filter(models.AudioRecord.audio_id == audio_id)
                .first()
            )
            if record:
                record.status = "completed"
            db.commit()

            _notify_job_terminal(notify_url, audio_id, prompt_id, "completed")

        except Exception as exc:
            db.rollback()
            # Re-query — object is detached after rollback
            record = (
                db.query(models.AudioRecord)
                .filter(models.AudioRecord.audio_id == audio_id)
                .first()
            )
            if record:
                record.status = "failed"
                record.error_detail = str(exc)
                db.commit()
                _notify_job_terminal(
                    notify_url, audio_id, prompt_id,
                    "failed", record.error_detail,
                )

    finally:
        db.close()
        if tmp_path is not None and tmp_path.exists():
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/process-audio/")
def process_audio(
    request: AudioRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Create one DB row (and audio_id) per link and return immediately.
    The download + AI pipeline runs in the background.

    Why synchronous + BackgroundTasks instead of async?
    ----------------------------------------------------
    Downloading inside an async handler blocks the event loop for large files.
    By keeping the POST handler synchronous and delegating all I/O to a
    background thread we stay non-blocking without needing asyncio in the
    worker itself (which would complicate SQLAlchemy usage).
    """
    results = []

    for link in request.audio_links:
        unique_audio_id = str(uuid.uuid4())

        new_record = models.AudioRecord(
            audio_id=unique_audio_id,
            prompt_id=request.prompt_id,
            source_url=link,
            audio_file=None,          # downloaded in the background
            status="pending_download",
            notify_url=request.notify_url,
        )
        db.add(new_record)
        db.commit()

        background_tasks.add_task(_download_and_process, unique_audio_id)

        results.append(
            {
                "link": link,
                "audio_id": unique_audio_id,
                **_status_payload("pending_download"),
            }
        )

    return {
        "message": (
            "IDs returned immediately; audio is fetched in the background. "
            "Poll GET /status/{audio_id}. "
            "If notify_url was set, a POST is fired there when each job finishes."
        ),
        "prompt_id": request.prompt_id,
        "notify_url": request.notify_url,
        "details": results,
    }


@app.get("/status/{audio_id}")
def check_status(audio_id: str, db: Session = Depends(get_db)):
    """
    Return the current phase and, once completed, the transcript text.

    Legacy status normalization
    ---------------------------
    Older rows may have:
      - "pending"     → mapped to "downloaded" (bytes present) or "pending_download"
      - "uploaded"    → mapped to "downloaded"
      - "processing"  → mapped to "processing_diarization"
      - "failed: …"   → split into phase="failed" + error_detail
    """
    record = (
        db.query(models.AudioRecord)
        .filter(models.AudioRecord.audio_id == audio_id)
        .first()
    )
    if not record:
        raise HTTPException(status_code=404, detail="Audio ID not found")

    phase, legacy_err = _normalize_db_status(record.status)

    # Remap legacy phase names
    if phase == "pending":
        phase = "downloaded" if record.audio_file else "pending_download"
    elif phase == "uploaded":
        phase = "downloaded"
    elif phase == "processing":
        phase = "processing_diarization"
        legacy_err = None

    err = record.error_detail or legacy_err
    err_phases = ("failed", "failed_download", "failed_to_download")

    response = {
        "audio_id": audio_id,
        "prompt_id": record.prompt_id,
        "source_url": record.source_url,
        **_status_payload(phase, err if phase in err_phases else None),
    }

    if phase == "completed":
        transcript = (
            db.query(models.TranscriptResult)
            .filter(models.TranscriptResult.audio_id == audio_id)
            .first()
        )
        if transcript:
            response["transcript"] = transcript.transcript_text

    return response
