import os
from pathlib import Path
from typing import Any, Callable
from groq import APIStatusError, Groq

# Import the alignment function from our module
from aligned_transcript import align_combined_data

def _verbose_json_to_words(tr: Any) -> list[dict[str, Any]]:
    if hasattr(tr, "model_dump"): raw = tr.model_dump()
    elif isinstance(tr, dict): raw = tr
    else: raw = dict(tr) if hasattr(tr, "keys") else {}
    
    words = raw.get("words") or []
    out: list[dict[str, Any]] = []
    for w in words:
        if isinstance(w, dict):
            t = str(w.get("word") or w.get("text") or "").strip()
            if t: out.append({"text": t, "start": float(w["start"]), "end": float(w["end"])})
        else:
            t = str(getattr(w, "word", None) or getattr(w, "text", "") or "").strip()
            if t: out.append({"text": t, "start": float(getattr(w, "start")), "end": float(getattr(w, "end"))})
    return out

def _segments_to_pseudo_words(tr: Any) -> list[dict[str, Any]]:
    if hasattr(tr, "model_dump"): raw = tr.model_dump()
    elif isinstance(tr, dict): raw = tr
    else: raw = {}
    segments = raw.get("segments") or []
    out: list[dict[str, Any]] = []
    for seg in segments:
        if isinstance(seg, dict):
            t = str(seg.get("text") or "").strip()
            if t: out.append({"text": t, "start": float(seg["start"]), "end": float(seg["end"])})
    return out

def run_diarization(pyannote_key: str, audio_url: str, audio_path: Path, num_speakers: int = 2) -> list[dict[str, Any]]:
    from pyannoteai.sdk import Client
    client = Client(pyannote_key)
    try:
        # ATTEMPT 1: The Fast Route (URL)
        print(f"[Pyannote] Attempting to diarize using direct URL...")
        job_id = client.diarize(audio_url, model="community-1", num_speakers=num_speakers)
    except Exception as e:
        # ATTEMPT 2: The Fallback Route (Local File Upload)
        print(f"[Pyannote] URL access failed ({str(e)}). Falling back to local file upload...")
        media_url = client.upload(audio_path)
        job_id = client.diarize(media_url, model="community-1", num_speakers=num_speakers)

    diarization = client.retrieve(job_id)
    return diarization["output"]["diarization"]

def run_groq_translate_words(groq_key: str, audio_path: Path) -> tuple[list[dict[str, Any]], str]:
    client = Groq(api_key=groq_key)
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
        
    file_arg = (audio_path.name, audio_bytes)
    base_kw: dict[str, Any] = {
        "file": file_arg,
        "model": "whisper-large-v3",
        "response_format": "verbose_json",
    }
    try:
        tr = client.audio.translations.create(
            **base_kw,
            extra_body={"timestamp_granularities": ["word", "segment"]},
        )
    except APIStatusError as e:
        if getattr(e, "status_code", None) not in (400, 422): raise
        tr = client.audio.translations.create(**base_kw)
        
    words = _verbose_json_to_words(tr)
    if not words: words = _segments_to_pseudo_words(tr)
    
    text = getattr(tr, "text", None) or ""
    if not text and hasattr(tr, "model_dump"):
        text = str(tr.model_dump().get("text") or "")
        
    return words, text.strip()

def process_call_recording(
    audio_url: str,
    audio_path: Path,
    num_speakers: int = 2,
    merge_gap: float = 0.9,
    agent_speaker: str | None = None,
    strict_two: bool = False,
    on_stage: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    pyannote_key = os.getenv("PYANNOTE_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    if not pyannote_key: raise ValueError("Set PYANNOTE_API_KEY in .env")
    if not groq_key: raise ValueError("Set GROQ_API_KEY in .env")

    if on_stage:
        on_stage("processing_diarization")
    diar = run_diarization(pyannote_key, audio_url, audio_path, num_speakers=num_speakers)

    if on_stage:
        on_stage("processing_transcription")
    words, english = run_groq_translate_words(groq_key, audio_path)

    combined: dict[str, Any] = {
        "diarization": diar,
        "wordLevelTranscription": words,
    }

    if on_stage:
        on_stage("processing_alignment")
    aligned = align_combined_data(combined, merge_gap=merge_gap, agent_speaker=agent_speaker, strict_two=strict_two)
    
    return {
        **aligned,
        "english_translation": english,
        "combined_json": combined,
        "diarization": diar,
    }
