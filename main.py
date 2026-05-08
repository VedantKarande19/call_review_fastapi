import uuid
import httpx
import tempfile
import os
from pathlib import Path
from fastapi import FastAPI, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List

from database import engine, Base, get_db, SessionLocal
import models

# Import the orchestrator from our modified pipeline module
from call_audio_pipeline import process_call_recording 

# Create the database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Audio Transcription API")

class AudioRequest(BaseModel):
    audio_links: List[str]
    prompt_id: str

def process_audio_background(audio_id: str):
    db = SessionLocal()
    try:
        # Fetch the pending audio record
        record = db.query(models.AudioRecord).filter(models.AudioRecord.audio_id == audio_id).first()
        if not record: return
            
        record.status = "processing"
        db.commit()

        # Write BYTEA to temp file for pipeline.py
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(record.audio_file)
            tmp_path = Path(tmp_file.name)

        try:
            # Pass the temp file path to your custom pipeline
            ai_results = process_call_recording(tmp_path)
            
            # Save results to the new TranscriptResult table
            new_transcript = models.TranscriptResult(
                audio_id=audio_id,
                transcript_text=ai_results.get("text", ""),
                english_translation=ai_results.get("english_translation", ""),
                transcript_json=ai_results.get("combined_json", {})
            )
            db.add(new_transcript)
            
            record.status = "completed"
            db.commit()
            
        except Exception as e:
            db.rollback()
            record.status = f"failed: {str(e)}"
            db.commit()
        finally:
            if tmp_path.exists():
                os.remove(tmp_path)
                
    finally:
        db.close()

@app.post("/process-audio/")
async def process_audio(request: AudioRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    results = []
    
    async with httpx.AsyncClient() as client:
        for link in request.audio_links:
            try:
                response = await client.get(link, timeout=30.0)
                response.raise_for_status()
                
                unique_audio_id = str(uuid.uuid4())
                
                new_record = models.AudioRecord(
                    audio_id=unique_audio_id,
                    prompt_id=request.prompt_id,
                    audio_file=response.content,
                    status="pending"
                )
                db.add(new_record)
                db.commit()
                
                background_tasks.add_task(process_audio_background, unique_audio_id)
                
                results.append({
                    "link": link,
                    "status": "queued",
                    "audio_id": unique_audio_id
                })
                
            except Exception as e:
                db.rollback()
                results.append({
                    "link": link,
                    "status": "failed_to_download",
                    "error": str(e)
                })

    return {
        "message": "Files downloaded and queued for AI pipeline.",
        "prompt_id": request.prompt_id,
        "details": results
    }

from fastapi import HTTPException

# Add this route below your POST route in main.py
@app.get("/status/{audio_id}")
def check_status(audio_id: str, db: Session = Depends(get_db)):
    # 1. Check the raw audio table for the current status
    record = db.query(models.AudioRecord).filter(models.AudioRecord.audio_id == audio_id).first()
    
    if not record:
        raise HTTPException(status_code=404, detail="Audio ID not found")

    response_data = {
        "audio_id": audio_id,
        "status": record.status,
    }

    # 2. If it's done, fetch the results from the transcript table!
    if record.status == "completed":
        transcript = db.query(models.TranscriptResult).filter(models.TranscriptResult.audio_id == audio_id).first()
        if transcript:
            response_data["transcript"] = transcript.transcript_text
            # You can also return transcript.transcript_json here if you want to see the raw data in Swagger

    return response_data