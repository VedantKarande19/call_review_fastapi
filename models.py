from sqlalchemy import Column, Integer, String, LargeBinary, ForeignKey, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from database import Base


class AudioRecord(Base):
    __tablename__ = "audio_records"

    id = Column(Integer, primary_key=True, index=True)
    audio_id = Column(String, unique=True, index=True, nullable=False)
    prompt_id = Column(String, index=True, nullable=False)
    source_url = Column(Text, nullable=True)
    audio_file = Column(LargeBinary, nullable=True)
    status = Column(String, default="pending_download")
    error_detail = Column(Text, nullable=True)
    notify_url = Column(Text, nullable=True)

    transcript = relationship("TranscriptResult", back_populates="audio", uselist=False)


class TranscriptResult(Base):
    __tablename__ = "transcript_results"

    id = Column(Integer, primary_key=True, index=True)
    audio_id = Column(String, ForeignKey("audio_records.audio_id"), unique=True, nullable=False)
    transcript_text = Column(Text, nullable=True)
    english_translation = Column(Text, nullable=True)
    transcript_json = Column(JSONB, nullable=True)

    audio = relationship("AudioRecord", back_populates="transcript")
