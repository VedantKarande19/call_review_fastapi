# from sqlalchemy import Column, Integer, String, LargeBinary
# from database import Base

# class AudioRecord(Base):
#     __tablename__ = "audio_records"

#     id = Column(Integer, primary_key=True, index=True)
#     audio_id = Column(String, unique=True, index=True, nullable=False)
#     prompt_id = Column(String, index=True, nullable=False)
#     audio_file = Column(LargeBinary, nullable=False) # Stores the actual file bytes



from sqlalchemy import Column, Integer, String, LargeBinary, ForeignKey, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from database import Base

class AudioRecord(Base):
    __tablename__ = "audio_records"

    id = Column(Integer, primary_key=True, index=True)
    audio_id = Column(String, unique=True, index=True, nullable=False)
    prompt_id = Column(String, index=True, nullable=False)
    audio_file = Column(LargeBinary, nullable=False)
    status = Column(String, default="pending") 

    # Link to the transcript table
    transcript = relationship("TranscriptResult", back_populates="audio", uselist=False)


class TranscriptResult(Base):
    __tablename__ = "transcript_results"

    id = Column(Integer, primary_key=True, index=True)
    audio_id = Column(String, ForeignKey("audio_records.audio_id"), unique=True, nullable=False)
    
    transcript_text = Column(Text, nullable=True)     # Clean Agent/Customer text
    english_translation = Column(Text, nullable=True) # Raw translation text
    transcript_json = Column(JSONB, nullable=True)    # The complete JSON response

    # Link back to the audio record
    audio = relationship("AudioRecord", back_populates="transcript")