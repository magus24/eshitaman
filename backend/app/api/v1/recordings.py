from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, File, Form, UploadFile, HTTPException, status
from app.schemas.recording import RecordingStatus, RecordingUploadResponse
from app.services.storage import storage_service
from app.services.validation import (
    validate_recording_id,
    validate_file_size,
    validate_audio_type,
)

router = APIRouter(prefix="/recordings", tags=["recordings"])


@router.post(
    "",
    response_model=RecordingUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a recorded audio file",
)
async def upload_recording(
    file: UploadFile = File(...),
    recording_id: str = Form(...),
    duration_seconds: Optional[float] = Form(None),
    duration_millis: Optional[int] = Form(None),
    mime_type: Optional[str] = Form(None),
    codec: Optional[str] = Form(None),
    sample_rate: Optional[int] = Form(None),
    bit_rate: Optional[int] = Form(None),
    channel_count: Optional[int] = Form(None),
    metadata_json: Optional[str] = Form(None),
):
    # 1. Validate ID
    clean_id = validate_recording_id(recording_id)

    # 2. Check for duplicate ID
    if await storage_service.recording_exists(clean_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Recording with ID '{clean_id}' already exists",
        )

    # 3. Read content and validate size
    content = await file.read()
    validate_file_size(len(content))

    # 4. Validate filename & MIME
    raw_content_type = mime_type or file.content_type
    ext, resolved_mime = validate_audio_type(file.filename, raw_content_type)

    # 5. Parse duration (seconds or millis)
    calc_duration: Optional[float] = None
    if duration_seconds is not None:
        calc_duration = float(duration_seconds)
    elif duration_millis is not None:
        calc_duration = round(duration_millis / 1000.0, 3)

    # 6. Parse any extra metadata from optional metadata_json
    extra_metadata = {}
    if metadata_json:
        try:
            extra_metadata = json.loads(metadata_json)
        except Exception:
            pass

    resolved_codec = codec or extra_metadata.get("codec") or ("opus" if "webm" in resolved_mime else "aac_lc")
    created_at = datetime.now(timezone.utc)

    metadata_dict = {
        "recording_id": clean_id,
        "filename": f"audio{ext}",
        "file_size": len(content),
        "file_size_bytes": len(content),
        "duration_seconds": calc_duration or extra_metadata.get("durationSeconds"),
        "duration_millis": duration_millis or extra_metadata.get("durationMilliseconds") or extra_metadata.get("durationMillis"),
        "mime_type": resolved_mime,
        "codec": resolved_codec,
        "sample_rate": sample_rate or extra_metadata.get("sampleRate", 44100),
        "bit_rate": bit_rate or extra_metadata.get("bitRate", 128000),
        "channel_count": channel_count or extra_metadata.get("channelCount", 1),
        "status": RecordingStatus.UPLOADED.value,
        "created_at": created_at.isoformat(),
    }

    # 7. Persist to storage
    await storage_service.save_recording(
        recording_id=clean_id,
        audio_content=content,
        filename=file.filename or f"audio{ext}",
        metadata=metadata_dict,
    )

    return RecordingUploadResponse(
        recording_id=clean_id,
        status=RecordingStatus.UPLOADED,
        file_size=len(content),
        duration=calc_duration or metadata_dict.get("duration_seconds"),
        mime_type=resolved_mime,
        created_at=created_at,
        message="Audio recording uploaded successfully",
    )


from app.schemas.processing import ProcessingResponse
from app.services.audio import (
    AudioProcessingPipeline,
    AudioProcessingError,
    FFmpegNotFoundError,
)

def get_pipeline() -> AudioProcessingPipeline:
    return AudioProcessingPipeline(storage=storage_service)


@router.post(
    "/{recording_id}/process",
    response_model=ProcessingResponse,
    status_code=status.HTTP_200_OK,
    summary="Process recording for STT preparation (VAD, normalization, chunking)",
)
async def process_recording(recording_id: str):
    clean_id = validate_recording_id(recording_id)

    meta = await storage_service.get_recording_metadata(clean_id)
    if not meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recording with ID '{clean_id}' not found",
        )

    pipeline = get_pipeline()


    try:
        result = await pipeline.process_recording(clean_id)
        msg = "Recording already processed" if meta.get("status") == RecordingStatus.PROCESSED.value else "Audio processed successfully"
        return ProcessingResponse(
            recording_id=result.recording_id,
            status=result.status,
            duration_seconds=result.audio_info.duration_seconds,
            chunks_count=result.chunks_count,
            speech_segments_count=result.speech_segments_count,
            message=msg,
        )
    except FFmpegNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": e.error_code,
                "message": e.message,
            },
        )
    except AudioProcessingError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error_code": e.error_code,
                "message": e.message,
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "INTERNAL_PROCESSING_ERROR",
                "message": str(e),
            },
        )


from app.services.stt import (
    TranscriptionPipeline,
    TranscribeRequest,
    TranscriptionResponse,
    Transcript,
    STTError,
    STTAuthenticationError,
    STTRateLimitError,
)


def get_transcription_pipeline() -> TranscriptionPipeline:
    return TranscriptionPipeline(storage=storage_service)


@router.post(
    "/{recording_id}/transcribe",
    response_model=TranscriptionResponse,
    status_code=status.HTTP_200_OK,
    summary="Transcribe processed audio recording chunks with multilingual speech recognition",
)
async def transcribe_recording(
    recording_id: str,
    request: Optional[TranscribeRequest] = None,
    language_hint: Optional[str] = None,
):
    clean_id = validate_recording_id(recording_id)

    meta = await storage_service.get_recording_metadata(clean_id)
    if not meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recording with ID '{clean_id}' not found",
        )

    pipeline = get_transcription_pipeline()
    effective_language_hint = (request.language_hint if request and request.language_hint else None) or language_hint

    try:
        transcript = await pipeline.transcribe_recording(
            recording_id=clean_id,
            language_hint=effective_language_hint,
        )
        msg = "Recording already transcribed" if meta.get("status") == RecordingStatus.TRANSCRIBED.value else "Audio transcribed successfully"
        return TranscriptionResponse(
            recording_id=transcript.recording_id,
            status=RecordingStatus.TRANSCRIBED.value,
            language=transcript.language,
            segments_count=len(transcript.segments),
            duration_ms=transcript.duration_ms,
            message=msg,
        )
    except STTAuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error_code": e.error_code,
                "message": e.message,
            },
        )
    except STTRateLimitError as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error_code": e.error_code,
                "message": e.message,
            },
        )
    except STTError as e:
        err_code = e.error_code
        status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
        if err_code == "RECORDING_NOT_FOUND":
            status_code = status.HTTP_404_NOT_FOUND
        elif err_code == "RECORDING_NOT_PROCESSED":
            status_code = status.HTTP_400_BAD_REQUEST

        raise HTTPException(
            status_code=status_code,
            detail={
                "error_code": err_code,
                "message": e.message,
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "INTERNAL_TRANSCRIPTION_ERROR",
                "message": str(e),
            },
        )


@router.get(
    "/{recording_id}/transcript",
    response_model=Transcript,
    summary="Retrieve the full timestamped transcript of a recording",
)
async def get_recording_transcript(recording_id: str):
    clean_id = validate_recording_id(recording_id)

    meta = await storage_service.get_recording_metadata(clean_id)
    if not meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recording with ID '{clean_id}' not found",
        )

    pipeline = get_transcription_pipeline()
    transcript = await pipeline.get_transcript(clean_id)
    if not transcript:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transcript for recording '{clean_id}' not found or not yet generated",
        )

    return transcript


from app.services.llm import (
    AIUnderstandingPipeline,
    AIAnalysis,
    AnalysisResponse,
    LLMError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    """Optional contextual instructions for the AI analysis."""
    context_instructions: Optional[str] = None


def get_understanding_pipeline() -> AIUnderstandingPipeline:
    return AIUnderstandingPipeline(storage=storage_service)


@router.post(
    "/{recording_id}/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate structured educational AI understanding from transcript",
)
async def analyze_recording(
    recording_id: str,
    request: Optional[AnalyzeRequest] = None,
):
    clean_id = validate_recording_id(recording_id)

    meta = await storage_service.get_recording_metadata(clean_id)
    if not meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recording with ID '{clean_id}' not found",
        )

    pipeline = get_understanding_pipeline()
    instructions = request.context_instructions if request else None

    try:
        analysis = await pipeline.analyze_recording(
            recording_id=clean_id,
            context_instructions=instructions,
        )
        msg = (
            "Recording already analyzed"
            if meta.get("status") == RecordingStatus.ANALYZED.value
            else "Analysis generated successfully"
        )
        return AnalysisResponse(
            recording_id=analysis.recording_id,
            status=RecordingStatus.ANALYZED.value,
            language=analysis.language,
            summary=analysis.summary,
            key_points_count=len(analysis.key_points),
            terms_count=len(analysis.terms),
            questions_count=len(analysis.questions),
            message=msg,
        )
    except LLMAuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error_code": e.error_code,
                "message": e.message,
            },
        )
    except LLMRateLimitError as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error_code": e.error_code,
                "message": e.message,
            },
        )
    except LLMTimeoutError as e:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={
                "error_code": e.error_code,
                "message": e.message,
            },
        )
    except LLMError as e:
        err_code = e.error_code
        status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
        if err_code in ("RECORDING_NOT_FOUND", "TRANSCRIPT_NOT_FOUND"):
            status_code = status.HTTP_404_NOT_FOUND
        elif err_code == "RECORDING_NOT_TRANSCRIBED":
            status_code = status.HTTP_400_BAD_REQUEST

        raise HTTPException(
            status_code=status_code,
            detail={
                "error_code": err_code,
                "message": e.message,
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "INTERNAL_ANALYSIS_ERROR",
                "message": str(e),
            },
        )


@router.get(
    "/{recording_id}/analysis",
    response_model=AIAnalysis,
    summary="Retrieve the full structured AI understanding of a recording",
)
async def get_recording_analysis(recording_id: str):
    clean_id = validate_recording_id(recording_id)

    meta = await storage_service.get_recording_metadata(clean_id)
    if not meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recording with ID '{clean_id}' not found",
        )

    pipeline = get_understanding_pipeline()
    analysis = await pipeline.get_analysis(clean_id)
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis for recording '{clean_id}' not found or not yet generated",
        )

    return analysis


@router.get(
    "/{recording_id}",
    response_model=RecordingUploadResponse,
    summary="Get status and metadata of an uploaded recording",
)
async def get_recording(recording_id: str):
    clean_id = validate_recording_id(recording_id)
    metadata = await storage_service.get_recording_metadata(clean_id)
    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recording with ID '{clean_id}' not found",
        )

    return RecordingUploadResponse(
        recording_id=metadata["recording_id"],
        status=RecordingStatus(metadata.get("status", RecordingStatus.UPLOADED.value)),
        file_size=metadata.get("file_size_bytes", metadata.get("file_size", 0)),
        duration=metadata.get("duration_seconds"),
        mime_type=metadata.get("mime_type", "audio/m4a"),
        created_at=datetime.fromisoformat(metadata["created_at"])
        if "created_at" in metadata
        else datetime.now(timezone.utc),
        message="Recording found",
        audio=metadata.get("audio"),
        processing=metadata.get("processing"),
        transcript=metadata.get("transcript"),
        analysis=metadata.get("analysis"),
        rag=metadata.get("rag"),
        error_code=metadata.get("error_code"),
        error_message=metadata.get("error_message"),
    )


from app.services.rag import (
    RAGPipeline,
    IndexingResponse,
    ChatRequest,
    ChatResponse,
    RAGError,
    EmbeddingAuthenticationError,
    EmbeddingRateLimitError,
    EmbeddingTimeoutError,
    EmbeddingError,
    VectorStoreError,
)


def get_rag_pipeline() -> RAGPipeline:
    return RAGPipeline(storage=storage_service)


@router.post(
    "/{recording_id}/index",
    response_model=IndexingResponse,
    status_code=status.HTTP_200_OK,
    summary="Index transcribed recording into vector store for RAG question answering",
)
async def index_recording(recording_id: str):
    clean_id = validate_recording_id(recording_id)

    meta = await storage_service.get_recording_metadata(clean_id)
    if not meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recording with ID '{clean_id}' not found",
        )

    pipeline = get_rag_pipeline()

    try:
        response = await pipeline.index_recording(clean_id)
        return response
    except EmbeddingAuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error_code": e.error_code,
                "message": e.message,
            },
        )
    except EmbeddingRateLimitError as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error_code": e.error_code,
                "message": e.message,
            },
        )
    except EmbeddingTimeoutError as e:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={
                "error_code": e.error_code,
                "message": e.message,
            },
        )
    except RAGError as e:
        err_code = e.error_code
        status_code_val = e.status_code
        raise HTTPException(
            status_code=status_code_val,
            detail={
                "error_code": err_code,
                "message": e.message,
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "INTERNAL_RAG_INDEX_ERROR",
                "message": str(e),
            },
        )


@router.post(
    "/{recording_id}/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question about a specific recording and receive a grounded answer with timestamp sources",
)
async def chat_with_recording(recording_id: str, request: ChatRequest):
    clean_id = validate_recording_id(recording_id)

    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error_code": "EMPTY_QUESTION",
                "message": "Question text cannot be empty",
            },
        )

    meta = await storage_service.get_recording_metadata(clean_id)
    if not meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recording with ID '{clean_id}' not found",
        )

    pipeline = get_rag_pipeline()

    try:
        answer = await pipeline.answer_question(
            recording_id=clean_id,
            question=request.question,
            top_k=request.top_k,
        )
        return ChatResponse(
            recording_id=answer.recording_id,
            answer=answer.answer,
            sources=answer.sources,
            language=answer.language,
            message="Question answered successfully",
        )
    except EmbeddingAuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error_code": e.error_code,
                "message": e.message,
            },
        )
    except EmbeddingRateLimitError as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error_code": e.error_code,
                "message": e.message,
            },
        )
    except EmbeddingTimeoutError as e:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={
                "error_code": e.error_code,
                "message": e.message,
            },
        )
    except RAGError as e:
        err_code = e.error_code
        status_code_val = e.status_code
        raise HTTPException(
            status_code=status_code_val,
            detail={
                "error_code": err_code,
                "message": e.message,
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "INTERNAL_RAG_CHAT_ERROR",
                "message": str(e),
            },
        )



