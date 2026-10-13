class AudioProcessingError(Exception):
    """Base exception for all audio processing operations."""
    def __init__(self, message: str, error_code: str = "AUDIO_PROCESSING_ERROR", cause: Exception = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.cause = cause


class AudioInspectionError(AudioProcessingError):
    def __init__(self, message: str, cause: Exception = None):
        super().__init__(message, error_code="AUDIO_INSPECTION_ERROR", cause=cause)


class FFmpegNotFoundError(AudioProcessingError):
    def __init__(self, message: str = "FFmpeg is not installed or not found in system PATH / FFMPEG_PATH."):
        super().__init__(message, error_code="FFMPEG_NOT_FOUND")


class AudioConversionError(AudioProcessingError):
    def __init__(self, message: str, cause: Exception = None):
        super().__init__(message, error_code="AUDIO_CONVERSION_ERROR", cause=cause)


class AudioNormalizationError(AudioProcessingError):
    def __init__(self, message: str, cause: Exception = None):
        super().__init__(message, error_code="AUDIO_NORMALIZATION_ERROR", cause=cause)


class VADError(AudioProcessingError):
    def __init__(self, message: str, cause: Exception = None):
        super().__init__(message, error_code="VAD_ERROR", cause=cause)


class ChunkingError(AudioProcessingError):
    def __init__(self, message: str, cause: Exception = None):
        super().__init__(message, error_code="CHUNKING_ERROR", cause=cause)
