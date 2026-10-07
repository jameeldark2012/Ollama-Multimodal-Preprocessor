import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # "ollama" uses Ollama's /api/chat format; "llamacpp" uses OpenAI-compatible /v1/chat/completions
    backend: str = os.getenv("BACKEND", "ollama").strip().lower()

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "fredrezones55/Qwen3.5-APEX:latest")

    llamacpp_base_url: str = os.getenv("LLAMACPP_BASE_URL", "http://127.0.0.1:8081/v1").rstrip("/")
    llamacpp_model: str = os.getenv("LLAMACPP_MODEL", "Qwen3.5-35B-A3B-Tier15-s.gguf")

    public_model_name: str = os.getenv("PUBLIC_MODEL_NAME", "ollama-ocr")
    ollama_timeout_seconds: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "100000"))
    ollama_num_predict: int = int(os.getenv("OLLAMA_NUM_PREDICT", "8192"))
    ocr_debug: bool = os.getenv("OCR_DEBUG", "true").strip().lower() in {"1", "true", "yes", "on"}
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "400"))
    max_pdf_pages: int = int(os.getenv("MAX_PDF_PAGES", "10000"))
    video_max_upload_mb: int = int(os.getenv("VIDEO_MAX_UPLOAD_MB", "5000"))
    video_max_duration_seconds: float = float(os.getenv("VIDEO_MAX_DURATION_SECONDS", "1800"))
    video_sample_fps: float = float(os.getenv("VIDEO_SAMPLE_FPS", "0.5"))
    video_min_frames: int = int(os.getenv("VIDEO_MIN_FRAMES", "4"))
    video_max_frames: int = int(os.getenv("VIDEO_MAX_FRAMES", "24"))
    video_frame_max_dimension: int = int(os.getenv("VIDEO_FRAME_MAX_DIMENSION", "768"))
    video_jpeg_quality: int = int(os.getenv("VIDEO_JPEG_QUALITY", "5"))
    video_ffmpeg_timeout_seconds: float = float(os.getenv("VIDEO_FFMPEG_TIMEOUT_SECONDS", "30"))

    @property
    def active_model(self) -> str:
        """Return the model name for the currently configured backend."""
        return self.llamacpp_model if self.backend == "llamacpp" else self.ollama_model

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def video_max_upload_bytes(self) -> int:
        return self.video_max_upload_mb * 1024 * 1024


settings = Settings()