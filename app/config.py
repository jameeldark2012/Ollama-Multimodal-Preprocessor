import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # Backend: "ollama", "llamacpp", or "gemini"
    # Ollama uses /api/chat format; llamacpp uses OpenAI-compatible /v1/chat/completions
    # Gemini uses Google's Generative AI API
    backend: str = os.getenv("BACKEND", "ollama").strip().lower()

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "fredrezones55/Qwen3.5-APEX:latest")

    llamacpp_base_url: str = os.getenv("LLAMACPP_BASE_URL", "http://127.0.0.1:8081/v1").rstrip("/")
    llamacpp_model: str = os.getenv("LLAMACPP_MODEL", "Qwen3.5-35B-A3B-Tier15-s.gguf")

    # Gemini backend settings
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    gemini_batch_size: int = int(os.getenv("GEMINI_BATCH_SIZE", "5"))
    gemini_rpm_limit: int = int(os.getenv("GEMINI_RPM_LIMIT", "10"))
    gemini_tpm_limit: int = int(os.getenv("GEMINI_TPM_LIMIT", "65000"))
    gemini_safety_margin: float = float(os.getenv("GEMINI_SAFETY_MARGIN", "0.8"))
    gemini_timeout_seconds: float = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "300"))
    gemini_fallback_to_local: bool = os.getenv("GEMINI_FALLBACK_TO_LOCAL", "true").strip().lower() in {"1", "true", "yes", "on"}
    gemini_local_fallback_backend: str = os.getenv("GEMINI_LOCAL_FALLBACK_BACKEND", "").strip().lower()
    gemini_max_output_tokens: int = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "8192"))

    public_model_name: str = os.getenv("PUBLIC_MODEL_NAME", "ollama-ocr")
    ollama_timeout_seconds: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "900"))
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
    checkpoint_dir: str = os.getenv("CHECKPOINT_DIR", "checkpoints")
    checkpoint_save_interval: int = int(os.getenv("CHECKPOINT_SAVE_INTERVAL", "20"))

    @property
    def active_model(self) -> str:
        """Return the model name for the currently configured backend."""
        if self.backend == "llamacpp":
            return self.llamacpp_model
        elif self.backend == "gemini":
            return self.gemini_model
        else:
            return self.ollama_model

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def video_max_upload_bytes(self) -> int:
        return self.video_max_upload_mb * 1024 * 1024


settings = Settings()