# Ollama OCR backend

A small FastAPI service that sends scanned PDF pages and images to a local Ollama vision model, and exposes an OpenAI-compatible chat endpoint for Open WebUI. Text-like files (`txt`, `md`, `csv`, `json`, `html`, `xml`, `yaml`) are returned as decoded text without changing their contents.

## Start

From PowerShell in this folder:

```powershell
Copy-Item .env.example .env
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Ollama must be running and have the configured vision model installed. Defaults use `fredrezones55/Qwen3.5-APEX:latest` at `http://127.0.0.1:11434`. The output token limit defaults to 8,192 (`OLLAMA_NUM_PREDICT`), and the request timeout defaults to 900 seconds (`OLLAMA_TIMEOUT_SECONDS`).

Set `OCR_DEBUG=true` in `.env` and restart the backend to log the processing stages, PDF render time, per-page model and total time, and overall request duration. Set it to `false` (or remove it) to disable these logs.

## Open WebUI

In Open WebUI, open **Admin Settings → Connections → OpenAI → Add Connection**. Set the API base URL to `http://127.0.0.1:8000/v1` when Open WebUI runs directly on this computer; use `http://host.docker.internal:8000/v1` when it runs in Docker. Use any non-empty API key, such as `ollama-ocr`. The `ollama-ocr` model should appear in the model picker.

The `/v1/chat/completions` endpoint accepts OpenAI-format text and inline base64 image messages. Open WebUI's normal attachment handling may extract PDFs inside Open WebUI before sending chat context; use the direct upload endpoint below when the backend itself must receive and OCR the PDF.

For Open WebUI's **Admin Settings → Documents → Content Extraction Engine → External**, set the Document Loader URL to `http://127.0.0.1:8000` when Open WebUI runs directly on this computer, or `http://host.docker.internal:8000` when Open WebUI runs in Docker. Open WebUI appends `/process` and sends each file as a raw `PUT` body. Put any non-empty placeholder such as `local-no-auth` in API Key; this backend does not check that value. Set **Supported Media MIME Types** to `image/*` so uploaded still images use OCR but videos are not sent through the text-only document loader. Leave Headers blank. If Open WebUI runs in a different machine/container without `host.docker.internal`, use a host-reachable backend address instead.

## Video in Chat

Ollama's chat API accepts images, not raw video. The separate `POST /v1/video/frames` endpoint samples timestamped JPEG frames; it does not OCR them or return document text. The Open WebUI filter in `openwebui_video_filter.py` adds those images to the user's chat message, so the currently selected vision model (for example APEX) receives them directly. PDF, image OCR, text extraction, and RAG remain on their existing paths.

Install the filter from **Admin Panel → Functions → Create**. Use an ID such as `ollama_video_frames`, paste the contents of `openwebui_video_filter.py` into the editor, and save. Open WebUI Functions run administrator-provided Python code on the Open WebUI server; review the file before saving it.

In the filter's valves, set **Frame Service URL** to `http://host.docker.internal:8000` for Docker-based Open WebUI on this Windows host, or `http://127.0.0.1:8000` if Open WebUI runs directly on Windows. Keep **Context Size** at `32768` for the APEX model. The defaults reserve 8,192 tokens for instructions, history, and reply; estimate 1,536 tokens per frame; and cap the video at 24 frames resized to 768 pixels. The filter also accounts for existing text and image attachments and reduces the frame budget accordingly; start a fresh chat for videos when an existing conversation has little context left. The model context applies to the entire prompt, not just the video.

Sampled frames are saved as image attachments on the original user message. On later requests, the filter restores their image data into that same turn if Open WebUI has not already included it, so follow-up questions retain the correct video context without re-sampling the MKV or appending images to the newest user turn. These image files remain in Open WebUI storage and consume model context on each request. Updating `openwebui_video_filter.py` locally does not update the Function already saved in Open WebUI; paste the revised Function there and save it.

Attach the filter to APEX in **Workspace → Models → Edit model → Filters**, then enable its toggle in the chat Integrations menu (or select it as a default filter for that model). The default sampler takes up to 24 evenly spaced frames at 0.5 FPS across the entire clip, not just from the beginning. Video processing requires `ffmpeg` and `ffprobe` on the backend machine; defaults limit video uploads to 200 MB and 1,800 seconds (30 minutes). These limits and sampling values are configurable with the `VIDEO_*` settings in `.env`.

The default sampler uses 0.5 FPS across the entire clip with a four-frame minimum when the budget permits. A short clip therefore still gets several moments; longer clips are evenly covered up to the 24-frame cap. `VIDEO_MIN_FRAMES`, `VIDEO_SAMPLE_FPS`, `VIDEO_MAX_FRAMES`, `VIDEO_FRAME_MAX_DIMENSION`, upload size, and duration limits can be tuned in `.env`.

Open WebUI stores chat uploads behind its authenticated file API. The filter reads only the attached file IDs through that API, then calls this backend to sample frames; it does not accept arbitrary file paths or URLs. Do not expose the frame endpoint to untrusted networks without adding authentication.

## Upload files directly

`POST /v1/ocr` accepts multipart form data with a `file` field and returns the extracted text as JSON. The interactive API is available at `http://127.0.0.1:8000/docs`.

```powershell
curl.exe -F "file=@E:\Downloads\Noor-Book.com  حلية الأولياء وطبقات الأصفياء-3.pdf" http://127.0.0.1:8000/v1/ocr
```

Run the included live test to process the sample PDF with the configured Ollama model and write a `- OCR.txt` file beside it:

```powershell
.\.venv\Scripts\python.exe scripts\test_sample_pdf.py
```

The service accepts PDF, common image, and text/JSON formats. Upload size and PDF page limits are configurable in `.env`. OCR is performed page-by-page; the configured Ollama model must support vision.

## Tests

```powershell
python -m pytest
```