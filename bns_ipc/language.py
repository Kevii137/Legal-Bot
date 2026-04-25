"""
Sarvam AI language wrapper — translation, speech-to-text, text-to-speech.

Single API key (`SARVAM_API_KEY` env var) powers four functions:
    to_english(text, source_lang="auto") -> str
    from_english(text, target_lang)      -> str
    speech_to_english(wav_bytes)         -> str
    english_to_speech(text, target_lang) -> bytes  (WAV audio)

Pipeline (UI side):
    user input (any language, text or voice)
        -> to_english() / speech_to_english()
        -> Comparator/CaseRetriever (always English internally)
        -> from_english() / english_to_speech()
        -> user output

Reference: https://docs.sarvam.ai
Set SARVAM_API_KEY in env before calling.
"""

from __future__ import annotations

import base64
import io
import os
import re
import wave

import requests

# ----------------------------------------------------------------------------
# Endpoints (override via env if Sarvam moves them)
# ----------------------------------------------------------------------------
SARVAM_TRANSLATE_URL = os.environ.get(
    "SARVAM_TRANSLATE_URL", "https://api.sarvam.ai/translate"
)
SARVAM_STT_URL = os.environ.get(
    "SARVAM_STT_URL", "https://api.sarvam.ai/speech-to-text"
)
SARVAM_TTS_URL = os.environ.get(
    "SARVAM_TTS_URL", "https://api.sarvam.ai/text-to-speech"
)

# Defaults — overridable per call
DEFAULT_STT_MODEL = os.environ.get("SARVAM_STT_MODEL", "saaras:v3")
DEFAULT_TTS_MODEL = os.environ.get("SARVAM_TTS_MODEL", "bulbul:v3")

# ----------------------------------------------------------------------------
# Sarvam supported language codes (BCP-47-ish)
# ----------------------------------------------------------------------------
LANG_CODES = {
    "english": "en-IN",   "en": "en-IN",
    "hindi": "hi-IN",     "hi": "hi-IN",
    "tamil": "ta-IN",     "ta": "ta-IN",
    "telugu": "te-IN",    "te": "te-IN",
    "kannada": "kn-IN",   "kn": "kn-IN",
    "malayalam": "ml-IN", "ml": "ml-IN",
    "marathi": "mr-IN",   "mr": "mr-IN",
    "bengali": "bn-IN",   "bn": "bn-IN",
    "gujarati": "gu-IN",  "gu": "gu-IN",
    "punjabi": "pa-IN",   "pa": "pa-IN",
    "odia": "od-IN",      "or": "od-IN",
    "urdu": "ur-IN",      "ur": "ur-IN",
}


def normalize_lang(code: str) -> str:
    """Accepts 'hi', 'hindi', 'hi-IN', etc. Returns the BCP-47 form Sarvam uses."""
    if not code:
        return "en-IN"
    code = code.strip().lower()
    if code in LANG_CODES:
        return LANG_CODES[code]
    # Already in BCP-47 form like "hi-IN"
    if "-" in code and len(code) == 5:
        return code.split("-")[0] + "-" + code.split("-")[1].upper()
    return "en-IN"


# ----------------------------------------------------------------------------
# Auth
# ----------------------------------------------------------------------------
def _api_key() -> str:
    key = os.environ.get("SARVAM_API_KEY", "").strip()
    if not key:
        raise RuntimeError("SARVAM_API_KEY not set in environment.")
    return key


def _json_headers() -> dict:
    return {"api-subscription-key": _api_key(), "Content-Type": "application/json"}


def _form_headers() -> dict:
    """For multipart/form-data (STT)."""
    return {"api-subscription-key": _api_key()}


# ----------------------------------------------------------------------------
# Translation
# ----------------------------------------------------------------------------
# Sarvam Mayura silently keeps the input unchanged when the text exceeds
# ~500 chars. Chunk on paragraph boundaries to avoid this trap.
_TRANSLATE_CHUNK_LIMIT = 500


def _translate_once(text: str, source: str, target: str, timeout: int = 60) -> str:
    body = {
        "input": text,
        "source_language_code": source,
        "target_language_code": target,
    }
    r = requests.post(SARVAM_TRANSLATE_URL, headers=_json_headers(), json=body, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    for k in ("translated_text", "output", "text"):
        v = data.get(k)
        if v and str(v).strip():
            return str(v).strip()
    raise ValueError(f"Unexpected translate response shape: {data!r}")


def _chunked_translate(text: str, source: str, target: str) -> str:
    """Split on paragraph boundaries, translate each chunk, rejoin."""
    parts = text.split("\n")
    chunks, current = [], ""
    for p in parts:
        if len(current) + len(p) + 1 > _TRANSLATE_CHUNK_LIMIT and current:
            chunks.append(current)
            current = p
        else:
            current = f"{current}\n{p}" if current else p
    if current:
        chunks.append(current)

    out = []
    for c in chunks:
        if not c.strip():
            out.append(c)
            continue
        try:
            out.append(_translate_once(c, source, target))
        except Exception:
            # On translation failure, keep the original chunk so we don't lose content
            out.append(c)
    return "\n".join(out)


def to_english(text: str, source_lang: str = "auto") -> str:
    """Translate any-language text to English. Pass 'auto' to let Sarvam detect."""
    if not text or not text.strip():
        return text
    src = "auto" if source_lang == "auto" else normalize_lang(source_lang)
    if src == "en-IN":
        return text
    if len(text) <= _TRANSLATE_CHUNK_LIMIT:
        return _translate_once(text, src, "en-IN")
    return _chunked_translate(text, src, "en-IN")


def from_english(text: str, target_lang: str) -> str:
    """Translate English text into one of Sarvam's supported Indian languages."""
    if not text or not text.strip():
        return text
    tgt = normalize_lang(target_lang)
    if tgt == "en-IN":
        return text
    if len(text) <= _TRANSLATE_CHUNK_LIMIT:
        return _translate_once(text, "en-IN", tgt)
    return _chunked_translate(text, "en-IN", tgt)


# ----------------------------------------------------------------------------
# Speech to text (Saaras)
# ----------------------------------------------------------------------------
def speech_to_english(
    wav_bytes: bytes,
    *,
    filename: str = "audio.wav",
    model: str | None = None,
    timeout: int = 120,
) -> str:
    """
    Take WAV audio bytes (any supported Indian language), return English transcript.
    Uses mode='translate' so we get English directly without a separate Mayura call.
    """
    if not wav_bytes:
        raise ValueError("Empty audio bytes.")
    files = {"file": (filename, wav_bytes, "audio/wav")}
    data = {"model": model or DEFAULT_STT_MODEL, "mode": "translate"}
    r = requests.post(
        SARVAM_STT_URL, headers=_form_headers(), files=files, data=data, timeout=timeout
    )
    r.raise_for_status()
    payload = r.json()
    transcript = payload.get("transcript")
    if transcript is None:
        raise ValueError(f"Unexpected STT response shape: {payload!r}")
    return str(transcript).strip()


# ----------------------------------------------------------------------------
# Text to speech (Bulbul)
# ----------------------------------------------------------------------------
def english_to_speech(
    text: str,
    target_lang: str,
    *,
    speaker: str | None = None,
    model: str | None = None,
    timeout: int = 120,
) -> bytes:
    """
    Translate English -> target_lang (via Mayura), then synthesize WAV audio (Bulbul).
    Returns raw WAV bytes ready to play in any audio widget.
    """
    if not text or not text.strip():
        raise ValueError("Empty text for TTS.")
    tgt = normalize_lang(target_lang)
    spoken = text if tgt == "en-IN" else from_english(text, target_lang)

    body = {
        "text": _strip_markdown(spoken)[:2500],   # Bulbul has length cap; strip markdown formatting
        "target_language_code": tgt,
        "model": model or DEFAULT_TTS_MODEL,
    }
    if speaker:
        body["speaker"] = speaker

    r = requests.post(SARVAM_TTS_URL, headers=_json_headers(), json=body, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    audios = data.get("audios")
    if not audios:
        raise ValueError(f"Unexpected TTS response shape: {data!r}")
    return base64.b64decode(audios[0])


# ----------------------------------------------------------------------------
# Audio helpers (Gradio integration)
# ----------------------------------------------------------------------------
def numpy_to_wav_bytes(samples, sample_rate: int) -> bytes:
    """Float32 numpy mono/stereo -> 16-bit mono WAV bytes (what Sarvam STT expects)."""
    import numpy as np
    if samples is None or len(samples) == 0:
        raise ValueError("Empty audio.")
    s = np.asarray(samples, dtype=np.float32)
    if s.ndim == 2:
        s = s.mean(axis=1)
    s = np.clip(s, -1.0, 1.0)
    pcm = (s * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def wav_bytes_to_numpy(wav_bytes: bytes):
    """Decode WAV bytes to (sample_rate, numpy float32 mono in [-1, 1]) for Gradio Audio."""
    import numpy as np
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sw == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sw}")
    if n_channels > 1:
        x = x.reshape(-1, n_channels).mean(axis=1)
    return sr, x


def _strip_markdown(text: str, max_chars: int = 2400) -> str:
    """Bulbul reads text literally; strip markdown so we don't pronounce asterisks."""
    t = re.sub(r"```[\s\S]*?```", " ", text)
    t = re.sub(r"`([^`]+)`", r"\1", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
    t = re.sub(r"[*_#>|]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:max_chars]


# ----------------------------------------------------------------------------
# Convenience: is the wrapper configured?
# ----------------------------------------------------------------------------
def is_configured() -> bool:
    return bool(os.environ.get("SARVAM_API_KEY", "").strip())