"""
Nyaya Sahayak — FastAPI Backend
Serves the chat API, FIR drafting endpoints, and case retriever.
"""

from __future__ import annotations
import os
import sys
import uuid
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import (
    process_query,
    find_similar_cases,
    route_query,
    translate_to_english,
    MAPPING_PATH, CORPUS_PATH, CASE_SUMMARIES_PATH, DATA_DIR,
)

# Sarvam STT
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "bns_ipc"))
    from language import speech_to_english, is_configured as sarvam_configured
    SARVAM_STT_AVAILABLE = sarvam_configured()
except ImportError:
    SARVAM_STT_AVAILABLE = False
    def speech_to_english(wav_bytes, **kwargs): raise RuntimeError("Sarvam not available")

# FIR Bot
try:
    from FIR_drafter.fir_drafter import FIRBot
    FIR_AVAILABLE = True
except ImportError:
    try:
        from fir_drafter import FIRBot
        FIR_AVAILABLE = True
    except ImportError:
        FIR_AVAILABLE = False
        print("Warning: FIR drafter not available. Using mock.")


app = FastAPI(title="Nyaya Sahayak API", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_sessions: dict[str, list[dict]] = {}
fir_sessions: dict[str, object] = {}


class ChatRequest(BaseModel):
    session_id: str
    message: str
    language: str = "en"

class ChatResponse(BaseModel):
    session_id: str
    responses: list[dict]
    recommend_fir: bool
    fir_reason: str
    agents_used: list[str]
    needs_fir: bool
    language: str

class RoutePreviewRequest(BaseModel):
    message: str
    language: str = "en"

class CasesRequest(BaseModel):
    query: str
    language: str = "en"
    k: int = 5
    rerank: bool = False

class FIRStartRequest(BaseModel):
    session_id: str
    bns_sections: Optional[str] = None
    complaint_summary: Optional[str] = None

class FIRChatRequest(BaseModel):
    session_id: str
    fir_session_id: str
    message: str

class FIRChatResponse(BaseModel):
    message: str
    field_label: Optional[str]
    fir_text: Optional[str]
    progress: dict
    done: bool

class NewSessionResponse(BaseModel):
    session_id: str


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Nyaya Sahayak</h1><p>Frontend not found.</p>")


@app.post("/api/session/new", response_model=NewSessionResponse)
async def new_session():
    session_id = str(uuid.uuid4())
    chat_sessions[session_id] = []
    return {"session_id": session_id}


# ---------------------------------------------------------------------------
# Route preview — used by frontend to show the right loading message
# while the slow /api/chat call runs
# ---------------------------------------------------------------------------
@app.post("/api/route-preview")
async def route_preview(req: RoutePreviewRequest):
    """
    Lightweight pre-classification call.
    Frontend hits this first to know which agents will run, so it can show
    contextual loading text like "Searching over 1,400 past cases…".
    Cost: ~1 router LLM call (~150 tokens). Fast.
    """
    try:
        english = translate_to_english(req.message, req.language) if req.language == "ta" else req.message
        routing = route_query(english, history=[])
        return {
            "agents": routing.get("agents", ["ipc_bns"]),
            "language": routing.get("language", req.language),
            "reasoning": routing.get("reasoning", ""),
        }
    except Exception:
        # If the preview fails, just return a default — the main chat call still works
        return {"agents": ["ipc_bns"], "language": req.language, "reasoning": "fallback"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if req.session_id not in chat_sessions:
        chat_sessions[req.session_id] = []

    history = chat_sessions[req.session_id]

    try:
        result = process_query(req.message, history, req.language)
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline error: {type(e).__name__}: {str(e)}")

    history.append({"role": "user", "content": req.message})
    if result["responses"]:
        combined = "\n\n".join(
            f"[{r['agent'].upper()} Agent]: {r['response']}"
            for r in result["responses"]
        )
        history.append({"role": "assistant", "content": combined})

    if len(history) > 40:
        chat_sessions[req.session_id] = history[-40:]

    return ChatResponse(
        session_id=req.session_id,
        responses=result["responses"],
        recommend_fir=result["recommend_fir"],
        fir_reason=result.get("fir_reason", ""),
        agents_used=result["agents_used"],
        needs_fir=result["needs_fir"],
        language=result["language"],
    )


@app.post("/api/cases")
async def cases(req: CasesRequest):
    """Standalone case search (lawyer mode UI tab can call this directly)."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")
    try:
        return find_similar_cases(
            query=req.query, language=req.language, k=req.k, rerank=req.rerank,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Case data not loaded: {e}")
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Case search failed: {type(e).__name__}: {str(e)}")


@app.post("/api/fir/start")
async def fir_start(req: FIRStartRequest):
    fir_session_id = str(uuid.uuid4())
    if FIR_AVAILABLE:
        bot = FIRBot(
            bns_sections=req.bns_sections or "Section 318(4) BNS (Cheating)",
            complaint_summary=req.complaint_summary or "User reported an incident.",
        )
    else:
        bot = MockFIRBot(
            bns_sections=req.bns_sections or "Section 318(4) BNS",
            complaint_summary=req.complaint_summary or "",
        )

    fir_sessions[fir_session_id] = bot
    opening, first_label = bot.start()
    return {
        "fir_session_id": fir_session_id,
        "message": opening,
        "field_label": first_label,
        "progress": bot.progress(),
        "done": False,
    }


@app.post("/api/fir/chat", response_model=FIRChatResponse)
async def fir_chat(req: FIRChatRequest):
    bot = fir_sessions.get(req.fir_session_id)
    if not bot:
        raise HTTPException(status_code=404, detail="FIR session not found")

    message, field_label, fir_text = bot.chat(req.message)
    progress = bot.progress()
    done = field_label is None or bot.fir_drafted

    return FIRChatResponse(
        message=message, field_label=field_label, fir_text=fir_text,
        progress=progress, done=done,
    )


@app.get("/api/fir/{fir_session_id}/download")
async def fir_download(fir_session_id: str):
    bot = fir_sessions.get(fir_session_id)
    if not bot:
        raise HTTPException(status_code=404, detail="FIR session not found")
    if not bot._fir_text:
        raise HTTPException(status_code=400, detail="FIR not yet completed")

    import tempfile, traceback

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
        result_path = bot.export_pdf(tmp_path)

        if result_path.endswith(".txt"):
            return FileResponse(result_path, filename="FIR_Draft.txt", media_type="text/plain")
        return FileResponse(result_path, filename="FIR_Draft.pdf", media_type="application/pdf")
    except Exception as e:
        traceback.print_exc()
        try:
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as tmp:
                tmp.write(bot._fir_text)
                tmp_path = tmp.name
            return FileResponse(tmp_path, filename="FIR_Draft.txt", media_type="text/plain")
        except Exception as inner:
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"PDF export failed: {type(e).__name__}: {str(e)}. "
                       f"TXT fallback also failed: {type(inner).__name__}: {str(inner)}"
            )


@app.post("/api/stt")
async def speech_to_text(audio: UploadFile = File(...), lang: str = "en"):
    if not SARVAM_STT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Speech-to-text not available.")
    wav_bytes = await audio.read()
    if not wav_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")
    try:
        transcript = speech_to_english(wav_bytes, filename=audio.filename or "audio.webm")
        if lang == "ta":
            try:
                from language import from_english
                display_text = from_english(transcript, target_lang="ta-IN")
            except Exception:
                display_text = transcript
        else:
            display_text = transcript
        return {
            "transcript": transcript,
            "display_text": display_text,
            "lang": lang,
            "source": "sarvam",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT error: {str(e)}")


@app.get("/api/health")
async def health():
    from main import SCHEME_BOT_AVAILABLE
    return {
        "status": "ok",
        "fir_available": FIR_AVAILABLE,
        "stt_available": SARVAM_STT_AVAILABLE,
        "scheme_bot": "delta" if SCHEME_BOT_AVAILABLE else "llm_fallback",
    }


@app.get("/api/debug/data")
async def debug_data():
    info = {
        "DATA_DIR": DATA_DIR,
        "data_dir_exists": os.path.exists(DATA_DIR),
        "data_dir_listing": (
            sorted(os.listdir(DATA_DIR)) if os.path.exists(DATA_DIR) else "DIR_NOT_FOUND"
        ),
        "files": {},
    }
    for label, path in [
        ("mapping", MAPPING_PATH),
        ("corpus", CORPUS_PATH),
        ("case_summaries", CASE_SUMMARIES_PATH),
    ]:
        info["files"][label] = {
            "path": path,
            "exists": os.path.exists(path),
            "is_file": os.path.isfile(path),
            "is_dir": os.path.isdir(path),
            "size_mb": round(os.path.getsize(path) / 1e6, 3) if os.path.isfile(path) else None,
        }
    return info


# ---------------------------------------------------------------------------
# Mock FIR Bot
# ---------------------------------------------------------------------------
class MockFIRBot:
    FIELDS = [
        {"key": "police_station", "label": "Police Station", "prompt": "What is the name of the police station where you want to file the FIR?", "optional": False},
        {"key": "full_name", "label": "Full Name", "prompt": "What is your full name?", "optional": False},
        {"key": "age", "label": "Age", "prompt": "How old are you?", "optional": False},
        {"key": "address", "label": "Residential Address", "prompt": "What is your full residential address?", "optional": False},
        {"key": "date_of_incident", "label": "Date of Incident", "prompt": "On what date did the incident occur?", "optional": False},
        {"key": "incident_details", "label": "Details of Incident", "prompt": "Please describe what happened in detail.", "optional": False},
        {"key": "informant_contact", "label": "Contact Number", "prompt": "What is your contact phone number?", "optional": False},
    ]

    def __init__(self, bns_sections="", complaint_summary=""):
        self.bns_sections = bns_sections
        self.complaint_summary = complaint_summary
        self.collected = {}
        self.current_index = 0
        self.fir_drafted = False
        self._fir_text = None

    def start(self):
        field = self.FIELDS[0]
        msg = f"I'll help you draft a formal FIR.\nApplicable sections: {self.bns_sections}\n\n{field['prompt']}"
        return msg, field["label"]

    def chat(self, answer):
        if not answer.strip():
            return "Please provide an answer.", self.FIELDS[self.current_index]["label"], None
        field = self.FIELDS[self.current_index]
        self.collected[field["key"]] = answer
        self.current_index += 1
        if self.current_index >= len(self.FIELDS):
            self.fir_drafted = True
            self._fir_text = self._render()
            return "Thank you! Your FIR draft is ready.", None, self._fir_text
        nxt = self.FIELDS[self.current_index]
        return nxt["prompt"], nxt["label"], None

    def _render(self):
        c = self.collected
        return f"""TO
The Station House Officer,
Police Station: {c.get('police_station', '___')}

Subject: First Information Report

I, {c.get('full_name', '___')}, aged {c.get('age', '___')} years,
resident of {c.get('address', '___')}, do hereby lodge the following complaint:

DATE OF OCCURRENCE: {c.get('date_of_incident', '___')}

DETAILS:
{c.get('incident_details', '___')}

APPLICABLE SECTIONS: {self.bns_sections}

(Signature of Informant)
Name: {c.get('full_name', '___')}
Contact: {c.get('informant_contact', '___')}"""

    def progress(self):
        return {
            "collected": self.current_index,
            "total": len(self.FIELDS),
            "percent": int(self.current_index / len(self.FIELDS) * 100),
        }

    def export_pdf(self, path="FIR_Draft.pdf"):
        try:
            from FIR_drafter.fir_drafter import save_fir_as_pdf
            return save_fir_as_pdf(self._fir_text, path)
        except Exception:
            with open(path.replace(".pdf", ".txt"), "w") as f:
                f.write(self._fir_text)
            return path.replace(".pdf", ".txt")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)