"""
Nyaya Sahayak — FastAPI Backend
Serves the chat API and FIR drafting endpoints.
"""

from __future__ import annotations
import os
import sys
import json
import uuid
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import process_query

# FIR Bot — import carefully
try:
    from FIR_drafter.fir_drafter import FIRBot
    FIR_AVAILABLE = True
except ImportError:
    try:
        # Try flat import if running from same directory
        from fir_drafter import FIRBot
        FIR_AVAILABLE = True
    except ImportError:
        FIR_AVAILABLE = False
        print("Warning: FIR drafter not available. Using mock.")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="Nyaya Sahayak API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------
chat_sessions: dict[str, list[dict]] = {}
fir_sessions: dict[str, object] = {}  # session_id -> FIRBot instance

# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main frontend."""
    html_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Nyaya Sahayak</h1><p>Frontend not found. Run the app from the project root.</p>")


@app.post("/api/session/new", response_model=NewSessionResponse)
async def new_session():
    """Create a new chat session."""
    session_id = str(uuid.uuid4())
    chat_sessions[session_id] = []
    return {"session_id": session_id}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Main chat endpoint — routes to appropriate agents."""
    if req.session_id not in chat_sessions:
        chat_sessions[req.session_id] = []

    history = chat_sessions[req.session_id]

    try:
        result = process_query(req.message, history, req.language)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    # Update history
    history.append({"role": "user", "content": req.message})
    if result["responses"]:
        # Combine all agent responses into history
        combined = "\n\n".join(
            f"[{r['agent'].upper()} Agent]: {r['response']}"
            for r in result["responses"]
        )
        history.append({"role": "assistant", "content": combined})

    # Keep history bounded
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


@app.post("/api/fir/start")
async def fir_start(req: FIRStartRequest):
    """Start a new FIR drafting session."""
    fir_session_id = str(uuid.uuid4())

    if FIR_AVAILABLE:
        bot = FIRBot(
            bns_sections=req.bns_sections or "Section 318(4) BNS (Cheating)",
            complaint_summary=req.complaint_summary or "User reported an incident.",
        )
    else:
        # Mock bot for testing
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
    """Continue FIR drafting conversation."""
    bot = fir_sessions.get(req.fir_session_id)
    if not bot:
        raise HTTPException(status_code=404, detail="FIR session not found")

    message, field_label, fir_text = bot.chat(req.message)
    progress = bot.progress()
    done = field_label is None or bot.fir_drafted

    return FIRChatResponse(
        message=message,
        field_label=field_label,
        fir_text=fir_text,
        progress=progress,
        done=done,
    )


@app.get("/api/fir/{fir_session_id}/download")
async def fir_download(fir_session_id: str):
    """Download the completed FIR as PDF."""
    bot = fir_sessions.get(fir_session_id)
    if not bot:
        raise HTTPException(status_code=404, detail="FIR session not found")
    if not bot._fir_text:
        raise HTTPException(status_code=400, detail="FIR not yet completed")

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        path = bot.export_pdf(tmp.name)

    return FileResponse(path, filename="FIR_Draft.pdf", media_type="application/pdf")


@app.get("/api/health")
async def health():
    return {"status": "ok", "fir_available": FIR_AVAILABLE}


# ---------------------------------------------------------------------------
# Mock FIR Bot (fallback)
# ---------------------------------------------------------------------------
class MockFIRBot:
    """Minimal mock when fir_drafter isn't importable."""
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
        next_field = self.FIELDS[self.current_index]
        return next_field["prompt"], next_field["label"], None

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)