"""
FIR Drafting Bot - Backend
Designed to plug into a larger pipeline that retrieves BNS sections
and then offers to draft an FIR based on the user's situation.

Collection strategy: Python drives the field order strictly one at a time.
The LLM is only used to (a) validate / clean each answer and (b) generate
a follow-up question if the answer is insufficient.
"""

import os
import re
import json
from groq import Groq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# Placeholder data injected by the upstream pipeline
# ---------------------------------------------------------------------------
PLACEHOLDER_BNS_SECTIONS = (
    "Section 318(4) BNS (Cheating), Section 316(5) BNS (Criminal Breach of Trust)"
)
PLACEHOLDER_COMPLAINT_SUMMARY = (
    "The complainant transferred Rs.2,50,000 to the accused on the promise of "
    "delivering goods within 15 days. The accused neither delivered the goods "
    "nor returned the money and is now unreachable."
)

# ---------------------------------------------------------------------------
# Field definitions — ordered list drives the collection sequence
# ---------------------------------------------------------------------------
FIELDS = [
    {
        "key": "police_station",
        "label": "Police Station",
        "prompt": "What is the name of the police station where you want to file the FIR?",
        "hint": "location of the police station",
        "optional": False
    },
    {
        "key": "full_name",
        "label": "Full Name",
        "prompt": "What is your full name?",
        "hint": "e.g. Ramesh Kumar Sharma",
        "optional": False,
    },
    {
        "key": "fathers_name",
        "label": "Father's / Spouse's Name",
        "prompt": "What is your father's name (or spouse's name if applicable)?",
        "hint": "e.g. Suresh",
        "optional": False,
    },
    {
        "key": "age",
        "label": "Age",
        "prompt": "How old are you?",
        "hint": "e.g. 35",
        "optional": False,
    },
    {
        "key": "address",
        "label": "Residential Address",
        "prompt": "What is your full residential address? Include house no., street and PIN code",
        "hint": "Include house no., street, and PIN code",
        "optional": False,
    },
    {
        "key": "date_of_incident",
        "label": "Date of Incident",
        "prompt": "On what date did the incident occur?",
        "hint": "e.g. 20 April 2026",
        "optional": False,
    },
    {
        "key": "time_of_incident",
        "label": "Time of Incident",
        "prompt": "At approximately what time did the incident occur?",
        "hint": "e.g. 3:30 PM, or 'evening' if unsure",
        "optional": False,
    },
    {
        "key": "incident_details",
        "label": "Details of Incident",
        "prompt": "Please describe what happened in detail.",
        "hint": "A brief of what happened",
        "optional": False,
    },
    {
        "key": "witnesses",
        "label": "Witnesses",
        "prompt": "Were there any witnesses to the incident? Provide names and contact details (optional)",
        "hint": "Provide names and contact details, or type 'None' if no witnesses",
        "optional": True,
    },
    {
        "key": "property_involved",
        "label": "Property / Loss Involved",
        "prompt": "Was any property stolen or damaged, or was there a financial loss? (optional)",
        "hint": "Describe items and approximate value, or type 'None' if not applicable",
        "optional": True,
    },
    {
        "key": "informant_contact",
        "label": "Your Contact Number",
        "prompt": "What is your contact phone number?",
        "hint": "e.g. 9876543210",
        "optional": False,
    },
]

# ---------------------------------------------------------------------------
# FIR plain-text template (used internally / for display)
# ---------------------------------------------------------------------------
FIR_TEMPLATE = """
TO
The Station House Officer,
Police Station: {police_station}


Subject: First Information Report under Section 173 BNSS, 2023

Respected Sir/Madam,

I, {full_name}, S/o / D/o {fathers_name}, aged {age} years, resident of {address}, do hereby lodge the following complaint:

1. DATE AND TIME OF OCCURRENCE:
   On {date_of_incident} at approximately {time_of_incident}.

2. DETAILS OF INCIDENT:
   {incident_details}

3. WITNESSES:
   {witnesses}

4. PROPERTY INVOLVED:
   {property_involved}

5. SECTIONS APPLICABLE:
   The above offence is punishable under Section(s) {bns_sections} of the Bharatiya Nyaya Sanhita, 2023.

I request you to register an FIR and investigate the matter. I am ready to cooperate fully in the investigation.


(Signature of Informant)
Name:    {full_name}
Contact: {informant_contact}
""".strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_json_raw(text: str) -> dict | None:
    """Try to parse JSON from model output (with or without code fences)."""
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    raw = fenced.group(1) if fenced else text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# PDF Export
# ---------------------------------------------------------------------------
def save_fir_as_pdf(fir_text: str, output_path: str = "FIR_Draft.pdf") -> str:
    """
    Render the plain-text FIR as a properly formatted PDF.
    Returns the path to the saved PDF.

    Requires: reportlab  (pip install reportlab)
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
    from reportlab.lib import colors

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
    )

    # --- Styles ---
    heading_style = ParagraphStyle(
        "Heading",
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=18,
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    subheading_style = ParagraphStyle(
        "SubHeading",
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
        spaceAfter=2,
    )
    label_style = ParagraphStyle(
        "Label",
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        spaceAfter=2,
    )
    body_style = ParagraphStyle(
        "Body",
        fontName="Helvetica",
        fontSize=10,
        leading=15,
        spaceAfter=6,
    )
    right_style = ParagraphStyle(
        "Right",
        fontName="Helvetica",
        fontSize=10,
        leading=15,
        alignment=TA_RIGHT,
        spaceAfter=4,
    )
    bold_right_style = ParagraphStyle(
        "BoldRight",
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=15,
        alignment=TA_RIGHT,
        spaceAfter=2,
    )

    # Parse the flat text back into structured fields for nicer rendering
    # We'll do a simple line-by-line approach on the template output
    lines = fir_text.splitlines()

    story = []

    # Title block
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("FIRST INFORMATION REPORT", heading_style))
    story.append(Paragraph("Under Section 173, Bharatiya Nagarik Suraksha Sanhita (BNSS), 2023", subheading_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.black, spaceAfter=10))

    in_section = False
    section_lines = []
    current_section_title = ""

    def flush_section():
        nonlocal section_lines, current_section_title
        if current_section_title:
            story.append(Paragraph(current_section_title, label_style))
        if section_lines:
            content = " ".join(l.strip() for l in section_lines if l.strip())
            story.append(Paragraph(content, body_style))
            story.append(Spacer(1, 0.2 * cm))
        section_lines = []
        current_section_title = ""

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip blank lines between sections — we handle spacing ourselves
        if not stripped:
            i += 1
            continue

        # "TO" block
        if stripped == "TO":
            story.append(Paragraph("TO", label_style))
            i += 1
            continue

        # Station / District lines
        if stripped.startswith("The Station House Officer") or \
           stripped.startswith("Police Station") or \
           stripped.startswith("District"):
            story.append(Paragraph(stripped, body_style))
            i += 1
            continue

        # Subject line
        if stripped.startswith("Subject:"):
            story.append(Spacer(1, 0.3 * cm))
            story.append(Paragraph(f"<b>{stripped}</b>", body_style))
            story.append(Spacer(1, 0.3 * cm))
            i += 1
            continue

        # Salutation
        if stripped.startswith("Respected Sir"):
            story.append(Paragraph(stripped, body_style))
            i += 1
            continue

        # Opening sentence (I, <name>, ...)
        if stripped.startswith("I,") and "do hereby" in stripped:
            story.append(Paragraph(stripped, body_style))
            story.append(Spacer(1, 0.3 * cm))
            i += 1
            continue

        # Numbered sections  e.g. "1. DATE AND TIME..."
        section_match = re.match(r"^(\d+)\.\s+(.+)$", stripped)
        if section_match:
            flush_section()
            num = section_match.group(1)
            title = section_match.group(2)
            current_section_title = f"{num}. {title}"
            i += 1
            continue

        # Indented content under a section
        if line.startswith("   ") and current_section_title:
            section_lines.append(stripped)
            i += 1
            continue

        # Closing paragraph
        if stripped.startswith("I request"):
            flush_section()
            story.append(Spacer(1, 0.4 * cm))
            story.append(Paragraph(stripped, body_style))
            story.append(Spacer(1, 0.8 * cm))
            i += 1
            continue

        # Signature block — right-aligned
        if stripped.startswith("(Signature") or stripped.startswith("Name:") or stripped.startswith("Contact:"):
            flush_section()
            story.append(Paragraph(stripped, right_style))
            i += 1
            continue

        # Anything else
        flush_section()
        story.append(Paragraph(stripped, body_style))
        i += 1

    flush_section()  # flush any trailing section

    # Footer rule
    story.append(Spacer(1, 0.5 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))


    doc.build(story)
    return output_path


# ---------------------------------------------------------------------------
# Core FIR Bot
# ---------------------------------------------------------------------------
class FIRBot:
    """
    Collects FIR fields strictly one at a time.

    Public API:
        opening, label = bot.start()
        message, label, fir = bot.chat(user_answer)
        pdf_path = bot.export_pdf(output_path="FIR_Draft.pdf")
        info = bot.progress()
    """

    CANCEL_WORDS = {"exit", "quit", "stop", "cancel", "nevermind", "abort"}

    def __init__(self, bns_sections: str = None, complaint_summary: str = None):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.bns_sections = bns_sections or PLACEHOLDER_BNS_SECTIONS
        self.complaint_summary = complaint_summary or PLACEHOLDER_COMPLAINT_SUMMARY

        self.collected: dict[str, str] = {}
        self.current_index: int = 0
        self.fir_drafted: bool = False
        self._fir_text: str | None = None  # cached rendered FIR

    # ------------------------------------------------------------------
    # Internal LLM call
    # ------------------------------------------------------------------
    def _call(self, system: str, user_msg: str, max_tokens: int = 256) -> str:
        resp = self.client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # Validate + clean one answer via LLM
    # ------------------------------------------------------------------
    def _validate(self, field: dict, answer: str) -> dict:
        system = (
            f'You are validating a single field for an Indian FIR form.\n'
            f'Field name: {field["label"]}\n'
            f'Hint: {field["hint"]}\n'
            f'Optional: {field["optional"]}\n\n'
            'Rules:\n'
            '- If optional=true and user says none/no/not applicable (any case), '
            'set ok=true and cleaned="None".\n'
            '- Accept reasonable informal answers and clean/format them.\n'
            '- Reject only if the answer is genuinely unusable (e.g. blank, gibberish, '
            'or missing critical info like no city in an address).\n'
            '- followup must be a single short question, empty string if ok=true.\n\n'
            'Respond with ONLY a raw JSON object, no markdown:\n'
            '{"ok": true/false, "cleaned": "<value>", "followup": "<question or empty>"}'
        )
        raw = self._call(system, answer)
        result = extract_json_raw(raw)
        if result and "ok" in result:
            return result
        return {"ok": True, "cleaned": answer, "followup": ""}

    # ------------------------------------------------------------------
    # Current field
    # ------------------------------------------------------------------
    @property
    def _current_field(self) -> dict | None:
        if self.current_index < len(FIELDS):
            return FIELDS[self.current_index]
        return None

    # ------------------------------------------------------------------
    # start()
    # ------------------------------------------------------------------
    def start(self) -> tuple[str, str]:
        field = self._current_field
        opening = (
            "I'll help you draft a formal FIR based on your situation.\n"
            f"Applicable sections: {self.bns_sections}\n\n"
            "I'll ask you one question at a time — please answer as clearly as you can.\n\n"
            f"{field['prompt']}"
        )
        if field["optional"]:
            opening += "\n(Optional — type 'None' if not applicable)"
        return opening, field["label"]

    # ------------------------------------------------------------------
    # chat()
    # ------------------------------------------------------------------
    def chat(self, user_answer: str) -> tuple[str, str | None, str | None]:
        """
        Returns: (message, field_label | None, fir_text | None)
        fir_text is set when all fields are collected.
        field_label is None when done or cancelled.
        """
        if user_answer.strip().lower() in self.CANCEL_WORDS:
            self.fir_drafted = True
            return "Understood. The FIR session has been cancelled.", None, None

        field = self._current_field
        if field is None:
            return "All fields collected.", None, self._fir_text

        result = self._validate(field, user_answer)

        if result.get("ok"):
            self.collected[field["key"]] = result.get("cleaned", user_answer)
            self.current_index += 1

            next_field = self._current_field
            if next_field is None:
                self.fir_drafted = True
                self._fir_text = self._render_fir()
                return (
                    "Thank you! I have all the details. Your FIR draft is ready.",
                    None,
                    self._fir_text,
                )

            msg = next_field["prompt"]
            if next_field["optional"]:
                msg += "\n(Optional — type 'None' if not applicable)"
            return msg, next_field["label"], None

        else:
            followup = result.get("followup") or f"Could you please clarify your {field['label']}?"
            return followup, field["label"], None

    # ------------------------------------------------------------------
    # Render FIR text
    # ------------------------------------------------------------------
    def _render_fir(self) -> str:
        fields = {**self.collected, "bns_sections": self.bns_sections}
        return FIR_TEMPLATE.format(**fields)

    # ------------------------------------------------------------------
    # Export to PDF
    # ------------------------------------------------------------------
    def export_pdf(self, output_path: str = "FIR_Draft.pdf") -> str:
        """
        Export the completed FIR as a PDF.
        Returns the path to the saved file.
        Raises RuntimeError if the FIR is not yet complete.
        """
        if not self._fir_text:
            raise RuntimeError("FIR is not yet complete. Finish the conversation first.")
        return save_fir_as_pdf(self._fir_text, output_path)

    # ------------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------------
    def progress(self) -> dict:
        return {
            "collected": self.current_index,
            "total": len(FIELDS),
            "percent": int(self.current_index / len(FIELDS) * 100),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_fir_bot(bns_sections: str = None, complaint_summary: str = None) -> FIRBot:
    return FIRBot(bns_sections=bns_sections, complaint_summary=complaint_summary)


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------
def run_cli():
    print("=" * 60)
    print("  FIR DRAFTING BOT  —  CLI Test Mode")
    print("  Powered by Groq + LLaMA 3.3 70B")
    print("=" * 60)
    print()

    bot = FIRBot(
        bns_sections=PLACEHOLDER_BNS_SECTIONS,
        complaint_summary=PLACEHOLDER_COMPLAINT_SUMMARY,
    )

    opening, first_label = bot.start()
    print(f"  [Asking for: {first_label}]")
    print(f"Bot: {opening}\n")

    while not bot.fir_drafted:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break

        if not user_input:
            continue

        reply, field_label, fir = bot.chat(user_input)
        p = bot.progress()

        print(f"\n  [Progress: {p['collected']}/{p['total']} fields | {p['percent']}%]")
        if field_label:
            print(f"  [Asking for: {field_label}]")
        print(f"Bot: {reply}\n")

        if fir:
            print("\n" + "=" * 60)
            print("  FIR DRAFT (plain text)")
            print("=" * 60)
            print(fir)
            print("=" * 60)

            # --- PDF export prompt ---
            print("\nWould you like to download the FIR as a PDF? (yes/no): ", end="")
            try:
                choice = input().strip().lower()
            except (EOFError, KeyboardInterrupt):
                choice = "no"

            if choice in {"yes", "y"}:
                print("Enter filename (press Enter for 'FIR_Draft.pdf'): ", end="")
                try:
                    fname = input().strip()
                except (EOFError, KeyboardInterrupt):
                    fname = ""
                fname = fname if fname.endswith(".pdf") else (fname + ".pdf" if fname else "FIR_Draft.pdf")
                try:
                    path = bot.export_pdf(fname)
                    print(f"\n  PDF saved to: {path}")
                except Exception as e:
                    print(f"\n  Error saving PDF: {e}")
            else:
                print("No PDF saved.")
            break


if __name__ == "__main__":
    run_cli()