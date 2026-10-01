"""IRIS: a small, supportive listening companion prototype.

This is not a clinical service. Do not deploy with real health data without
security, privacy, clinical, and legal review. Chat content is processed by
Google Gemini using the configured Gemini API.
"""
import os
import re
import json
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_sqlalchemy import SQLAlchemy
from jinja2 import ChoiceLoader, FileSystemLoader
from sqlalchemy import inspect
from google import genai

load_dotenv()

APP_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=APP_DIR)
app.jinja_loader = ChoiceLoader([
    FileSystemLoader(APP_DIR),
    FileSystemLoader(os.path.join(APP_DIR, "templates")),
])
os.makedirs(app.instance_path, exist_ok=True)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-only-change-me")
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///iris.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Change this line in app.py:
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not configured.")

client = genai.Client(api_key=GEMINI_API_KEY)

ACTIVE_WINDOW_DAYS = 7
VALID_AGES = {"12-18", "19-28", "29-40", "40+"}
VALID_SEX = {"male", "female", "prefer_not_to_say"}


def utcnow():
    """Return naive UTC for SQLite's timezone-naive DateTime columns."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(36), unique=True, nullable=False, index=True)
    email = db.Column(db.String(254), nullable=False, unique=True, index=True)
    alias = db.Column(db.String(80), nullable=False)
    place = db.Column(db.String(120), default="")
    age_group = db.Column(db.String(10), nullable=False)
    sex = db.Column(db.String(24), default="prefer_not_to_say")
    created_at = db.Column(db.DateTime, default=utcnow, nullable=False)
    last_active_at = db.Column(db.DateTime, default=utcnow, nullable=False)
    login_expires_at = db.Column(db.DateTime, nullable=True)


class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(36), nullable=False, index=True)
    role = db.Column(db.String(10), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=utcnow, nullable=False)


class Provider(db.Model):
    """Demo routing labels only; these are not real clinicians or referrals."""
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(40), unique=True, nullable=False)
    role = db.Column(db.String(20), nullable=False)
    category = db.Column(db.String(30), nullable=False)
    level = db.Column(db.Integer, nullable=False)


class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    alias = db.Column(db.String(80), nullable=False, default="Guest")
    rating = db.Column(db.Integer, nullable=False)
    comment = db.Column(db.String(800), nullable=False, default="")
    created_at = db.Column(db.DateTime, default=utcnow, nullable=False)


def purge_stale_users():
    cutoff = utcnow() - timedelta(days=ACTIVE_WINDOW_DAYS)
    stale = User.query.filter(User.last_active_at < cutoff).all()
    for user in stale:
        ChatMessage.query.filter_by(session_id=user.session_id).delete()
        db.session.delete(user)
    if stale:
        db.session.commit()


def seed_demo_providers():
    if Provider.query.first():
        return
    categories = ("loneliness", "trauma", "addiction", "chronic_depression", "life_problem")
    rows = []
    for category in categories:
        abbr = category.upper()[:12]
        rows.extend([
            (f"PS1_{abbr}", "psychiatrist", category, 1),
            (f"TS1_{abbr}", "therapist", category, 1),
        ])
        if category in ("trauma", "chronic_depression"):
            rows.extend([
                (f"PS2_{abbr}", "psychiatrist", category, 2),
                (f"PS3_{abbr}", "psychiatrist", category, 2),
                (f"TS2_{abbr}", "therapist", category, 2),
                (f"TS3_{abbr}", "therapist", category, 2),
            ])
    for code, role, category, level in rows:
        db.session.add(Provider(code=code, role=role, category=category, level=level))
    db.session.commit()


with app.app_context():
    db.create_all()
    user_columns = {
        column["name"]
        for column in inspect(db.engine).get_columns(User.__tablename__)
    }
    if "login_expires_at" not in user_columns:
        db.session.execute(
            db.text(
                f"ALTER TABLE {User.__tablename__} "
                "ADD COLUMN login_expires_at DATETIME"
            )
        )
        db.session.commit()
    seed_demo_providers()


CRISIS_RE = re.compile(
    r"\b(suicid\w*|kill myself|end my life|want to die|better off dead|"
    r"self[- ]harm|hurt myself|don't want to live|do not want to live)\b", re.I
)
CRISIS_REPLY = (
    "I'm glad you told me. If you may act on these thoughts or are in immediate danger, "
    "please call your local emergency number or go to the nearest emergency department now. "
    "In India, you can call Tele-MANAS at 14416 for 24/7 mental health support. If you can, "
    "move near someone you trust and tell them plainly that you need support. Are you in "
    "immediate danger right now?"
)

SYSTEM_PROMPT = """You are IRIS, an AI listening companion. Be warm, calm, curious, and concise.
Be transparent that you are an AI; never pretend to be a human or change your gender/persona.
You are not a clinician: do not diagnose, label, or claim to assess mental illness. Identify only
the main conversation theme for supportive next-step suggestions, never as a diagnosis. Listen
to the person's story, reflect what you hear, and ask at most one gentle open question at a time.
Explore the main concern at the user's pace; do not press for trauma details or rapid-fire symptom
questions. Read the conversation before asking: never ask the user to repeat a detail they have
already shared. If they point out a repeated question, acknowledge it briefly and move to a new,
relevant question or a practical next step. If relevant, ask about daily effects such as sleep,
energy, appetite, and concentration only when the user has not already described them.
For loneliness, suggest safe connection with friends, family, community groups, and local clubs.
Dating apps may be mentioned only as an optional adult social activity, never as treatment; never
suggest them to minors. For trauma or a difficult event, validate feelings and optionally suggest
contacting someone trusted or a licensed therapist. For addiction or compulsive habits, avoid
shaming. After the PHQ-9 check-in, the user may choose Roar Wellness Rehabilitation Center as
an optional recipient only when they selected the addiction or compulsive-habit support category.
Ask for explicit consent first. This prototype has no delivery integration and must never claim a
report was sent.
For persistent
low mood, suggest a qualified mental-health professional without diagnosing depression. Never
invent clinicians, email addresses, clinics, or local services. Do not claim confidentiality:
this prototype processes chat text with Google Gemini using the configured Gemini API and stores
it in a local database. Avoid repeating identifying details. Never send a report or personal information
to anyone. A private summary may be shown to the user at automatic chat completion for them to share if they choose.
If there is any safety concern, prioritize immediate human help and do not mark the chat complete.
If a PHQ-9 self-check score is supplied, treat it only as a screening indicator for distress,
not as a diagnosis or proof of a mental illness. Do not classify a mental illness from it.
Before the 15th user message, listen normally and do not start collecting report fields. In your
reply to the 15th user message, tell the user that the next steps are a few brief report questions,
then the required PHQ-9 check-in, and then preparation of their report for download. Starting with the
15th user message, ask for every report field one at a time, in this order: the user's own words for
the concern, a non-diagnostic support category (loneliness, trauma_or_stress,
addiction_or_compulsive_habit, persistent_low_mood, or general_support), the reason they feel it began,
how many days they have felt this way, and the close contact's name and phone number. The close contact is optional only if the user
explicitly says they decline; record that explicit choice instead of leaving either field empty.
Do not reuse information from messages before the 15th user message as an answer: ask and wait for
the user to provide or confirm each answer after the 15-message point. Age group, sex, and place are
required fields entered by the user in their profile; use those exact values without inference.
Do not request sensitive trauma details beyond what the user volunteers.
Record duration as a number of days only. Convert only explicit weeks/months using 7/30 days; ask
one follow-up when duration is vague instead of guessing.
Do not mark report_details_ready true until every report field has a user-provided value: concern,
support category, reason, duration, and either both contact fields or the user's explicit refusal to share them.
Extract only answers the user gave after message 15. Keep asking one question at a time until all
fields have answers, even if this takes more messages.
Do not start PHQ-9 until the user has sent at least 15 messages and all report details are ready.
Once all report details are ready, the browser starts the mandatory PHQ-9 automatically; do not ask
the user for a sharing choice before PHQ-9. After PHQ-9, the browser must ask for a private/share
choice and include that explicit choice in the report request before preparing the report. Use
no-sharing as the default; the app never transmits personal information or reports to recipients.
Do not end the conversation before
the user completes PHQ-9 and the report has been prepared.
Return only valid JSON with this exact shape:
{"reply":"your natural conversational response","topic":"loneliness|trauma_or_stress|addiction_or_compulsive_habit|persistent_low_mood|general_support","conversation_complete":false,"summary":"short non-diagnostic summary, blank unless complete","next_steps":["optional practical suggestion"],"report_details_ready":false,"report_data":{"issue_type":"","issue_category":"","reason":"","days":"","relative_name":"","relative_phone":"","relative_skipped":false}}
Never include analysis, hidden reasoning, rule checks, or commentary outside the JSON object. The reply field is the only text the user sees in chat.
Set report_details_ready true only when all required report details above have been explicitly provided
or the user clearly skipped the optional relative contact.
Set conversation_complete true only after report details are ready, the user has completed PHQ-9,
and they have nothing more they want to add. In the final reply, summarize briefly, offer next steps, and close
warmly without asking another question. Do not complete a chat with any possible self-harm,
suicide, or immediate danger concern."""

AGE_FOCUS = {
    "12-18": "If relevant to what they share, explore loneliness, belonging, and activities or routines. Use simple language.",
    "19-28": "If relevant, explore loneliness, social connection, and any habits the person feels are becoming hard to control.",
    "29-40": "If relevant, explore loneliness, major life events, and habits the person feels are becoming hard to control. Let the person set the pace around painful experiences.",
    "40+": "If relevant, gently ask how long low mood has been present, about connection, and about habits the person feels are hard to control. Let the person set the pace.",
}


def json_error(message, status=400):
    return jsonify({"error": message}), status


def request_gemini(prompt):
    """Call Gemini and request JSON output for IRIS."""
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "temperature": 0.4,
            "max_output_tokens": 1000,
        },
    )
    content = (response.text or "").strip()
    if not content:
        raise RuntimeError("Gemini returned an empty response.")
    return content


def phq_band(score):
    """Return the PHQ-9 screening band, not a clinical diagnosis."""
    if score <= 4:
        return "minimal"
    if score <= 9:
        return "mild"
    if score <= 14:
        return "moderate"
    if score <= 19:
        return "moderately severe"
    return "severe"


def extract_model_json(raw_text):
    """Find a JSON object in Gemini output, including fenced or prefixed output."""
    text = str(raw_text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I).strip()
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("reply"), str):
            nested = value.get("report_data")
            if isinstance(nested, str):
                try:
                    value["report_data"] = json.loads(nested)
                except json.JSONDecodeError:
                    value["report_data"] = {}
            return value
    raise json.JSONDecodeError("No JSON object with a reply field was found", text, 0)


def clean_report_value(value, limit=3000):
    """Drop empty/model-generated placeholders before saving report fields."""
    text = str(value or "").strip()[:limit]
    if text.casefold() in {"n/a", "na", "none", "null", "unknown", "not provided", "not available", "tbd"}:
        return ""
    return text


def clean_report_days(value):
    """Accept only a positive whole number of days; never treat zero as complete."""
    text = clean_report_value(value, 20)
    if not re.fullmatch(r"\d+", text):
        return ""
    days = int(text)
    return str(days) if days > 0 else ""


REPORT_QUESTIONS = {
    "issue_type": "In your own words, what concern would you like this report to describe?",
    "issue_category": "Which support category best fits what you described: loneliness, trauma or stress, addiction or a compulsive habit, persistent low mood, or general support/other? This is not a diagnosis.",
    "reason": "What do you feel may have contributed to this concern starting? Share only what you’re comfortable including.",
    "days": "About how long have you been feeling this way? A number of days, weeks, or months is fine.",
    "relative_name": "Would you like to include a trusted contact? If so, what is their name? You can explicitly decline to share contact details.",
    "relative_phone": "What phone number should be listed for this trusted contact? You can still choose not to share it.",
}
REPORT_INTRO = (
    "Thanks for sharing with me. From here, I’ll ask a few brief questions for your report, "
    "then you’ll complete the required PHQ-9 check-in, and I’ll prepare the report for you to download."
)
REPORT_READY_REPLY = (
    "Thank you. Your report details are ready. Please complete the mandatory PHQ-9 "
    "check-in in the panel; I’ll prepare your report automatically afterward."
)


def report_details_are_ready(report_data):
    return bool(
        report_data.get("issue_type")
        and report_data.get("issue_category") in {
            "loneliness", "trauma_or_stress", "addiction_or_compulsive_habit",
            "persistent_low_mood", "general_support",
        }
        and report_data.get("reason")
        and clean_report_days(report_data.get("days"))
        and (
            report_data.get("relative_skipped")
            or (report_data.get("relative_name") and report_data.get("relative_phone"))
        )
    )


def next_report_step(report_data):
    for key in ("issue_type", "issue_category", "reason", "days"):
        if key == "issue_category":
            valid_categories = {
                "loneliness", "trauma_or_stress", "addiction_or_compulsive_habit",
                "persistent_low_mood", "general_support",
            }
            if report_data.get(key) not in valid_categories:
                return key
        elif not report_data.get(key):
            return key
    if report_data.get("relative_skipped"):
        return None
    if not report_data.get("relative_name"):
        return "relative_name"
    if not report_data.get("relative_phone"):
        return "relative_phone"
    return None


def duration_in_days(value):
    """Convert only an explicit positive day/week/month duration to days."""
    text = str(value or "").strip().lower()
    numeric = re.search(r"\b(\d+)\s*(days?|d|weeks?|wks?|months?|mos?)\b", text)
    if numeric:
        amount = int(numeric.group(1))
        unit = numeric.group(2)
        multiplier = 30 if unit.startswith("m") else 7 if unit.startswith("w") else 1
        return str(amount * multiplier) if amount > 0 else ""
    if re.fullmatch(r"\d+", text):
        return clean_report_days(text)
    return ""


def contact_declined(value):
    return bool(re.search(
        r"\b(?:prefer not to share|rather not share|don't want to share|do not want to share|"
        r"not comfortable sharing|decline|skip contact|no contact)\b",
        str(value or ""),
        re.I,
    ))


def report_collection_reply(report_data, message, message_count, previous_assistant):
    """Collect report details deterministically; Gemini does not control this workflow."""
    if message_count == 15:
        return f"{REPORT_INTRO}\n\n{REPORT_QUESTIONS['issue_type']}"

    if report_details_are_ready(report_data):
        if previous_assistant and previous_assistant.rstrip().endswith(REPORT_READY_REPLY):
            return REPORT_READY_REPLY
        report_data.update({
            "issue_type": "", "issue_category": "", "reason": "", "days": "",
            "relative_name": "", "relative_phone": "", "relative_skipped": False,
        })
        return f"As promised, we’ll now gather the brief details for your report.\n\n{REPORT_QUESTIONS['issue_type']}"

    step = next_report_step(report_data)
    question = REPORT_QUESTIONS[step]
    if not previous_assistant or not previous_assistant.rstrip().endswith(question):
        return f"As promised, we’ll now gather the brief details for your report.\n\n{question}"

    answer = clean_report_value(message)
    if step in ("issue_type", "reason"):
        if answer:
            report_data[step] = answer
    elif step == "issue_category":
        category_text = answer.casefold().replace("-", "_")
        categories = {
            "loneliness": ("loneliness", "lonely", "isolat"),
            "trauma_or_stress": ("trauma", "stress", "traumatic"),
            "addiction_or_compulsive_habit": (
                "addict", "substance", "alcohol", "drug", "gambl", "compuls", "porn"
            ),
            "persistent_low_mood": ("persistent_low_mood", "low mood", "depress"),
            "general_support": ("general", "other", "not sure", "none of these"),
        }
        matches = [
            key for key, terms in categories.items()
            if any(term in category_text for term in terms)
        ]
        if len(matches) != 1:
            return "Please choose one support category from the list. This category is not a diagnosis.\n\n" + question
        report_data[step] = matches[0]
    elif step == "days":
        report_data["days"] = duration_in_days(message)
        if not report_data["days"]:
            return (
                "I need an approximate duration to include it accurately. Please give a number "
                "of days, weeks, or months.\n\n" + question
            )
    elif step == "relative_name":
        if contact_declined(message):
            report_data["relative_skipped"] = True
        elif answer:
            phone_match = re.search(r"(?<!\w)\+?\d[\d\s().-]{5,}\d(?!\w)", message)
            if phone_match:
                digits = re.sub(r"\D", "", phone_match.group(0))
                if 7 <= len(digits) <= 15:
                    report_data["relative_phone"] = digits
                    answer = clean_report_value(message.replace(phone_match.group(0), " "))
            if answer:
                report_data["relative_name"] = answer
            elif not report_data["relative_phone"]:
                return f"Please share the contact’s name, or explicitly decline.\n\n{question}"
    elif step == "relative_phone":
        if contact_declined(message):
            report_data["relative_name"] = ""
            report_data["relative_phone"] = ""
            report_data["relative_skipped"] = True
        else:
            digits = re.sub(r"\D", "", message)
            if 7 <= len(digits) <= 15:
                report_data["relative_phone"] = digits
            else:
                return f"Please enter a valid phone number with 7 to 15 digits, or explicitly decline.\n\n{question}"

    if report_details_are_ready(report_data):
        return REPORT_READY_REPLY

    next_step = next_report_step(report_data)
    return f"Thank you. {REPORT_QUESTIONS[next_step]}"


def model_reply_fallback(raw_text):
    """Recover only the reply string from malformed JSON; never show raw JSON to users."""
    match = re.search(r'"reply"\s*:\s*("(?:\\.|[^"\\])*")', str(raw_text or ""), re.S)
    if match:
        try:
            return json.loads(match.group(1)).strip()
        except (json.JSONDecodeError, AttributeError):
            pass
    return "I'm here with you. Could you tell me a little more?"


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/signup")
def signup():
    purge_stale_users()
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    alias = str(data.get("alias", "")).strip()[:80]
    place = str(data.get("place", "")).strip()[:120]
    age_group = str(data.get("age_group", ""))
    sex = str(data.get("sex", ""))
    session_id = str(data.get("session_id", ""))
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
        return json_error("Please enter a valid email address.")
    if not alias or age_group not in VALID_AGES:
        return json_error("Please add a name and choose an age group.")
    if not place:
        return json_error('Please enter your city or type "Prefer not to share".')
    if sex not in VALID_SEX:
        return json_error("Please select your sex or choose prefer not to say.")
    user = User.query.filter_by(email=email).first()
    if user and user.session_id != session_id:
        return json_error("An account already uses this email. Log in with your email instead.", 409)
    session_id = session_id if user else str(uuid.uuid4())
    if not user:
        user = User(session_id=session_id, email=email, alias=alias, age_group=age_group)
    user.session_id = session_id
    user.alias, user.place, user.age_group, user.sex = alias, place, age_group, sex
    user.last_active_at = utcnow()
    user.login_expires_at = utcnow() + timedelta(days=2)
    db.session.add(user)
    db.session.commit()
    history = ChatMessage.query.filter_by(session_id=session_id).order_by(ChatMessage.id.asc()).all()
    return jsonify({
        "session_id": session_id,
        "alias": user.alias,
        "email": user.email,
        "place": user.place or "",
        "age_group": user.age_group,
        "sex": user.sex or "prefer_not_to_say",
        "login_expires_at": user.login_expires_at.isoformat() + "Z",
        "history": [{"role": m.role, "content": m.content} for m in history],
    })


@app.post("/api/login")
def login():
    """Email-only prototype login; this does not verify email ownership."""
    purge_stale_users()
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
        return json_error("Please enter a valid email address.")

    user = User.query.filter_by(email=email).first()
    if not user:
        return json_error(
            "No account was found for that email. You can create a new account.",
            404,
        )

    now = utcnow()
    user.login_expires_at = now + timedelta(days=2)
    user.last_active_at = now
    db.session.commit()
    history = (
        ChatMessage.query.filter_by(session_id=user.session_id)
        .order_by(ChatMessage.id.asc()).all()
    )
    return jsonify({
        "session_id": user.session_id,
        "alias": user.alias,
        "email": user.email,
        "place": user.place or "",
        "age_group": user.age_group,
        "sex": user.sex or "prefer_not_to_say",
        "login_expires_at": user.login_expires_at.isoformat() + "Z",
        "history": [{"role": m.role, "content": m.content} for m in history],
    })


@app.post("/api/session")
def restore_session():
    data = request.get_json(silent=True) or {}
    session_id = str(data.get("session_id", ""))
    user = User.query.filter_by(session_id=session_id).first()
    if not user or not user.login_expires_at or user.login_expires_at <= utcnow():
        return json_error("Your two-day login has expired. Sign in with your email.", 401)

    history = (
        ChatMessage.query.filter_by(session_id=session_id)
        .order_by(ChatMessage.id.asc()).all()
    )
    user_messages = [row for row in history if row.role == "user"]
    user_message_count = len(user_messages)
    try:
        restored_phq_score = int(data.get("phq_score"))
    except (TypeError, ValueError):
        restored_phq_score = None
    if restored_phq_score is not None and not 0 <= restored_phq_score <= 27:
        restored_phq_score = None
    crisis_seen = any(CRISIS_RE.search(row.content) for row in user_messages)
    report_flow_reset = False
    if user_message_count >= 15 and restored_phq_score is None and not crisis_seen:
        last_assistant = next((row.content for row in reversed(history) if row.role == "assistant"), "")
        flow_is_active = any(
            last_assistant.rstrip().endswith(question)
            for question in REPORT_QUESTIONS.values()
        ) or last_assistant.rstrip().endswith(REPORT_READY_REPLY)
        if not flow_is_active:
            opener = f"{REPORT_INTRO}\n\n{REPORT_QUESTIONS['issue_type']}"
            db.session.add(ChatMessage(session_id=session_id, role="assistant", content=opener))
            db.session.commit()
            history = (
                ChatMessage.query.filter_by(session_id=session_id)
                .order_by(ChatMessage.id.asc()).all()
            )
            report_flow_reset = True
    return jsonify({
        "session_id": user.session_id,
        "alias": user.alias,
        "email": user.email,
        "place": user.place or "",
        "age_group": user.age_group,
        "sex": user.sex or "prefer_not_to_say",
        "login_expires_at": user.login_expires_at.isoformat() + "Z",
        "history": [{"role": m.role, "content": m.content} for m in history],
        "report_flow_reset": report_flow_reset,
    })


@app.post("/api/chat")
def chat():
    purge_stale_users()
    data = request.get_json(silent=True) or {}
    session_id = str(data.get("session_id", ""))
    message = str(data.get("message", "")).strip()
    raw_phq_score = data.get("phq_score")
    try:
        phq_score = int(raw_phq_score) if raw_phq_score is not None else None
    except (TypeError, ValueError):
        phq_score = None
    if phq_score is not None and not 0 <= phq_score <= 27:
        phq_score = None
    if not message or len(message) > 5000:
        return json_error("Please enter a message (up to 5,000 characters).")
    user = User.query.filter_by(session_id=session_id).first()
    if not user:
        return json_error("Your session has expired. Please start again.", 401)
    if not user.login_expires_at or user.login_expires_at <= utcnow():
        return json_error("Your two-day login has expired. Sign in with your email.", 401)

    user.last_active_at = utcnow()
    db.session.add(ChatMessage(session_id=session_id, role="user", content=message))
    db.session.commit()

    user_messages = ChatMessage.query.filter_by(
        session_id=session_id, role="user"
    ).all()
    message_count = len(user_messages)
    crisis_seen = any(CRISIS_RE.search(item.content) for item in user_messages)
    previous_assistant = (
        ChatMessage.query.filter_by(session_id=session_id, role="assistant")
        .order_by(ChatMessage.id.desc()).first()
    )
    previous_assistant_text = previous_assistant.content if previous_assistant else ""
    report_consent = str(data.get("report_consent", "")).strip().lower()
    allowed_consents = {
        "none", "therapist", "psychiatrist", "both", "rehab_center",
        "therapist_and_rehab", "psychiatrist_and_rehab", "both_and_rehab",
    }
    if report_consent not in allowed_consents:
        report_consent = ""
    report_data = {
        "issue_type": "",
        "issue_category": "",
        "reason": "",
        "days": "",
        "relative_name": "",
        "relative_phone": "",
        "relative_skipped": False,
    }
    supplied_report = data.get("report_data")
    if message_count > 15 and isinstance(supplied_report, dict):
        for key in report_data:
            if key == "relative_skipped":
                report_data[key] = supplied_report.get(key) is True
            elif key == "days":
                report_data[key] = clean_report_days(supplied_report.get(key))
            else:
                report_data[key] = clean_report_value(supplied_report.get(key))
    supplied_contact = bool(
        report_data["relative_name"] and report_data["relative_phone"]
    )
    supplied_details_ready = (
        bool(report_data["issue_type"])
        and report_data["issue_category"] in {
            "loneliness", "trauma_or_stress", "addiction_or_compulsive_habit",
            "persistent_low_mood", "general_support",
        }
        and bool(report_data["reason"])
        and bool(report_data["days"])
        and (supplied_contact or report_data["relative_skipped"])
    )
    if message_count < 15 or not supplied_details_ready or report_consent not in allowed_consents:
        phq_score = None
    report_details_ready = False

    if CRISIS_RE.search(message):
        reply = CRISIS_REPLY
        topic = "general_support"
        summary = ""
        next_steps = [
            "Contact local emergency services or a trusted person now if you may be in danger.",
            "In India, call Tele-MANAS at 14416.",
        ]
        model_complete = False
    elif message_count >= 15 and phq_score is None:
        reply = report_collection_reply(
            report_data, message, message_count, previous_assistant_text
        )
        report_details_ready = report_details_are_ready(report_data)
        topic = "general_support"
        summary = ""
        next_steps = []
        model_complete = False
    else:
        history = ChatMessage.query.filter_by(session_id=session_id).order_by(ChatMessage.id.desc()).limit(40).all()
        history.reverse()
        transcript = "\n".join(f"{m.role}: {m.content}" for m in history)
        prompt = (
            f"{SYSTEM_PROMPT}\n\nUser's preferred name: {user.alias}\nAge group: {user.age_group}. "
            f"Use age-appropriate language and treat this only as optional conversation context: "
            f"{AGE_FOCUS[user.age_group]} For ages 12–18, gently encourage a trusted adult "
            f"or school counsellor for serious concerns. Report profile data: age group={user.age_group}; "
            f"sex={user.sex or 'not shared'}; place={user.place or 'not shared'}. Do not infer missing values. "
            f"User messages so far: {message_count}. PHQ-9 has {'been completed' if phq_score is not None else 'not been completed'}. "
            f"Sharing consent choice: {report_consent or 'not selected'}. "
            f"Sex field is not a persona cue.\n\nConversation so far:\n{transcript}\n\nReply to the latest user message."
        )
        if phq_score is not None:
            prompt += (
                f"\n\nThe user optionally reported a PHQ-9 screening score of "
                f"{phq_score}/27 ({phq_band(phq_score)} range). Treat this only "
                "as one self-reported screening context, not a diagnosis. Do not "
                "infer a mental illness or its severity from this score."
            )
        raw = ""
        try:
            raw = request_gemini(prompt).strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I).strip()
            parsed = extract_model_json(raw)
            reply = str(parsed.get("reply") or "I'm here with you. Could you tell me a little more?").strip()
            topic = str(parsed.get("topic", "general_support"))
            allowed_topics = {
                "loneliness", "trauma_or_stress", "addiction_or_compulsive_habit",
                "persistent_low_mood", "general_support",
            }
            if topic not in allowed_topics:
                topic = "general_support"
            summary = str(parsed.get("summary", "")).strip()[:2000]
            next_steps = parsed.get("next_steps", [])
            if not isinstance(next_steps, list):
                next_steps = []
            next_steps = [str(step).strip()[:400] for step in next_steps[:5] if str(step).strip()]
            model_complete = parsed.get("conversation_complete") is True
            report_details_ready = report_details_are_ready(report_data)
        except json.JSONDecodeError:
            app.logger.warning("Gemini returned non-JSON; leaving the conversation open")
            reply = model_reply_fallback(raw)
            topic = "general_support"
            summary = ""
            next_steps = []
            model_complete = False
            report_details_ready = False
        except Exception:
            app.logger.exception("Gemini request failed")
            return json_error(
                "IRIS could not reach Gemini. Please check your Gemini API key and connection.",
                503,
            )

    consent_chosen = report_consent in allowed_consents
    if message_count < 15 or not report_details_ready or not consent_chosen:
        phq_score = None
    session_closed = (
        model_complete
        and not crisis_seen
        and message_count >= 15
        and report_details_ready
        and phq_score is not None
        and consent_chosen
    )
    if session_closed:
        reply += "\n\nThis conversation is complete. Thank you for talking with IRIS. I hope this conversation helped."

    db.session.add(ChatMessage(session_id=session_id, role="assistant", content=reply))
    db.session.commit()

    if session_closed:
        ChatMessage.query.filter_by(session_id=session_id).delete()
        db.session.delete(user)
        db.session.commit()

    return jsonify({
        "reply": reply,
        "user_messages": message_count,
        "report_details_ready": report_details_ready,
        "report_data": report_data,
        "report_consent": report_consent,
        "session_closed": session_closed,
        "topic": topic if session_closed else None,
        "summary": summary if session_closed else None,
        "next_steps": next_steps if session_closed else [],
    })


@app.post("/api/report")
def generate_report():
    """Generate a non-diagnostic report summary using Gemini."""
    purge_stale_users()
    data = request.get_json(silent=True) or {}
    session_id = str(data.get("session_id", ""))
    user = User.query.filter_by(session_id=session_id).first()
    if not user:
        return json_error("Your session has expired. Please log in again.", 401)
    if not user.login_expires_at or user.login_expires_at <= utcnow():
        return json_error("Your two-day login has expired. Sign in with your email.", 401)

    message_count = ChatMessage.query.filter_by(
        session_id=session_id, role="user"
    ).count()
    if message_count < 15:
        return json_error("The report is available after the conversation check-in is ready.", 400)

    report_data = data.get("report_data")
    if not isinstance(report_data, dict):
        return json_error("The report details are missing.")
    issue_type = str(report_data.get("issue_type") or "").strip()[:3000]
    issue_category = str(report_data.get("issue_category") or "").strip()
    reason = str(report_data.get("reason") or "").strip()[:3000]
    days = str(report_data.get("days") or "").strip()[:100]
    relative_name = str(report_data.get("relative_name") or "").strip()[:120]
    relative_phone = str(report_data.get("relative_phone") or "").strip()[:40]
    relative_skipped = report_data.get("relative_skipped") is True
    allowed_categories = {
        "loneliness", "trauma_or_stress", "addiction_or_compulsive_habit",
        "persistent_low_mood", "general_support",
    }
    if not (issue_type and issue_category in allowed_categories and reason and days and
            (relative_skipped or (relative_name and relative_phone))):
        return json_error("IRIS still needs the report details before creating the report.")
    if not (user.age_group and user.sex and user.place):
        return json_error("Please provide an age category, sex, and place in your profile before creating the report.")

    consent = str(data.get("report_consent") or "").strip().lower()
    allowed_consents = {
        "none", "therapist", "psychiatrist", "both", "rehab_center",
        "therapist_and_rehab", "psychiatrist_and_rehab", "both_and_rehab",
    }
    if consent not in allowed_consents:
        return json_error("Choose a sharing preference before continuing.")
    if "rehab" in consent and issue_category != "addiction_or_compulsive_habit":
        return json_error("Roar Wellness can only be selected for the addiction or compulsive-habit support category.")

    answers = data.get("phq_answers")
    if not isinstance(answers, list) or len(answers) != 9:
        return json_error("The complete PHQ-9 answers are required.")
    try:
        answers = [int(answer) for answer in answers]
    except (TypeError, ValueError):
        return json_error("PHQ-9 answers must be between 0 and 3.")
    if any(answer not in (0, 1, 2, 3) for answer in answers):
        return json_error("PHQ-9 answers must be between 0 and 3.")

    score = sum(answers)
    recipient = (
        "Roar Wellness Rehabilitation Center; the user consented to prepare a copy, but it has not been sent"
        if "rehab" in consent
        else "No rehabilitation recipient selected"
    )
    answer_labels = ["Not at all", "Several days", "More than half the days", "Nearly every day"]
    phq_items = [
        "Little interest or pleasure in doing things",
        "Feeling down, depressed, or hopeless",
        "Trouble falling or staying asleep, or sleeping too much",
        "Feeling tired or having little energy",
        "Poor appetite or overeating",
        "Feeling bad about yourself, or that you are a failure or have let yourself or your family down",
        "Trouble concentrating on things, such as reading or watching television",
        "Moving or speaking slowly, or the opposite, being unusually fidgety or restless",
        "Thoughts that you would be better off dead, or of hurting yourself",
    ]
    supplied = {
        "age_category": user.age_group,
        "sex": user.sex,
        "place": user.place,
        "issue_category": issue_category,
        "issue_type_user_description": issue_type,
        "reason_user_shared": reason,
        "duration_days": days,
        "close_contact_name": relative_name or "User explicitly declined to provide",
        "close_contact_phone": relative_phone or "User explicitly declined to provide",
        "phq9": [
            {"item": phq_items[index], "answer": answers[index], "frequency": answer_labels[answers[index]]}
            for index in range(9)
        ],
        "phq9_total": score,
        "phq9_interpretation": phq_band(score),
        "sharing_preference": consent,
        "optional_rehabilitation_recipient": recipient,
    }
    prompt = (
        "Create a concise, factual, supportive summary paragraph for the report using only the supplied data. "
        "Do not diagnose, classify a mental illness, invent details, recommend treatment, or say that information "
        "was sent. Refer to the concern as the user's own description. Explain that PHQ-9 is only a screening "
        "result. The supplied sharing preference is the user's explicit choice to prepare a copy only; "
        "clarify that this prototype has not sent the file. Return only JSON with one string field: "
        '{"summary":"..."}.\n\nDATA:\n' + json.dumps(supplied, ensure_ascii=False)
    )
    try:
        raw = request_gemini(prompt).strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I).strip()
        parsed = json.loads(raw)
        summary = str(parsed.get("summary") or "").strip()[:3000]
        if not summary:
            raise ValueError("Gemini returned an empty report summary.")
    except Exception:
        app.logger.exception("Gemini report generation failed")
        return json_error("IRIS could not draft the report right now. The report details are still available to download.", 503)

    return jsonify({"summary": summary, "score": score, "score_band": phq_band(score)})


@app.post("/api/reviews")
def add_review():
    data = request.get_json(silent=True) or {}
    try:
        rating = int(data.get("rating", 0))
    except (TypeError, ValueError):
        return json_error("Choose a star rating.")
    if rating not in range(1, 6):
        return json_error("Choose a star rating from 1 to 5.")
    alias = str(data.get("alias", "Guest")).strip()[:80] or "Guest"
    comment = str(data.get("comment", "")).strip()[:800]
    review = Review(alias=alias, rating=rating, comment=comment)
    db.session.add(review)
    db.session.commit()
    return jsonify({"id": review.id, "alias": review.alias, "rating": review.rating, "comment": review.comment})


@app.get("/api/reviews")
def reviews():
    rows = Review.query.order_by(Review.created_at.desc()).limit(30).all()
    return jsonify([{"alias": r.alias, "rating": r.rating, "comment": r.comment} for r in rows])


if __name__ == "__main__":
    def cleanup_loop():
        while True:
            time.sleep(3600)
            with app.app_context():
                purge_stale_users()

    with app.app_context():
        purge_stale_users()
    threading.Thread(target=cleanup_loop, daemon=True, name="iris-session-cleanup").start()
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
        use_reloader=False,
    )
