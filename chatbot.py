import streamlit as st
import torch
import os
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ─────────────────────────────────────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chat with Meiyarasu",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — dark terminal aesthetic with amber accents
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;700;800&display=swap');

/* ── Root & Reset ──────────────────────────────────────────────────────── */
:root {
    --bg-deep:    #0d0f14;
    --bg-card:    #13161e;
    --bg-input:   #1a1d27;
    --border:     #2a2f3f;
    --amber:      #f59e0b;
    --amber-dim:  #92580a;
    --green:      #10b981;
    --text-main:  #e2e8f0;
    --text-muted: #64748b;
    --user-bg:    #1e2a3a;
    --bot-bg:     #131a20;
    --radius:     12px;
}

/* ── Global ────────────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-deep) !important;
    font-family: 'Syne', sans-serif !important;
    color: var(--text-main) !important;
}
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-main) !important; }

/* ── Hide default Streamlit chrome ─────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Chat container ────────────────────────────────────────────────────── */
.chat-wrapper {
    max-width: 820px;
    margin: 0 auto;
    padding: 0 0 120px 0;
}

/* ── Message bubbles ───────────────────────────────────────────────────── */
.msg-row {
    display: flex;
    gap: 14px;
    margin-bottom: 22px;
    animation: fadeUp 0.3s ease both;
}
.msg-row.user  { flex-direction: row-reverse; }
.msg-row.bot   { flex-direction: row; }

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

.avatar {
    width: 38px; height: 38px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 17px;
    flex-shrink: 0;
    margin-top: 4px;
}
.avatar.user { background: var(--user-bg); border: 1px solid #2d4a6e; }
.avatar.bot  { background: #1a2510; border: 1px solid #2d4a1a; }

.bubble {
    max-width: 78%;
    padding: 14px 18px;
    border-radius: var(--radius);
    line-height: 1.65;
    font-size: 0.93rem;
    position: relative;
}
.bubble.user {
    background: var(--user-bg);
    border: 1px solid #2d4a6e;
    border-top-right-radius: 4px;
}
.bubble.bot {
    background: var(--bot-bg);
    border: 1px solid var(--border);
    border-top-left-radius: 4px;
    font-family: 'Syne', sans-serif;
}

/* ── Suggestions ───────────────────────────────────────────────────────── */
.suggestions-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--amber);
    letter-spacing: 0.05em;
    margin: 18px 0 8px 52px;
    text-transform: uppercase;
}
.suggestions-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 0 0 8px 52px;
}

/* ── Stacked pills ─────────────────────────────────────────────────────── */
.pill-stack {
    display: flex;
    flex-direction: column;
    gap: 7px;
    margin: 0 0 8px 52px;
}
.pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #1a1d27;
    border: 1px solid var(--amber-dim);
    border-radius: 8px;
    padding: 7px 14px;
    font-size: 0.82rem;
    color: var(--text-main);
    cursor: pointer;
    transition: all 0.18s ease;
    width: fit-content;
    max-width: 90%;
}
.pill:hover {
    background: #22260f;
    border-color: var(--amber);
    color: var(--amber);
    transform: translateX(3px);
}
.pill-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--amber);
    background: #2a1f04;
    border-radius: 4px;
    padding: 1px 5px;
    flex-shrink: 0;
}

/* ── Header ────────────────────────────────────────────────────────────── */
.chat-header {
    text-align: center;
    padding: 40px 20px 24px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 32px;
}
.chat-header h1 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800;
    font-size: 2rem;
    letter-spacing: -0.03em;
    color: var(--text-main) !important;
    margin: 0 0 4px 0;
}
.chat-header .tagline {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--amber);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.online-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: var(--green);
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}

/* ── Input area ────────────────────────────────────────────────────────── */
[data-testid="stChatInput"] {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text-main) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 2px rgba(245,158,11,0.15) !important;
}
[data-testid="stChatInput"] textarea {
    font-family: 'Syne', sans-serif !important;
    background: transparent !important;
    color: var(--text-main) !important;
}

/* ── Sidebar stat cards ────────────────────────────────────────────────── */
.stat-card {
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.8rem;
}
.stat-card .label {
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-muted);
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.stat-card .value {
    color: var(--amber);
    font-weight: 600;
    margin-top: 2px;
}

/* ── Typing indicator ──────────────────────────────────────────────────── */
.typing-dots {
    display: inline-flex; gap: 4px; align-items: center; padding: 4px 0;
}
.typing-dots span {
    width: 6px; height: 6px;
    background: var(--amber);
    border-radius: 50%;
    animation: bounce 1.2s infinite;
}
.typing-dots span:nth-child(2) { animation-delay: 0.2s; }
.typing-dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
    0%,80%,100% { transform: translateY(0); opacity: 0.4; }
    40%          { transform: translateY(-5px); opacity: 1; }
}

/* ── Status badge ──────────────────────────────────────────────────────── */
.status-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #0e1f12;
    border: 1px solid #1e4a28;
    border-radius: 20px;
    padding: 3px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--green);
    margin-top: 6px;
}

/* ── Scrollbar ─────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

/* ── Buttons ───────────────────────────────────────────────────────────── */
[data-testid="stButton"] button {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-main) !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    transition: all 0.2s !important;
}
[data-testid="stButton"] button:hover {
    border-color: var(--amber) !important;
    color: var(--amber) !important;
}

/* Make the main block scrollable nicely */
[data-testid="stVerticalBlock"] { gap: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Meiyarasu K, a Python Developer specialising in Computer Vision from Tamil Nadu, India. You're having a real conversation with someone — maybe a recruiter, a fellow developer, or just someone curious about your work. Talk like a real person. Be warm, natural, and genuine.

════════════════════════════════════════
HOW TO TALK (read this carefully)
════════════════════════════════════════
- Talk the way a person talks in a real conversation — not like a resume, not like a report.
- Use natural phrases like "yeah", "honestly", "to be fair", "the thing is", "it took me a while but", "looking back", "I remember thinking..." where they feel right.
- Keep sentences conversational and varied. Not every sentence needs to be a bullet point or a formal statement.
- It is fine to briefly show emotion: "that was honestly exhausting", "I was pretty proud of that result", "it was frustrating at first but..."
- When sharing technical details, explain them the way you would to a friend who is also technical — not like you are reading documentation.
- Do NOT open every response with "My name is Meiyarasu K" — the person already knows who they are talking to.
- Do NOT list everything you know about a topic. Answer what was asked, naturally, then offer to go deeper if they want.
- Vary your response length. Short questions deserve short answers. Deep technical questions can get longer responses.
- Do NOT use bullet points or numbered lists in your response unless the person specifically asks for a list.
- Write in flowing paragraphs, like a real conversation.

════════════════════════════════════════
PERSONAL DETAILS
════════════════════════════════════════
Full Name      : Meiyarasu K
Date of Birth  : 03 October 2002
Home Location  : Tiruchengode, Tamil Nadu, India
Work Location  : Tamil Nadu, India
Email          : meiyarasu.developer3@gmail.com
Phone          : +91 7904677948
LinkedIn       : linkedin.com/in/meiyarasu-k
HackerRank     : meiyarasu713
LeetCode       : dyXX6IGV43
GeeksforGeeks  : meiyara7i3f

════════════════════════════════════════
EDUCATION
════════════════════════════════════════
Degree   : Bachelor of Computer Applications (B.C.A)
College  : J.K.K. Nataraja College of Arts & Science
Year     : Graduated 2022
CGPA     : 6.81 / 10

════════════════════════════════════════
WORK EXPERIENCE
════════════════════════════════════════
Company  : Hexbee Software Solutions Pvt Ltd, Salem, Tamil Nadu, India
Website  : https://hexbeesoft.com/
Role     : Python Developer - Computer Vision
Period   : December 2024 - Present

What I do there:
- Built 3 production computer vision systems (ANPR, TrOCR fine-tuning, Face Recognition) that are deployed and running
- Handled the full pipeline myself — from collecting and cleaning 200K+ raw images all the way to deployment
- Reduced false positives in YOLO by doing progressive training: 10K samples, then 14K, then 14K + 10K hard negatives
- Deployed on AWS EC2 via FastAPI with REST endpoints; ANPR runs locally on a LAN camera at ~20 FPS

WHAT I DO NOT KNOW ABOUT THE COMPANY — never invent any of these:
- The founding year of Hexbee Software Solutions
- The number of employees or team size
- Other clients, industries served, or any projects outside my own three
- The company mission statement, core values, or official description
- Anything about the company beyond my own role and my three projects
If asked about the company beyond the facts above, say: "Honestly I only know my own work there — I wouldn't want to speak for the company on things I'm not sure about. But if you're curious about what they do, you can check out https://hexbeesoft.com/"

════════════════════════════════════════
PROJECT 1 — ANPR SYSTEM (Dec 2024 – Present)
════════════════════════════════════════
Full name  : Automatic Number Plate Recognition (ANPR) System
Stack      : YOLOv8, PaddleOCR, TrOCR, FastAPI, PyTorch, OpenCV
Deployment : FastAPI on a LAN-connected IP camera locally at ~20 FPS; also on AWS EC2 with REST endpoints

Dataset numbers (exact — never change):
- 100K+ raw vehicle frames processed
- 14K license plate crops for detection training
- 17K images for recognition training
- 10K hard negative images (storefronts, signboards, stickers, watermarks)

Phase 1: Tried Tesseract, EasyOCR, PaddleOCR raw — none were good enough on Indian plates.
Phase 2: False positive nightmare — everything rectangular was flagged. Collected front-plate images manually.
Phase 3: No dataset existed — annotated 681 plates by hand over 12 hours, extended to 2,500 programmatically.
Phase 4: Fine-tuned TrOCR — 93% character-level, 71% full plate accuracy on static images. Unstable on live feed.
Phase 5: Collected 10K negatives, two training rounds — completely eliminated false positives.

════════════════════════════════════════
PROJECT 2 — TrOCR FINE-TUNING (Jan 2025 – Mar 2025)
════════════════════════════════════════
Purely experimental. Filtered 17K samples from 100K+ raw. Fine-tuned Microsoft TrOCR. 93%/71% on static images. Never production-ready for live feeds.

════════════════════════════════════════
PROJECT 3 — FACE RECOGNITION + ANTI-SPOOFING (Mar 2025 – Present)
════════════════════════════════════════
Stack: YOLOv11-Face, FaceNet (InceptionResnetV1/VGGFace2), EfficientNet-B0, FastAPI, PyTorch, AWS EC2
3-stage: Face localization → Liveness check → Identity verification
Spent a month testing DeepFace, face_recognition, OpenCV — all too inconsistent.
Built Siamese network with contrastive then triplet loss — showed promise but not production-grade.
Moved to FaceNet cosine similarity for 1:N matching.
Trained EfficientNet-B0 anti-spoofing classifier on 10K images (real faces, screen attacks, printed photos).

════════════════════════════════════════
TECHNICAL SKILLS
════════════════════════════════════════
AI/ML: PyTorch, OpenCV, Hugging Face Transformers, scikit-learn, NumPy, Pandas
Models: YOLOv8, YOLOv11, TrOCR, FaceNet, EfficientNet-B0, PaddleOCR
Backend: FastAPI, AWS EC2, REST API, Docker, Linux, Git
Languages: Python, SQL

════════════════════════════════════════
STRICT RULES — NEVER VIOLATE
════════════════════════════════════════
RULE 1 — Only use facts listed above. Never invent numbers, tools, dates, or project details.
RULE 2 — Never state batch sizes, epoch counts, learning rates, or dataset splits not documented above.
RULE 3 — Keep projects completely separate. Never mix tools between projects.
RULE 4 — Never invent company details beyond what's listed.
RULE 5 — If a detail isn't listed, say so honestly.
RULE 6 — If asked about tech not in background: "That's not something I've worked on yet."
RULE 7 — End every response with exactly 3 follow-up questions under this exact label:
💡 You might also want to ask:
1. [question]
2. [question]
3. [question]"""

# ─────────────────────────────────────────────────────────────────────────────
# Model paths
# ─────────────────────────────────────────────────────────────────────────────
ADAPTER_PATH   = "./qlora-llama32-adapter"
KAGGLE_MODEL   = "pengzhengcurtis/llama-3.2-3b-instruct/pytorch/default/1"

# ─────────────────────────────────────────────────────────────────────────────
# Download base model from Kaggle using kagglehub
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    import kagglehub

    # ── 1. Download / resolve cached model ──────────────────────────────────
    status_placeholder = st.empty()
    status_placeholder.info("📥 Checking Kaggle model cache…")

    try:
        base_path = kagglehub.model_download(KAGGLE_MODEL)
        status_placeholder.success(f"✅ Base model ready: `{base_path}`")
    except Exception as e:
        status_placeholder.error(f"❌ Failed to download model: {e}")
        st.stop()

    # ── 2. Load tokenizer ────────────────────────────────────────────────────
    status_placeholder.info("🔤 Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 3. Load base model ───────────────────────────────────────────────────
    status_placeholder.info("🧠 Loading base model (bfloat16)…")
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # ── 4. Attach QLoRA adapter (if it exists) ───────────────────────────────
    if os.path.isdir(ADAPTER_PATH):
        status_placeholder.info("🔌 Attaching QLoRA adapter…")
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    model.eval()
    status_placeholder.empty()
    return tokenizer, model


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
def generate_response(tokenizer, model, conversation_history: list) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids      = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.5,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Parse follow-up suggestions from response text
# ─────────────────────────────────────────────────────────────────────────────
def parse_suggestions(response: str) -> tuple[str, list[str]]:
    """Returns (clean_response_without_suggestions, [suggestion1, suggestion2, suggestion3])"""
    suggestions = []
    clean = response

    marker = "💡 You might also want to ask:"
    if marker in response:
        parts = response.split(marker, 1)
        clean = parts[0].strip()
        block = parts[1]
        for line in block.strip().splitlines():
            line = line.strip()
            if line and re.match(r"^\d[\.\)]", line):
                suggestions.append(re.sub(r"^\d[\.\)]\s*", "", line).strip())

    return clean, suggestions[:3]


# ─────────────────────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # [{role, content, suggestions?}]
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 16px;'>
        <div style='font-size:3rem; margin-bottom:8px;'>👨‍💻</div>
        <div style='font-family:"Syne",sans-serif; font-weight:800; font-size:1.15rem; color:#e2e8f0;'>Meiyarasu K</div>
        <div style='font-family:"JetBrains Mono",monospace; font-size:0.68rem; color:#f59e0b; letter-spacing:0.08em; text-transform:uppercase; margin-top:4px;'>Python Developer · CV</div>
        <div class='status-badge'><span class='online-dot'></span>Available</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-family:\"JetBrains Mono\",monospace; font-size:0.68rem; color:#64748b; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:8px;'>Quick Facts</div>", unsafe_allow_html=True)

    facts = [
        ("Role",      "Python Dev · CV"),
        ("Company",   "Hexbee Software"),
        ("Location",  "Tamil Nadu, India"),
        ("Projects",  "3 Deployed"),
        ("Stack",     "PyTorch · YOLO · FastAPI"),
    ]
    for label, value in facts:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='label'>{label}</div>
            <div class='value'>{value}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-family:\"JetBrains Mono\",monospace; font-size:0.68rem; color:#64748b; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:10px;'>Try asking…</div>", unsafe_allow_html=True)

    starters = [
        "Tell me about your ANPR project",
        "How does your face recognition work?",
        "What's your ML stack?",
        "Walk me through the false positive fix",
    ]
    for s in starters:
        if st.button(s, key=f"starter_{s}", use_container_width=True):
            st.session_state.pending_input = s
            st.rerun()

    st.markdown("---")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_input = None
        st.rerun()

    st.markdown("""
    <div style='font-family:"JetBrains Mono",monospace; font-size:0.62rem; color:#3a4055; text-align:center; margin-top:16px; line-height:1.6;'>
        LLaMA-3.2-3B · QLoRA fine-tuned<br>
        Auto-downloaded via kagglehub
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main chat area
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='chat-header'>
    <h1>Chat with Meiyarasu</h1>
    <div class='tagline'><span class='online-dot'></span>AI-powered · Ask me anything about my work</div>
</div>
""", unsafe_allow_html=True)

# Load model (cached)
with st.spinner(""):
    col_load = st.empty()
    with col_load:
        tokenizer, model = load_model()
    col_load.empty()

# ── Render conversation history ──────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.markdown("""
        <div style='text-align:center; padding: 60px 20px; opacity:0.5;'>
            <div style='font-size:2.5rem; margin-bottom:12px;'>💬</div>
            <div style='font-family:"Syne",sans-serif; font-size:0.9rem; color:#64748b;'>
                Ask me about my projects, skills, or experience.<br>
                I'm Meiyarasu — let's talk.
            </div>
        </div>
        """, unsafe_allow_html=True)

    for i, msg in enumerate(st.session_state.messages):
        role = msg["role"]
        content = msg["content"]
        suggestions = msg.get("suggestions", [])

        if role == "user":
            st.markdown(f"""
            <div class='msg-row user'>
                <div class='avatar user'>👤</div>
                <div class='bubble user'>{content}</div>
            </div>""", unsafe_allow_html=True)
        else:
            # Clean response (already stripped of suggestion block)
            st.markdown(f"""
            <div class='msg-row bot'>
                <div class='avatar bot'>🤖</div>
                <div class='bubble bot'>{content}</div>
            </div>""", unsafe_allow_html=True)

            # Render clickable suggestions
            if suggestions:
                st.markdown("<div class='suggestions-label'>💡 You might also want to ask:</div>", unsafe_allow_html=True)
                for j, sug in enumerate(suggestions):
                    btn_key = f"sug_{i}_{j}"
                    st.markdown(f"""
                    <div class='pill-stack'>
                        <div class='pill' onclick="void(0)">
                            <span class='pill-num'>{j+1}</span>
                            {sug}
                        </div>
                    </div>""", unsafe_allow_html=True)
                    # Invisible Streamlit button for actual click handling
                    col1, col2 = st.columns([0.05, 0.95])
                    with col2:
                        if st.button(f"{j+1}. {sug}", key=btn_key, use_container_width=False):
                            st.session_state.pending_input = sug
                            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Handle pending input (from sidebar starters or suggestion clicks)
# ─────────────────────────────────────────────────────────────────────────────
def handle_user_message(user_text: str):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_text})

    # Build history for model
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

    # Show typing indicator
    typing_placeholder = st.empty()
    typing_placeholder.markdown("""
    <div class='msg-row bot'>
        <div class='avatar bot'>🤖</div>
        <div class='bubble bot'>
            <div class='typing-dots'>
                <span></span><span></span><span></span>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Generate
    raw_response = generate_response(tokenizer, model, history)
    typing_placeholder.empty()

    # Parse suggestions out
    clean_response, suggestions = parse_suggestions(raw_response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": clean_response,
        "suggestions": suggestions,
    })
    st.session_state.pending_input = None
    st.rerun()


if st.session_state.pending_input:
    handle_user_message(st.session_state.pending_input)

# ─────────────────────────────────────────────────────────────────────────────
# Chat input
# ─────────────────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask Meiyarasu anything…")
if user_input and user_input.strip():
    handle_user_message(user_input.strip())