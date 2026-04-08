import streamlit as st
import torch
import os
import re
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# =========================================================
# 🔥 FIX 1 — REMOVE WARNINGS & LOGS
# =========================================================
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore")

# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="Chat with Meiyarasu",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# SYSTEM PROMPT (unchanged)
# =========================================================
SYSTEM_PROMPT = """YOUR EXISTING SYSTEM PROMPT HERE (KEEP SAME)"""

# =========================================================
# MODEL PATHS
# =========================================================
ADAPTER_PATH = "./qlora-llama32-adapter"
KAGGLE_MODEL = "pengzhengcurtis/llama-3.2-3b-instruct/pytorch/default/1"

# =========================================================
# 🔥 SAFE MODEL LOADING
# =========================================================
@st.cache_resource(show_spinner=False)
def load_model():
    import kagglehub

    try:
        base_path = kagglehub.model_download(KAGGLE_MODEL)

        tokenizer = AutoTokenizer.from_pretrained(base_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        if os.path.isdir(ADAPTER_PATH):
            model = PeftModel.from_pretrained(model, ADAPTER_PATH)

        model.eval()
        return tokenizer, model

    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

# =========================================================
# RESPONSE GENERATION
# =========================================================
def generate_response(tokenizer, model, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            temperature=0.5,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
        )

    new_tokens = outputs[0][input_ids.shape[-1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # 🔥 FREE GPU MEMORY
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return response.strip()

# =========================================================
# SESSION STATE
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================================================
# SIDEBAR (CLEAN VERSION)
# =========================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0;'>
        <div style='font-size:3rem;'>👨‍💻</div>
        <div style='font-weight:800; font-size:1.2rem;'>Meiyarasu K</div>
        <div style='font-size:0.7rem; color:#f59e0b;'>Python Developer · CV</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ✅ PORTFOLIO BUTTON
    st.markdown("""
    <div style='text-align:center;'>
        <a href="https://meiyarasu3926.github.io/portfolio/" target="_blank"
        style="text-decoration:none; color:#f59e0b; border:1px solid #333;
        padding:8px 12px; border-radius:8px;">
        🌐 View Portfolio
        </a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

# =========================================================
# MAIN UI
# =========================================================
st.title("💬 Chat with Meiyarasu")

# =========================================================
# 🔥 LOADING UX (IMPORTANT)
# =========================================================
with st.spinner("Loading AI model... (first time may take 20–30s)"):
    tokenizer, model = load_model()

# =========================================================
# DISPLAY CHAT
# =========================================================
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# =========================================================
# USER INPUT
# =========================================================
user_input = st.chat_input("Ask me anything...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = generate_response(
                tokenizer,
                model,
                st.session_state.messages
            )
            st.write(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
