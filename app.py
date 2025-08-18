# app.py — Streamlit version of your guarded RAG + FT demo (RAG or FT modes)
import os, re, time, json, pickle
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
import numpy as np
import pandas as pd

# ---- Optional heavy deps (fail gracefully on Streamlit Cloud CPU) ----
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
except Exception:
    torch = None
try:
    import faiss  # type: ignore
except Exception:
    faiss = None
try:
    from scipy import sparse
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    sparse = None
    cosine_similarity = None
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# ===================== Config =====================
MAX_QUESTION_CHARS = 280
MODEL_NAME  = os.environ.get("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
RAG_BASE    = Path(os.environ.get("RAG_BASE", "inference_rag_ft"))
IDX         = RAG_BASE / "data" / "indexes"
ADAPTER_DIR = Path(os.environ.get(
    "ADAPTER_DIR",
    str(RAG_BASE / "outputs" / "finetune" / "TinyLlama-1.1B-Chat-v1.0_lora_qna_miniloop")
))

ASSETS = {
    "chunks100": {"faiss": IDX/"faiss_chunks100.index", "meta": IDX/"meta_chunks100.pkl",
                  "tfidf_mat": IDX/"tfidf_chunks100.npz", "tfidf_vec": IDX/"tfidf_chunks100.pkl"},
    "chunks400": {"faiss": IDX/"faiss_chunks400.index", "meta": IDX/"meta_chunks400.pkl",
                  "tfidf_mat": IDX/"tfidf_chunks400.npz", "tfidf_vec": IDX/"tfidf_chunks400.pkl"},
    "tables":    {"faiss": IDX/"faiss_tables.index",   "meta": IDX/"meta_tables.pkl",
                  "tfidf_mat": IDX/"tfidf_tables.npz", "tfidf_vec": IDX/"tfidf_tables.pkl"},
}

# ===================== Guardrails =====================
RE_EMAIL     = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
RE_PROFANITY = re.compile(r"\b(?:damn|hell|shit|fuck|bitch)\b", re.I)
RE_OOS_UNSAFE = re.compile(
    r"\b(?:weather|temperature|forecast|joke|poem|recipe|translate|"
    r"image|picture|photo|code this|python|bomb|explosive|weapon|attack|kill)\b",
    re.I
)

def apply_input_rules(q: str) -> List[str]:
    rules = []
    if not isinstance(q, str) or not q.strip():
        rules.append("input_empty")
        return rules
    if len(q) > MAX_QUESTION_CHARS: rules.append("input_too_long")
    if RE_EMAIL.search(q):           rules.append("input_pii")
    if RE_PROFANITY.search(q):       rules.append("input_profanity")
    if RE_OOS_UNSAFE.search(q):      rules.append("input_out_of_scope_or_unsafe")
    return rules

def refusal_text(rules: List[str]) -> str:
    msgs = {
        "input_empty":                 "Please enter a question.",
        "input_too_long":              f"Please shorten your question to ≤ {MAX_QUESTION_CHARS} characters.",
        "input_pii":                   "I can’t process personal data (e.g., email addresses).",
        "input_profanity":             "Let’s keep language respectful.",
        "input_out_of_scope_or_unsafe":"That looks out of scope or unsafe for this financial Q&A assistant.",
    }
    return " ".join(msgs.get(r, "") for r in rules if r in msgs).strip()

def _have_gpu() -> bool:
    try:
        return torch and torch.cuda.is_available()
    except Exception:
        return False

# ===================== Lazy caches =====================
@st.cache_resource(show_spinner=False)
def load_embedder():
    if SentenceTransformer is None:
        return None
    device = "cuda" if _have_gpu() else "cpu"
    try:
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_rag_indexes():
    DENSE, SPARSE, HAS = {}, {}, False
    def _load_dense(name):
        if not faiss: return None
        a = ASSETS[name]
        if not (a["faiss"].exists() and a["meta"].exists()): return None
        try:
            idx = faiss.read_index(str(a["faiss"]))
            meta = pickle.load(open(a["meta"], "rb"))
            return idx, meta
        except Exception:
            return None
    def _load_sparse(name):
        if sparse is None: return None
        a = ASSETS[name]
        if not (a["tfidf_mat"].exists() and a["tfidf_vec"].exists() and a["meta"].exists()): return None
        try:
            vec  = pickle.load(open(a["tfidf_vec"], "rb"))
            X    = sparse.load_npz(a["tfidf_mat"])
            meta = pickle.load(open(a["meta"], "rb"))
            return vec, X, meta
        except Exception:
            return None
    DENSE  = {k: _load_dense(k)  for k in ASSETS}
    SPARSE = {k: _load_sparse(k) for k in ASSETS}
    HAS = any(DENSE.values()) or any(SPARSE.values())
    return DENSE, SPARSE, HAS

@st.cache_resource(show_spinner=True)
def load_ft_model():
    """Load base + LoRA if available. Falls back gracefully on CPU-only runtimes."""
    if torch is None:
        return None, None
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    try:
        eot_id = tok.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot_id, list) or eot_id is None or eot_id == tok.unk_token_id:
            eot_id = None
    except Exception:
        eot_id = None
    tok._eot_id = eot_id

    use_4bit = _have_gpu()
    qcfg = None
    if use_4bit:
        try:
            qcfg = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
            )
        except Exception:
            qcfg = None

    try:
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=qcfg if qcfg else None,
            device_map="auto" if use_4bit else None,
            torch_dtype=torch.float16 if use_4bit else None,
            trust_remote_code=True
        ).eval()
    except Exception:
        # CPU fallback
        base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True).eval()

    if not ADAPTER_DIR.exists():
        # Adapters missing — return base only (or None to skip FT stage)
        return tok, None

    try:
        ft = PeftModel.from_pretrained(base, ADAPTER_DIR).eval()
    except Exception:
        ft = None
    return tok, ft

# ===================== RAG helpers =====================
def _norm(vals: List[float]) -> List[float]:
    if not vals: return []
    v = np.array(vals, dtype=float); lo, hi = v.min(), v.max()
    return [0.5]*len(v) if hi-lo < 1e-9 else ((v-lo)/(hi-lo)).tolist()

def _dense(name: str, q: str, DENSE, EMB, k=5) -> List[Dict[str,Any]]:
    pack = DENSE.get(name)
    if not (faiss and pack and EMB): return []
    index, meta = pack
    qv = EMB.encode([q], normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(qv, k)
    out = []
    for s, i in zip(scores[0], idxs[0]):
        if i == -1: continue
        row = meta[i] if isinstance(meta, list) else meta.iloc[i].to_dict()
        out.append({"corpus": name, "score_dense": float(s), **row})
    return out

def _tfidf(name: str, q: str, SPARSE, k=5) -> List[Dict[str,Any]]:
    pack = SPARSE.get(name)
    if not pack or cosine_similarity is None: return []
    vec, X, meta = pack
    sims = cosine_similarity(vec.transform([q]), X).ravel()
    idxs = np.argsort(-sims)[:k]
    out = []
    for i in idxs:
        row = meta[i] if isinstance(meta, list) else meta.iloc[i].to_dict()
        out.append({"corpus": name, "score_sparse": float(sims[i]), **row})
    return out

def hybrid_search(query: str, DENSE, SPARSE, EMB, k_each=5, top_k=8, use_tables=True):
    corpora = ["chunks100","chunks400"] + (["tables"] if use_tables and DENSE.get("tables") else [])
    pool = []
    for name in corpora:
        pool += _dense(name, query, DENSE, EMB, k_each)
        pool += _tfidf(name, query, SPARSE, k_each)

    grouped = {}
    for r in pool:
        key = (r["corpus"], r.get("id") or f"{r.get('doc_id')}|{r.get('section')}|{r.get('chunk_index')}")
        g = grouped.setdefault(key, {"row": r, "dense": [], "sparse": []})
        if "score_dense" in r:  g["dense"].append(r["score_dense"])
        if "score_sparse" in r: g["sparse"].append(r["score_sparse"])

    dn = _norm([np.mean(g["dense"])  if g["dense"]  else 0.0 for g in grouped.values()])
    sn = _norm([np.mean(g["sparse"]) if g["sparse"] else 0.0 for g in grouped.values()])

    fused = []
    for (key, g), d, s in zip(grouped.items(), dn, sn):
        row = g["row"].copy()
        row["score_fused"] = 0.65*d + 0.35*s
        fused.append(row)
    fused.sort(key=lambda x: -x["score_fused"])
    return fused[:top_k]

def table_lookup_soft(question: str, DENSE, SPARSE, EMB, topn_embed: int = 40):
    if not DENSE.get("tables"):
        return None
    cand = _tfidf("tables", question, SPARSE, k=topn_embed) or _dense("tables", question, DENSE, EMB, k=topn_embed) or []
    if not cand or EMB is None: return None
    qv = EMB.encode([question], normalize_embeddings=True)
    tv = EMB.encode([(c.get("text") or "") for c in cand], normalize_embeddings=True)
    sims = (qv @ tv.T).ravel()
    i = int(np.argmax(sims)); best = cand[i]
    return {
        "answer": (best.get("text") or "").strip(),
        "method": "extractive/tables",
        "sources": [{
            "source": "tables",
            "title": best.get("section",""),
            "text": (best.get("text") or "")[:300],
            "score": float(sims[i]),
        }]
    }

def rag_generate_answer(query: str) -> dict:
    DENSE, SPARSE, HAS = load_rag_indexes()
    if not HAS:
        return {"answer":"RAG indexes not found. Only FT model will be used.",
                "method":"RAG/disabled","sources":[],"seconds":0.0,"confidence":0.0}
    t0 = time.time()
    EMB = load_embedder()
    det = table_lookup_soft(query, DENSE, SPARSE, EMB, topn_embed=40)
    if det:
        det["seconds"] = round(time.time() - t0, 3); det["confidence"] = 0.9
        return det
    res = hybrid_search(query, DENSE, SPARSE, EMB, k_each=5, top_k=6, use_tables=False)
    if not res:
        return {"answer":"No relevant context found.","method":"RAG/retrieve_empty","sources":[],"seconds":round(time.time()-t0,3),"confidence":0.0}
    top = res[0]
    return {
        "answer": (top.get("text") or "").strip(),
        "method": f"retrieve/{top.get('corpus')}",
        "sources": [{
            "source": top.get("corpus"),
            "title": top.get("section",""),
            "text": (top.get("text") or "")[:300],
            "score": float(top.get("score_fused", 0.0)),
        }],
        "seconds": round(time.time()-t0, 3),
        "confidence": 0.6
    }

def build_prompt(q: str) -> str:
    return "<|system|>\nYou are a precise financial QA assistant.\n" \
           f"<|user|>\nQuestion: {q}\n<|assistant|>\nAnswer:"

def ft_generate(q: str, max_new: int = 64):
    tok, model = load_ft_model()
    if (tok is None) or (model is None):
        return None, "Fine-tuned model not available on this runtime."
    try:
        enc = tok(build_prompt(q), return_tensors="pt", truncation=True,
                  max_length=tok.model_max_length)
        dev = next(model.parameters()).device
        enc = {k: v.to(dev) for k, v in enc.items()}
        stop_ids = [tok.eos_token_id]
        eot_id = getattr(tok, "_eot_id", None)
        if eot_id is not None: stop_ids.append(eot_id)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new, do_sample=False, num_beams=1,
                eos_token_id=stop_ids, pad_token_id=tok.eos_token_id,
            )
        raw = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=False)
        import re as _re
        raw = _re.split(r"(?:<\|eot_id\|>|<\|user\|>|</s>)", raw)[0]
        return raw.strip().splitlines()[0].strip(), None
    except Exception as e:
        return None, f"Generation error: {e}"

# ===================== UI =====================
st.set_page_config(page_title="Financial Q&A — Guarded RAG + FT", page_icon="💬", layout="centered")

st.title("Financial Q&A — Guarded RAG Vs. Fine-Tuned TinyLlama")
st.write("Guardrails → (optional) RAG retrieve → (optional) LoRA model. Deployable on Streamlit Community Cloud.")

with st.sidebar:
    st.header("Status")
    DENSE, SPARSE, HAS_RAG = load_rag_indexes()
    st.write(f"RAG indexes found: **{HAS_RAG}**")
    emb_ok = load_embedder() is not None
    st.write(f"Embedder ready: **{emb_ok}**")
    tok, ftm = load_ft_model()
    st.write(f"FT model loaded: **{ftm is not None}**")

    if not HAS_RAG:
        st.warning("RAG indexes not found — RAG mode will show a placeholder message.", icon="⚠️")

    st.divider()
    mode = st.radio(
        "Run mode",
        ["RAG", "FT"],
        index=0,  # default to RAG; set 1 if you prefer FT
        help="Choose how to answer: retrieve (RAG) or fine-tuned model (FT)."
    )
    fallback_if_ft_missing = st.checkbox(
        "Fallback to RAG if FT unavailable (FT mode)",
        value=True
    )

    st.caption("Tip: Set env vars `RAG_BASE`, `ADAPTER_DIR` to custom paths.")

q = st.text_input("Your question", placeholder="e.g., What was total revenue in FY2023?")
max_new = st.slider("Max new tokens", 16, 256, 64, step=8)

if st.button("Run"):
    t0 = time.perf_counter()
    rules = apply_input_rules(q)
    if rules:
        st.error(refusal_text(rules))
    else:
        final = ""
        rag = None
        ft_text, ft_err = None, None

        if mode == "RAG":
            rag = rag_generate_answer(q)
            final = rag.get("answer") or ""
        else:  # mode == "FT"
            ft_text, ft_err = ft_generate(q, max_new=max_new)
            if ft_text:
                final = ft_text
            elif fallback_if_ft_missing:
                rag = rag_generate_answer(q)
                final = rag.get("answer") or ""
            else:
                st.error("Fine-tuned model not available on this runtime.")

        st.subheader("Answer")
        st.write(final if final else "_(no answer)_")

        with st.expander("Pipeline details"):
            summary = {
                "mode": mode,
                "guardrails": "applied",
                "rag_method": (rag or {}).get("method") if rag else None,
                "rag_seconds": (rag or {}).get("seconds") if rag else None,
                "rag_confidence": (rag or {}).get("confidence") if rag else None,
                "ft_available": ft_err is None if ft_err is not None else (tok is not None and ftm is not None),
                "latency_total_s": round(time.perf_counter() - t0, 3),
            }
            st.json(summary)

        if rag and rag.get("sources"):
            st.subheader("RAG Sources")
            st.dataframe(pd.DataFrame(rag["sources"]))

        if mode != "RAG" and ft_err:
            st.info(ft_err)
