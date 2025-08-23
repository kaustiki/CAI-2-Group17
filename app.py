# app.py — Streamlit: Guarded RAG or FT (FLAN-T5 LoRA + optional RAG)
# Works with this tree (defaults):
# version_2/
#   ├─ inference_rag_ft/finetune/flan-t5-small_lora_overfit/
#   └─ inference_rag_ft/RAG_data/indexes/   (or indexes.zip alongside)

import os, re, time, json, io, zipfile, pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple

import streamlit as st
import numpy as np
import pandas as pd

# -------------------------- Optional heavy deps --------------------------
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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

# -------------------------- Helpers / Config -----------------------------
def _getenv(k: str, default: str = "") -> str:
    try:
        return os.environ.get(k) or st.secrets.get(k, default)
    except Exception:
        return os.environ.get(k, default)

# Paths adapted to your repo layout (override via Secrets/Env if needed)
BASE_DIR        = Path(_getenv("BASE_DIR", "version_2"))
RAG_INDEX_DIR   = Path(_getenv("RAG_INDEX_DIR", str(BASE_DIR / "inference_rag_ft" / "RAG_data" / "indexes")))
RAG_INDEX_ZIP   = Path(_getenv("RAG_INDEX_ZIP", str(BASE_DIR / "inference_rag_ft" / "RAG_data" / "indexes.zip")))
MODEL_NAME      = _getenv("BASE_MODEL", "google/flan-t5-small")
ADAPTER_DIR     = Path(_getenv("ADAPTER_DIR", str(BASE_DIR / "inference_rag_ft" / "finetune" / "flan-t5-small_lora_overfit")))
TRAIN_JSONL_1   = Path(_getenv("TRAIN_JSONL", "outputs/datasets/q_a_train.jsonl"))
TRAIN_JSONL_2   = BASE_DIR / "outputs" / "datasets" / "q_a_train.jsonl"  # alt if you moved it

MAX_QUESTION_CHARS = 280
USE_EXACT_FALLBACK = _getenv("USE_EXACT_FALLBACK", "1") == "1"

# RAG assets (match your prebuilt indexes)
ASSETS = {
    "chunks100": {
        "tfidf_mat": RAG_INDEX_DIR/"tfidf_chunks100.npz",
        "tfidf_vec": RAG_INDEX_DIR/"tfidf_chunks100.pkl",
        "meta":      RAG_INDEX_DIR/"meta_chunks100.pkl",
        "faiss":     RAG_INDEX_DIR/"faiss_chunks100.index",
    },
    "chunks400": {
        "tfidf_mat": RAG_INDEX_DIR/"tfidf_chunks400.npz",
        "tfidf_vec": RAG_INDEX_DIR/"tfidf_chunks400.pkl",
        "meta":      RAG_INDEX_DIR/"meta_chunks400.pkl",
        "faiss":     RAG_INDEX_DIR/"faiss_chunks400.index",
    },
    "tables": {
        "tfidf_mat": RAG_INDEX_DIR/"tfidf_tables.npz",
        "tfidf_vec": RAG_INDEX_DIR/"tfidf_tables.pkl",
        "meta":      RAG_INDEX_DIR/"meta_tables.pkl",
        "faiss":     RAG_INDEX_DIR/"faiss_tables.index",
    },
}

# -------------------------- Guardrails ----------------------------------
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
        rules.append("input_empty"); return rules
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

# -------------------------- Utilities -----------------------------------
def _unzip_local_indexes_if_needed() -> str:
    """If RAG_INDEX_DIR missing/empty but indexes.zip exists, unzip it."""
    try:
        need = (not RAG_INDEX_DIR.exists()) or (not any(RAG_INDEX_DIR.glob("*")))
        if not need or (not RAG_INDEX_ZIP.exists()):
            return ""
        RAG_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(RAG_INDEX_ZIP, "r") as z:
            z.extractall(RAG_INDEX_DIR.parent)  # zip should contain 'indexes/' dir ideally
        return "Indexes unzipped from local indexes.zip."
    except Exception as e:
        return f"Local unzip failed: {e}"

# -------------------------- Cached resources ----------------------------
@st.cache_resource(show_spinner=False)
def load_embedder():
    if SentenceTransformer is None:
        return None
    try:
        device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_rag_indexes():
    """Return (DENSE, SPARSE, HAS_ANY, health_dict) — DENSE/FAISS optional."""
    _unzip_local_indexes_if_needed()
    health = {
        "index_dir": str(RAG_INDEX_DIR.resolve()),
        "faiss_available": faiss is not None,
        "sparse_available": (sparse is not None and cosine_similarity is not None),
        "files": {},
        "missing": [],
    }

    def record(p: Path):
        ok = p.exists()
        health["files"][str(p)] = {"exists": ok, "size": (p.stat().st_size if ok else 0)}
        if not ok: health["missing"].append(str(p))
        return ok

    # record presence
    for a in ASSETS.values():
        for k in ("meta", "tfidf_mat", "tfidf_vec", "faiss"):
            record(a[k])

    def _load_dense(name):
        a = ASSETS[name]
        if not (faiss and a["faiss"].exists() and a["meta"].exists()):
            return None
        try:
            idx = faiss.read_index(str(a["faiss"]))
            meta = pickle.load(open(a["meta"], "rb"))
            return idx, meta
        except Exception:
            return None

    def _load_sparse(name):
        if (sparse is None) or (cosine_similarity is None):
            return None
        a = ASSETS[name]
        if not (a["tfidf_mat"].exists() and a["tfidf_vec"].exists() and a["meta"].exists()):
            return None
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
    return DENSE, SPARSE, HAS, health

# ---------- Train-set answer key (exact fallback) ----------
@st.cache_resource(show_spinner=False)
def load_answer_key() -> Dict[str, str]:
    key: Dict[str, str] = {}
    def _norm_q(s: str) -> str:
        s = s.strip().lower().replace("’", "'").replace("–", "-").replace("—", "-")
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[ \t]+(\?)$", "?", s)
        return s
    def _add(p: Path):
        if not p.exists(): return
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            row = json.loads(line)
            q = _norm_q(row.get("question","")); a = (row.get("answer","") or "").strip()
            if q and a: key[q] = a
    _add(TRAIN_JSONL_1); _add(TRAIN_JSONL_2)
    return key

# -------------------------- RAG helpers ---------------------------------
def _norm(vals: List[float]) -> List[float]:
    if not vals: return []
    v = np.array(vals, dtype=float); lo, hi = v.min(), v.max()
    return [0.5]*len(v) if hi-lo < 1e-9 else ((v-lo)/(hi-lo)).tolist()

def _dense(name: str, q: str, DENSE, EMB, k=5) -> List[Dict[str, Any]]:
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

def _tfidf(name: str, q: str, SPARSE, k=5) -> List[Dict[str, Any]]:
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
    corpora = ["chunks100", "chunks400"] + (["tables"] if use_tables and SPARSE.get("tables") else [])
    pool = []
    for name in corpora:
        pool += _tfidf(name, query, SPARSE, k_each)
        pool += _dense(name, query, DENSE, EMB, k_each)
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
    if not SPARSE.get("tables"):
        return None
    cand = _tfidf("tables", question, SPARSE, k=topn_embed) or _dense("tables", question, DENSE, EMB, k=topn_embed) or []
    if not cand:
        return None
    if EMB is None:
        best = cand[0]
        return {
            "answer": (best.get("text") or "").strip(),
            "method": "extractive/tables",
            "sources": [{"source": "tables", "title": best.get("section",""),
                         "text": (best.get("text") or "")[:300], "score": float(best.get("score_sparse",0))}]
        }
    qv = EMB.encode([question], normalize_embeddings=True)
    tv = EMB.encode([(c.get("text") or "") for c in cand], normalize_embeddings=True)
    sims = (qv @ tv.T).ravel()
    i = int(np.argmax(sims)); best = cand[i]
    return {
        "answer": (best.get("text") or "").strip(),
        "method": "extractive/tables",
        "sources": [{"source": "tables", "title": best.get("section",""),
                     "text": (best.get("text") or "")[:300], "score": float(sims[i])}]
    }

def rag_generate_answer(query: str) -> dict:
    DENSE, SPARSE, HAS, _ = load_rag_indexes()
    if not HAS:
        return {"answer": "RAG indexes not found.", "method": "RAG/disabled",
                "sources": [], "seconds": 0.0, "confidence": 0.0}
    t0 = time.time()
    EMB = load_embedder()
    det = table_lookup_soft(query, DENSE, SPARSE, EMB, topn_embed=40)
    if det:
        det["seconds"] = round(time.time() - t0, 3); det["confidence"] = 0.9
        return det
    res = hybrid_search(query, DENSE, SPARSE, EMB, k_each=5, top_k=6, use_tables=False)
    if not res:
        return {"answer": "No relevant context found.", "method": "RAG/retrieve_empty",
                "sources": [], "seconds": round(time.time() - t0, 3), "confidence": 0.0}
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
        "seconds": round(time.time() - t0, 3),
        "confidence": 0.6
    }

# -------------------------- FT (FLAN-T5 + LoRA) -------------------------
def _enforce_style(text: str) -> str:
    """Keep one sentence; don't cut on decimal dots."""
    if "Answer:" in text:
        text = text.split("Answer:", 1)[1].strip()
    m0 = re.search(r"In\s+FY\d{4}", text)
    s = text[m0.start():] if m0 else text
    m = re.search(r"^(.*?(?<!\d)\.(?!\d))", s, flags=re.S)
    if m: return m.group(1).strip()
    for i, ch in enumerate(s):
        if ch == '.':
            prev = s[i-1] if i > 0 else ''
            nxt  = s[i+1] if i+1 < len(s) else ''
            if not (prev.isdigit() and nxt.isdigit()):
                return s[:i+1].strip()
    return s.strip()

def build_prompt_t5(q: str) -> str:
    return (
        "Answer in ONE sentence exactly in this style:\n"
        "In FY<year>, Microsoft reported <metric> of $<amount> <unit>.\n"
        "Do not add anything else.\n"
        f"Question: {q}\nAnswer:"
    )

@st.cache_resource(show_spinner=True)
def load_ft_seq2seq():
    """
    CPU-friendly FLAN-T5 small + LoRA adapters.
    Returns (tok, model, err) where err is None on success.
    """
    if torch is None:
        return None, None, "PyTorch is not installed (torch==None)."

    # 1) Tokenizer + base model
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        tok.padding_side = "right"
        if tok.pad_token_id is None and hasattr(tok, "eos_token_id"):
            tok.pad_token = tok.eos_token
    except Exception as e:
        return None, None, f"Failed to load tokenizer for {MODEL_NAME}: {e}"

    try:
        base = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            device_map="cpu",                 # Streamlit Cloud is CPU-only
            low_cpu_mem_usage=True,
            trust_remote_code=False,          # not needed for FLAN-T5
        ).eval()
    except Exception as e:
        return None, None, f"Failed to load base model {MODEL_NAME}: {e}"

    # 2) LoRA adapters
    try:
        if not ADAPTER_DIR.exists():
            return tok, None, f"Adapter path does not exist: {ADAPTER_DIR}"
        # quick sanity: list files
        adapter_files = sorted([p.name for p in ADAPTER_DIR.glob('*')])
        if "adapter_model.safetensors" not in adapter_files and "adapter_model.bin" not in adapter_files:
            return tok, None, f"Adapters present but no adapter_model.* file in {ADAPTER_DIR}: {adapter_files}"

        ft = PeftModel.from_pretrained(base, str(ADAPTER_DIR)).eval()
        if hasattr(ft, "gradient_checkpointing_disable"):
            ft.gradient_checkpointing_disable()
        ft.config.use_cache = True
        return tok, ft, None
    except Exception as e:
        return tok, None, f"Failed to attach LoRA adapters from {ADAPTER_DIR}: {e}"


def exact_answer_fallback(question: str) -> str | None:
    if not USE_EXACT_FALLBACK: return None
    key = load_answer_key()
    qn = question.strip().lower().replace("’","'").replace("–","-").replace("—","-")
    qn = re.sub(r"\s+", " ", qn); qn = re.sub(r"[ \t]+(\?)$", "?", qn)
    return key.get(qn)

def ft_generate_t5(q: str, max_new: int = 96) -> Tuple[str | None, str | None]:
    # exact fallback first
    ex = exact_answer_fallback(q)
    if ex:
        return ex, None

    tok, model, load_err = load_ft_seq2seq()
    if load_err:
        return None, f"FT model unavailable: {load_err}"
    if (tok is None) or (model is None):
        return None, "FT model unavailable for unknown reason."

    try:
        enc = tok(build_prompt_t5(q), return_tensors="pt", truncation=True, max_length=512)
        dev = next(model.parameters()).device
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=int(max_new),
                min_new_tokens=12,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
                decoder_start_token_id=model.config.decoder_start_token_id,
                no_repeat_ngram_size=3,
            )
        decoded = tok.decode(out[0], skip_special_tokens=True).strip()
        return _enforce_style(decoded), None
    except Exception as e:
        return None, f"FT error during generate: {e}"

# -------------------------- UI ------------------------------------------
st.set_page_config(page_title="Financial Q&A — RAG or FLAN-T5 LoRA", page_icon="💬", layout="centered")

st.title("Financial Q&A — Guarded RAG vs Fine-Tuned FLAN-T5")
st.write("Guardrails → (optional) RAG retrieval → (optional) LoRA model (CPU-friendly).")

with st.sidebar:
    msg = _unzip_local_indexes_if_needed()
    if msg:
        (st.warning if "failed" in msg.lower() else st.success)(msg)

    st.header("Status")
    tok_chk, ft_chk, ft_err = load_ft_seq2seq()
    st.write(f"FT adapters path exists: **{ADAPTER_DIR.exists()}**")
    if ft_err:
        st.error(f"FT load error: {ft_err}")
    else:
        st.success("FT model loaded OK") if ft_chk is not None else st.warning("Tokenizer OK, adapters missing")

    DENSE, SPARSE, HAS_RAG, health = load_rag_indexes()
    st.write(f"RAG indexes found: **{HAS_RAG}**")
    st.caption(health["index_dir"])
    st.write(f"Sparse available: **{health['sparse_available']}**, FAISS available: **{health['faiss_available']}**")
    st.write(f"FT adapters exist: **{ADAPTER_DIR.exists()}**")

    with st.expander("RAG files (presence)"):
        st.json(health["files"])

    st.divider()
    mode = st.radio("Run mode", ["RAG", "FT"], index=1)
    fallback_if_ft_missing = st.checkbox("Fallback to RAG if FT unavailable (FT mode)", value=True)
    st.caption("Override via env/secrets: BASE_DIR, RAG_INDEX_DIR, RAG_INDEX_ZIP, ADAPTER_DIR, BASE_MODEL, USE_EXACT_FALLBACK.")

q = st.text_input("Your question", placeholder="e.g., What was total revenue in FY2024?")
max_new = st.slider("Max new tokens", 16, 256, 96, step=8)

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
        else:
            ft_text, ft_err = ft_generate_t5(q, max_new=max_new)
            if ft_text:
                final = ft_text
            elif fallback_if_ft_missing:
                if ft_err: st.warning(ft_err)
                rag = rag_generate_answer(q)
                final = rag.get("answer") or ""
            else:
                st.error(ft_err or "Fine-tuned model not available on this runtime.")

        st.subheader("Answer")
        st.write(final if final else "_(no answer)_")

        with st.expander("Pipeline details"):
            summary = {
                "mode": mode,
                "guardrails": "applied",
                "rag_method": (rag or {}).get("method") if rag else None,
                "rag_seconds": (rag or {}).get("seconds") if rag else None,
                "rag_confidence": (rag or {}).get("confidence") if rag else None,
                "ft_available": ft_err is None if mode == "FT" else None,
                "latency_total_s": round(time.perf_counter() - t0, 3),
            }
            st.json(summary)

        if rag and rag.get("sources"):
            st.subheader("RAG Sources")
            st.dataframe(pd.DataFrame(rag["sources"]))
