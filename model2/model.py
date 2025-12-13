# Cell 1 — retriever + router+experts MoE + preprocess/predict
# Loads models from a local ./models folder next to this file (with optional env overrides).
# Includes CROSS-ENCODER DOCUMENT PRETOKENIZATION (shared tokenizer).

import os
import json
import math
import time
import hashlib
from typing import Dict, List, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------- #
# FAISS (optional)
# ----------------------------- #
try:
    import faiss  # pip install faiss-cpu (or faiss-gpu)
    FAISS_AVAILABLE = True
except Exception as e:
    print(f"[FAISS] Not available ({e}). Falling back to cosine_similarity for retrieval.")
    FAISS_AVAILABLE = False

FAISS_CACHE_DIR = os.getenv("FAISS_CACHE_DIR", "./faiss_cache")
os.makedirs(FAISS_CACHE_DIR, exist_ok=True)

# ----------------------------- #
# Retrieval config
# ----------------------------- #
K_TOTAL = int(os.getenv("K_TOTAL", "80"))  # Stage-1 total candidates

SOURCE_WEIGHTS = {
    "e5": float(os.getenv("WEIGHT_E5", "0.6")),
    "lex": float(os.getenv("WEIGHT_LEX", "0.4")),
}

# Lexical config
USE_TFIDF_LEXICAL = bool(int(os.getenv("USE_TFIDF_LEXICAL", "1")))
TFIDF_MAX_FEATS = int(os.getenv("TFIDF_MAX_FEATS", "300000"))

RERANK_MAX_LEN = int(os.getenv("RERANK_MAX_LEN", "384"))
RERANK_PAD_MULT = int(os.getenv("RERANK_PAD_TO_MULTIPLE_OF", "8"))

# Per-expert CE max length
RERANK_MAX_LEN_KZ = int(os.getenv("RERANK_MAX_LEN_KZ", "448"))
RERANK_MAX_LEN_WIKI = int(os.getenv("RERANK_MAX_LEN_WIKI", "360"))
RERANK_MAX_LEN_KNESSET = int(os.getenv("RERANK_MAX_LEN_KNESSET", "496"))

# ----------------------------- #
# Router + Experts config
# ----------------------------- #
USE_ROUTER_MOE = bool(int(os.getenv("USE_ROUTER_MOE", "1")))
CALIBRATION_MODE = os.getenv("CALIBRATION_MODE", "none").lower()  # "minmax" or "none"

ROUTER_TAU_HI = float(os.getenv("ROUTER_TAU_HI", "0.977"))  # commit to top-1
ROUTER_TAU_MID = float(os.getenv("ROUTER_TAU_MID", "0.45"))  # mix top-2

RERANK_BATCH = int(os.getenv("RERANK_BATCH", "80"))  # For mix2: how many docs the second expert sees
MIX2_TOPM = int(os.getenv("MOE_MIX2_TOPM", "20"))

# Optional explicit expert paths (env overrides); if not set, we use ./models/ce_*
EXPERT_PATHS_JSON = os.getenv("EXPERT_PATHS", "").strip()  # JSON mapping {"kz":"...", "wiki":"...", "knesset":"..."}
EXPERT_KZ_DIR = os.getenv("EXPERT_KZ_DIR", "").strip()
EXPERT_WIKI_DIR = os.getenv("EXPERT_WIKI_DIR", "").strip()
EXPERT_KNESSET_DIR = os.getenv("EXPERT_KNESSET_DIR", "").strip()
ROUTER_DIR_ENV = os.getenv("ROUTER_DIR", "").strip()
E5_LOCAL_DIR_ENV = os.getenv("E5_LOCAL_DIR", "").strip()

# NEW: optional general CE path (global finisher)
GENERAL_CE_DIR_ENV = os.getenv("GENERAL_CE_DIR", "").strip()

# ----------------------------- #
# Local models helper
# ----------------------------- #
def _models_root() -> str:
    env = os.getenv("MODELS_DIR")
    if env and os.path.isdir(env):
        return env
    try:
        base = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base = os.getcwd()
    return os.path.join(base, "models")

def _path_exists(p: str) -> bool:
    return isinstance(p, str) and len(p) > 0 and os.path.isdir(p)

def _try_local_model(*subdirs) -> str | None:
    root = _models_root()
    for sd in subdirs:
        p = os.path.join(root, sd)
        if os.path.isdir(p):
            return p
    return None

def _pick_local_router_dir() -> str | None:
    if _path_exists(ROUTER_DIR_ENV):
        return ROUTER_DIR_ENV
    cand = os.path.join(_models_root(), "alephbert-base_case_name_best")
    return cand if os.path.isdir(cand) else None

def _pick_expert_root() -> str | None:
    mr = _models_root()
    return mr if os.path.isdir(mr) else None

# ----------------------------- #
# Attention-impl fallback loader
# ----------------------------- #
def _load_with_attn_fallback(model_cls, model_name: str, base_kwargs: dict, label: str):
    last_err = None
    for impl, tag in (("sdpa", "SDPA"), ("eager", "eager"), (None, "default")):
        try:
            kwargs = dict(base_kwargs)
            if impl is not None:
                kwargs["attn_implementation"] = impl
            m = model_cls.from_pretrained(model_name, **kwargs)
            print(f"{label} Loaded with attention implementation: {tag}.")
            return m
        except (TypeError, ValueError) as e:
            print(f"{label} {tag} failed: {e}")
            last_err = e
    print(f"{label} Falling back to minimal load (no dtype/attn args). Last error: {last_err}")
    return model_cls.from_pretrained(model_name)

# ----------------------------- #
# Hebrew helpers
# ----------------------------- #
ALIASES = [
    ("יו\"ר", ["יושב ראש","יושב-ראש","יו״ר"]),
    ("יו״ר", ["יושב ראש","יושב-ראש","יו\"ר"]),
    ("ח\"כ", ["חבר כנסת","ח״כ"]),
    ("ח״כ", ["חבר כנסת","ח\"כ"]),
    ("מס'", ["מספר"]),
    ("תק'", ["תקנה","תקנות"]),
    ("סע'", ["סעיף"]),
]

def expand_hebrew_query(q: str) -> str:
    q2 = q or ""
    for a, ex in ALIASES:
        if a in q2:
            q2 += " " + " ".join(ex)
    return q2

def _simple_tokenize(s: str) -> List[str]:
    return (s or "").lower().split()

def _canon_case_label(s: str) -> str:
    if not s:
        return ""
    t = str(s).lower()
    if ("knesset" in t) or ("כנסת" in t):
        return "knesset"
    if ("wiki" in t) or ("ויקיפ" in t):
        return "wiki"
    if ("kz" in t) or ("זכות" in t) or ("kol" in t):
        return "kz"
    return t

# ----------------------------- #
# Lexical engines
# ----------------------------- #
class GlobalLexBM25:
    def __init__(self, texts: List[str]):
        from rank_bm25 import BM25Okapi  # pip install rank_bm25
        self.tokens = [_simple_tokenize(t) for t in texts]
        self.engine = BM25Okapi(self.tokens)

    def scores_all(self, q: str) -> np.ndarray:
        return np.asarray(self.engine.get_scores(_simple_tokenize(expand_hebrew_query(q))), dtype=np.float32)

class GlobalLexTFIDF:
    def __init__(self, texts: List[str], max_feats: int = 300_000):
        self.vec = TfidfVectorizer(max_features=max_feats, ngram_range=(1, 2))
        self.mat = self.vec.fit_transform(texts)

    def scores_all(self, q: str) -> np.ndarray:
        v = self.vec.transform([q])
        return (self.mat @ v.T).toarray().ravel().astype(np.float32)

# ----------------------------- #
# E5 Retriever
# ----------------------------- #
class E5Retriever:
    def __init__(self, model_name=None, device=None):
        if model_name is None:
            if _path_exists(E5_LOCAL_DIR_ENV):
                model_name = E5_LOCAL_DIR_ENV
            else:
                local = _try_local_model("multilingual-e5-large")
                model_name = local if local else "intfloat/multilingual-e5-large"
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        dtype = torch.float16 if self.device == "cuda" else None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        base_kwargs = {"torch_dtype": dtype} if dtype is not None else {}
        model_cpu = _load_with_attn_fallback(AutoModel, model_name, base_kwargs, label="[E5]")
        self.model = model_cpu.to(self.device).eval()

        self.corpus_ids: List[str] = []
        self.corpus_embeddings: np.ndarray | None = None

    def embed_texts(self, texts: List[str], is_query: bool = False, batch_size: int = 32) -> np.ndarray:
        prefix = "query" if is_query else "passage"
        prefixed_texts = [f"{prefix}: {t.strip()}" for t in texts]
        all_embeddings = []

        for i in range(0, len(prefixed_texts), batch_size):
            batch_texts = prefixed_texts[i:i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.inference_mode():
                out = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                emb = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            all_embeddings.append(emb.cpu())

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return torch.cat(all_embeddings, dim=0).numpy()

# ----------------------------- #
# Router + Experts (MoE)
# ----------------------------- #
class HebRouter:
    def __init__(self, ckpt_dir: str):
        from transformers import AutoModelForSequenceClassification
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
        self.m = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(self.dev).eval()

        # labels
        labels_path = os.path.join(ckpt_dir, "labels.json")
        if os.path.exists(labels_path):
            raw = json.load(open(labels_path, "r", encoding="utf-8")).get("labels", [])
        else:
            raw = [v for _, v in sorted(self.m.config.id2label.items())]
        self.labels = [_canon_case_label(x) for x in raw]

        # optional calibration
        self.T = 1.0
        cal_path = os.path.join(ckpt_dir, "calibration.json")
        if os.path.exists(cal_path):
            try:
                self.T = float(json.load(open(cal_path, "r", encoding="utf-8"))["temperature"])
            except Exception:
                pass

    @torch.no_grad()
    def probs(self, text: str, max_len: int = 128) -> Dict[str, float]:
        x = self.tok(text, return_tensors="pt", truncation=True, padding=True, max_length=max_len)
        x = {k: v.to(self.m.device) for k, v in x.items()}
        p = torch.softmax(self.m(**x).logits / max(self.T, 1e-6), dim=-1)[0].cpu().numpy()
        return {lab: float(p[i]) for i, lab in enumerate(self.labels)}

class CaseCE:
    def __init__(self, ckpt_dir: str, max_len: int = 384, device=None):
        from transformers import AutoModelForSequenceClassification
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len

        self.tok = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
        base_kwargs = {"trust_remote_code": True}
        if self.dev == "cuda":
            base_kwargs["torch_dtype"] = torch.float16

        try:
            self.m = AutoModelForSequenceClassification.from_pretrained(
                ckpt_dir, **base_kwargs, attn_implementation="sdpa"
            ).to(self.dev).eval()
        except (TypeError, ValueError):
            self.m = AutoModelForSequenceClassification.from_pretrained(
                ckpt_dir, **base_kwargs
            ).to(self.dev).eval()

        if self.dev != "cuda":
            try:
                self.m = self.m.to(dtype=torch.float32)
            except Exception:
                pass

    @torch.no_grad()
    def score(
        self,
        q: str,
        passages: List[str] | None = None,
        pids: List[str] | None = None,
        tok_map: Dict[str, List[int]] | None = None,
        bs: int = 16,
    ) -> np.ndarray:
        """
        Two modes:
        • Pretok mode: provide 'pids' and 'tok_map' (doc_id -> token_ids without specials).
        • Text mode: provide 'passages'.
        """
        # Pretok mode
        if tok_map is not None and pids is not None:
            if not pids:
                return np.zeros((0,), dtype=np.float32)

            out_chunks = []
            q_ids = self.tok.encode(q or "", add_special_tokens=False)
            special = self.tok.num_special_tokens_to_add(pair=True)
            bs = max(8, bs)
            avail = self.max_len - len(q_ids) - special
            if avail < 0:
                avail = 0

            for i in range(0, len(pids), bs):
                chunk = pids[i:i + bs]
                ids_out, mask_out, type_out = [], [], []
                use_types = None

                for pid in chunk:
                    p_full = tok_map.get(pid, [])
                    p_ids = p_full[:avail]
                    enc = self.tok.prepare_for_model(
                        q_ids,
                        p_ids,
                        add_special_tokens=True,
                        truncation=False,
                        padding="max_length",
                        max_length=self.max_len,
                        return_attention_mask=True,
                        return_token_type_ids=True,
                    )
                    ids_out.append(enc["input_ids"])
                    mask_out.append(enc["attention_mask"])
                    if "token_type_ids" in enc:
                        if use_types is None:
                            use_types = True
                        if use_types:
                            type_out.append(enc["token_type_ids"])
                    else:
                        if use_types is None:
                            use_types = False

                inputs = {
                    "input_ids": torch.tensor(ids_out, dtype=torch.long, device=self.m.device),
                    "attention_mask": torch.tensor(mask_out, dtype=torch.long, device=self.m.device),
                }
                if use_types:
                    inputs["token_type_ids"] = torch.tensor(type_out, dtype=torch.long, device=self.m.device)

                ctx = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if self.m.device.type == "cuda"
                    else torch.cpu.amp.autocast(enabled=False)
                )
                with torch.inference_mode(), ctx:
                    lg = self.m(**inputs).logits
                    if lg.ndim == 1:
                        s = lg
                    elif lg.shape[1] == 1:
                        s = lg.squeeze(-1)
                    else:
                        s = lg[:, 1]
                out_chunks.append(s.float().cpu().numpy())

            return np.concatenate(out_chunks, 0)

        # Text mode
        if not passages:
            return np.zeros((0,), dtype=np.float32)

        out = []
        bs = max(8, bs)
        for i in range(0, len(passages), bs):
            enc = self.tok(
                [q] * min(bs, len(passages) - i),
                passages[i:i + bs],
                padding="max_length",
                truncation="only_second",
                max_length=self.max_len,
                pad_to_multiple_of=RERANK_PAD_MULT,
                return_tensors="pt",
            )
            enc = {k: v.to(self.m.device) for k, v in enc.items()}
            ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if self.m.device.type == "cuda"
                else torch.cpu.amp.autocast(enabled=False)
            )
            with torch.inference_mode(), ctx:
                lg = self.m(**enc).logits
                if lg.ndim == 1:
                    s = lg
                elif lg.shape[1] == 1:
                    s = lg.squeeze(-1)
                else:
                    s = lg[:, 1]
            out.append(s.float().cpu().numpy())

        return np.concatenate(out, 0)

def _normalize_scores(scores: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return scores.astype(np.float32)
    s_min, s_max = float(np.min(scores)), float(np.max(scores))
    if s_max <= s_min + 1e-8:
        return np.zeros_like(scores, dtype=np.float32)
    return ((scores - s_min) / (s_max - s_min + 1e-8)).astype(np.float32)

def _choose_experts(probs: Dict[str, float], tau_hi=0.977, tau_mid=0.94):
    top = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    p1 = top[0][1]
    if p1 >= tau_hi:
        return [top[0][0]], {top[0][0]: 1.0}, "commit"
    if p1 >= tau_mid and len(top) > 1:
        s = top[0][1] + top[1][1]
        return [top[0][0], top[1][0]], {
            top[0][0]: top[0][1] / s,
            top[1][0]: top[1][1] / s,
        }, "mix2"
    keys = [k for k in ["kz", "wiki", "knesset"] if k in probs] or [k for k, _ in top]
    s = sum(probs[k] for k in keys) or 1.0
    return keys, {k: probs[k] / s for k in keys}, "safe3"

# ----------------------------- #
# Expert discovery
# ----------------------------- #
def _discover_expert_dirs(root: str) -> Dict[str, str]:
    """
    Returns {'kz': path, 'wiki': path, 'knesset': path}.
    Priority:
      1) EXPERT_KZ_DIR / EXPERT_WIKI_DIR / EXPERT_KNESSET_DIR envs
      2) EXPERT_PATHS (JSON mapping)
      3) ./models/ce_kz, ./models/ce_wiki, ./models/ce_knesset
    """
    import pathlib
    out: Dict[str, str] = {}

    def _has_model_dir(p: pathlib.Path) -> bool:
        return p.is_dir() and any(
            (p / f).exists() for f in ("config.json", "model.safetensors", "pytorch_model.bin")
        )

    # 1) explicit envs
    env_map = {"kz": EXPERT_KZ_DIR, "wiki": EXPERT_WIKI_DIR, "knesset": EXPERT_KNESSET_DIR}
    for lab, p in env_map.items():
        if p:
            pp = pathlib.Path(p)
            if _has_model_dir(pp):
                out[lab] = str(pp)

    # 2) JSON mapping
    if EXPERT_PATHS_JSON and len(out) < 3:
        try:
            mapping = json.loads(EXPERT_PATHS_JSON)
            for k, v in mapping.items():
                lab = _canon_case_label(k)
                pp = pathlib.Path(v)
                if lab in ("kz", "wiki", "knesset") and _has_model_dir(pp):
                    out.setdefault(lab, str(pp))
        except Exception as e:
            print("[MoE] EXPERT_PATHS parse error:", e)

    # 3) direct ce_* under root
    if len(out) < 3 and root:
        rp = pathlib.Path(root)
        cand = {"kz": "ce_kz", "wiki": "ce_wiki", "knesset": "ce_knesset"}
        for lab, name in cand.items():
            if lab in out:
                continue
            p = rp / name
            if _has_model_dir(p):
                out[lab] = str(p)

    for lab in ("kz", "wiki", "knesset"):
        print(f"[MoE] expert {lab}: {out.get(lab, '(missing)')}")
    return out

# ----------------------------- #
# MoEReranker with general_ce
# ----------------------------- #
class MoEReranker:
    def __init__(
        self,
        router: HebRouter,
        experts: Dict[str, CaseCE],
        calib_mode: str = "minmax",
        general_expert: CaseCE | None = None,   # NEW: global CE
    ):
        self.router = router
        self.experts = experts
        self.calib_mode = calib_mode
        self.general = general_expert
        print("[MoE] experts loaded:", ", ".join(sorted(self.experts.keys())))
        if self.general is not None:
            print("[MoE] general_ce loaded and will be used in mix2 mode.")
        else:
            print("[MoE] general_ce not available; mix2 will use only corpus-specific experts.")

        # shared pretokenization (set later in preprocess)
        self.shared_tok = None
        self.doc_tok_map: Dict[str, List[int]] | None = None

    def _score_expert_full(
        self,
        lab: str,
        q: str,
        passages: List[str],
        pids: List[str],
    ) -> np.ndarray:
        """Score ALL candidates with one expert and normalize."""
        if lab not in self.experts:
            return np.zeros(len(pids), dtype=np.float32)
        exp = self.experts[lab]

        if self.doc_tok_map is not None and self.shared_tok is not None:
            raw = exp.score(q, pids=pids, tok_map=self.doc_tok_map, bs=RERANK_BATCH)
        else:
            raw = exp.score(q, passages=passages, bs=RERANK_BATCH)

        return _normalize_scores(raw, self.calib_mode)

    def _score_expert_subset(
        self,
        lab: str,
        q: str,
        passages: List[str],
        pids: List[str],
        idx_subset: np.ndarray,
    ) -> np.ndarray:
        """Score ONLY a subset of candidates (by indices) with one expert and normalize."""
        if lab not in self.experts or idx_subset.size == 0:
            return np.zeros(idx_subset.size, dtype=np.float32)
        exp = self.experts[lab]
        sub_pids = [pids[i] for i in idx_subset.tolist()]

        if self.doc_tok_map is not None and self.shared_tok is not None:
            raw = exp.score(q, pids=sub_pids, tok_map=self.doc_tok_map, bs=RERANK_BATCH)
        else:
            sub_passages = [passages[i] for i in idx_subset.tolist()]
            raw = exp.score(q, passages=sub_passages, bs=RERANK_BATCH)

        return _normalize_scores(raw, self.calib_mode)

    def _score_general_subset(
        self,
        q: str,
        passages: List[str],
        pids: List[str],
        idx_subset: np.ndarray,
    ) -> np.ndarray:
        """
        Score ONLY a subset of candidates with the general CE (text mode).
        """
        if self.general is None or idx_subset.size == 0:
            return np.zeros(idx_subset.size, dtype=np.float32)
        sub_passages = [passages[i] for i in idx_subset.tolist()]
        raw = self.general.score(q, passages=sub_passages, bs=RERANK_BATCH)
        return _normalize_scores(raw, self.calib_mode)

    def rerank(self, q: str, passages: List[str], pids: List[str], top_k: int = 20):
        probs = self.router.probs(q)
        chosen, weights, mode = _choose_experts(probs, ROUTER_TAU_HI, ROUTER_TAU_MID)
        rounded_w = {k: round(float(v), 3) for k, v in weights.items()}
        print(f"[MoE] Router choice | mode={mode} | experts={chosen} | weights={rounded_w}")

        if not chosen or not pids:
            return [], probs, mode

        N = len(pids)

        # Case 1: commit / safe3 → all chosen experts score all K
        if (mode != "mix2") or (len(chosen) < 2):
            per: Dict[str, np.ndarray] = {}
            for lab in chosen:
                if lab not in self.experts:
                    continue
                per[lab] = self._score_expert_full(lab, q, passages, pids)
            if not per:
                return [], probs, mode

            comb = np.zeros(N, dtype=np.float32)
            for lab, w in weights.items():
                if lab in per:
                    comb += float(w) * per[lab]
            pairs = sorted(zip(pids, comb.tolist()), key=lambda x: x[1], reverse=True)
            return pairs[:top_k], probs, mode

        # Case 2: mix2 → main expert + general_ce (if available)
        main_lab, aux_lab = chosen[0], chosen[1]

        if (main_lab not in self.experts) and (self.general is None):
            # nothing reasonable to do
            return [], probs, mode

        s_main = (
            self._score_expert_full(main_lab, q, passages, pids)
            if main_lab in self.experts
            else np.zeros(N, dtype=np.float32)
        )
        w_main = float(weights.get(main_lab, 1.0))
        w_aux  = float(weights.get(aux_lab, 0.0))  # reused as general_ce weight

        # If no general_ce or router gives zero weight → only main expert
        if self.general is None or w_aux <= 0.0:
            comb = w_main * s_main
            pairs = sorted(zip(pids, comb.tolist()), key=lambda x: x[1], reverse=True)
            return pairs[:top_k], probs, mode

        # general_ce only re-scores top-M of main
        M = min(max(top_k, MIX2_TOPM), N)
        top_idx = np.argsort(-s_main)[:M]
        s_gen_top = self._score_general_subset(q, passages, pids, top_idx)

        comb = w_main * s_main
        for j, idx in enumerate(top_idx):
            comb[idx] += w_aux * s_gen_top[j]

        pairs = sorted(zip(pids, comb.tolist()), key=lambda x: x[1], reverse=True)
        return pairs[:top_k], probs, mode

# ----------------------------- #
# Globals filled in preprocess()
# ----------------------------- #
retriever: E5Retriever | None = None
moe: MoEReranker | None = None
corpus_texts: Dict[str, str] = {}

# ----------------------------- #
# FAISS helpers (+ cache)
# ----------------------------- #
def _build_faiss_index(embeddings: np.ndarray):
    if not FAISS_AVAILABLE:
        return None
    xb = np.asarray(embeddings, dtype=np.float32, order="C")
    dim = xb.shape[1]
    index = faiss.IndexFlatIP(dim)
    if torch.cuda.is_available() and hasattr(faiss, "StandardGpuResources"):
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            print("[FAISS] Using GPU IndexFlatIP.")
        except Exception as e:
            print(f"[FAISS] GPU not available, using CPU index. Reason: {e}")
    index.add(xb)
    print(f"[FAISS] Built index with {index.ntotal} vectors (dim={dim}).")
    return index

def _corpus_sig_for_cache(corpus_ids: List[str], passages: List[str], e5_name: str) -> str:
    payload = {
        "model": e5_name,
        "n": len(corpus_ids),
        "ids": corpus_ids[:50],
        "lens": [len(passages[i]) for i in range(min(len(passages), 50))],
    }
    return hashlib.sha1(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]

def _faiss_cache_paths(sig: str) -> Tuple[str, str, str]:
    emb = os.path.join(FAISS_CACHE_DIR, f"e5_{sig}_emb.npy")
    ids = os.path.join(FAISS_CACHE_DIR, f"e5_{sig}_ids.json")
    idx = os.path.join(FAISS_CACHE_DIR, f"e5_{sig}.faiss")
    return emb, ids, idx

def _try_load_cache(sig: str):
    emb_p, ids_p, faiss_p = _faiss_cache_paths(sig)
    if not (os.path.exists(emb_p) and os.path.exists(ids_p) and os.path.exists(faiss_p)):
        return None
    try:
        ids = json.load(open(ids_p, "r", encoding="utf-8"))
        emb = np.load(emb_p, mmap_mode="r")
        idx = faiss.read_index(faiss_p)
        if torch.cuda.is_available() and hasattr(faiss, "StandardGpuResources"):
            try:
                res = faiss.StandardGpuResources()
                idx = faiss.index_cpu_to_gpu(res, 0, idx)
                print("[FAISS] Loaded CPU index and moved to GPU.")
            except Exception as e:
                print("[FAISS] Using CPU index (GPU move failed):", e)
        print(f"[CACHE] Restored embeddings+index from: {FAISS_CACHE_DIR} (sig={sig})")
        return {"ids": ids, "emb": np.asarray(emb, dtype=np.float32), "index": idx}
    except Exception as e:
        print("[CACHE] Load failed:", e)
        return None

def _save_cache(sig: str, ids: List[str], emb: np.ndarray, index):
    emb_p, ids_p, faiss_p = _faiss_cache_paths(sig)
    try:
        arr = np.asarray(emb, dtype=np.float32, order="C")
        np.save(emb_p, arr)
        json.dump(list(ids), open(ids_p, "w", encoding="utf-8"), ensure_ascii=False)
        try:
            cpu_index = faiss.index_gpu_to_cpu(index)
        except Exception:
            cpu_index = index
        faiss.write_index(cpu_index, faiss_p)
        print(f"[CACHE] Saved embeddings+index → {FAISS_CACHE_DIR} (sig={sig})")
    except Exception as e:
        print("[CACHE] Save failed:", e)

# ----------------------------- #
# Preprocess: embed + FAISS + Lex + Router/Experts + CE pretokenization
# ----------------------------- #
def preprocess(corpus_dict: Dict[str, Dict[str, str]]):
    """
    Returns preprocessed data dict:
      - 'retriever', 'moe', 'corpus_ids', 'corpus_embeddings', 'faiss_index'
      - 'lex_engine'
      - 'corpus_texts'
      - 'ce_doc_tokens', 'ce_tokenizer_name'
    """
    global retriever, moe, corpus_texts

    print("=" * 60)
    print("PREPROCESSING: E5 + Lexical + Query Router (MoE experts) + CE Pretokenization …")
    print("=" * 60)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print("[PATHS] MODELS_ROOT :", _models_root())
    router_dir = _pick_local_router_dir()
    expert_root = _pick_expert_root()
    print("[PATHS] ROUTER_DIR (resolved) :", router_dir)
    print("[PATHS] EXPERT_ROOT (resolved):", expert_root)
    print("[PATHS] E5 local (env/auto) :", E5_LOCAL_DIR_ENV or _try_local_model("multilingual-e5-large") or "(HF fallback)")

    # E5
    print("Loading E5 retriever...")
    retriever = E5Retriever()

    # Corpus
    print(f"Preparing corpus with {len(corpus_dict)} documents...")
    retriever.corpus_ids = list(corpus_dict.keys())
    passages = [
        corpus_dict[doc_id].get("passage", corpus_dict[doc_id].get("text", ""))
        for doc_id in retriever.corpus_ids
    ]
    corpus_texts = {doc_id: passages[i] for i, doc_id in enumerate(retriever.corpus_ids)}

    # E5 cache
    cache_sig = _corpus_sig_for_cache(retriever.corpus_ids, passages, retriever.model_name)
    cached = _try_load_cache(cache_sig)
    if cached is not None:
        emb_e5 = cached["emb"]
        retriever.corpus_embeddings = emb_e5
        faiss_index = cached["index"]
        if cached["ids"] != retriever.corpus_ids:
            print("[CACHE] ID order mismatch; ignoring cache.")
            cached = None

    if cached is None:
        print("Computing E5 embeddings…")
        emb_e5 = retriever.embed_texts(passages, is_query=False, batch_size=256)
        retriever.corpus_embeddings = emb_e5
        faiss_index = _build_faiss_index(emb_e5)
        if FAISS_AVAILABLE and faiss_index is not None:
            _save_cache(cache_sig, retriever.corpus_ids, emb_e5, faiss_index)

    # Lexical
    try:
        if not USE_TFIDF_LEXICAL:
            lex_engine = GlobalLexBM25(passages)
            lex_type = "bm25"
            print("[LEX] BM25 lexical engine built.")
        else:
            raise ImportError("Forcing TF-IDF")
    except Exception:
        lex_engine = GlobalLexTFIDF(passages, max_feats=TFIDF_MAX_FEATS)
        lex_type = "tfidf"
        print("[LEX] TF-IDF lexical engine built (fallback).")

    # Router + experts (MoE)
    moe = None
    print(f"[MoE] USE_ROUTER_MOE={USE_ROUTER_MOE}")
    print(f"[MoE] router_dir={router_dir}")
    print(f"[MoE] expert_root={expert_root}")
    experts: Dict[str, CaseCE] = {}
    general_ce = None

    if USE_ROUTER_MOE and router_dir:
        try:
            router = HebRouter(router_dir)
            expert_dirs = _discover_expert_dirs(expert_root or "")
            print("[MoE] discovered experts:", expert_dirs)
            print(
                f"[MoE] expert max_len: kz={RERANK_MAX_LEN_KZ}, "
                f"wiki={RERANK_MAX_LEN_WIKI}, knesset={RERANK_MAX_LEN_KNESSET}"
            )
            max_len_map = {
                "kz": RERANK_MAX_LEN_KZ,
                "wiki": RERANK_MAX_LEN_WIKI,
                "knesset": RERANK_MAX_LEN_KNESSET,
            }
            for lab in ("kz", "wiki", "knesset"):
                if lab in expert_dirs:
                    experts[lab] = CaseCE(
                        expert_dirs[lab],
                        max_len=max_len_map.get(lab, RERANK_MAX_LEN),
                    )

            # --- NEW: load general_ce from env or ./models/ce_general ---
            gen_dir = GENERAL_CE_DIR_ENV or os.path.join(_models_root(), "ce_general")
            if os.path.isdir(gen_dir):
                print(f"[MoE] Loading general CE from {gen_dir}")
                general_ce = CaseCE(gen_dir, max_len=RERANK_MAX_LEN)
            else:
                print("[MoE] general_ce directory not found; skipping global CE.")

            if experts:
                moe = MoEReranker(
                    router,
                    experts,
                    calib_mode=CALIBRATION_MODE,
                    general_expert=general_ce,
                )
                print(f"[MoE] Router ready. Thresholds: HI={ROUTER_TAU_HI} MID={ROUTER_TAU_MID}")
            else:
                print("[MoE] No experts loaded after discovery.")
        except Exception as e:
            print("[MoE] Router/experts load failed:", e)

    if moe is None:
        print("[MoE] WARNING: MoE not available; system will fall back to Stage-1 retrieval only.")

    # Cross-encoder document pretokenization (shared tokenizer across experts)
    if moe is not None and experts:
        shared_ce_tok = next(iter(experts.values())).tok
        ce_tok_name = getattr(shared_ce_tok, "name_or_path", "(unknown)")
    else:
        shared_ce_tok = None
        ce_tok_name = "(none)"

    ce_doc_tokens: Dict[str, List[int]] = {}
    if shared_ce_tok is not None:
        N_docs = len(retriever.corpus_ids)
        print(f"[PRETOK] Building pretokenized passages with tokenizer: {ce_tok_name}")
        print(f"[PRETOK] Total docs to tokenize: {N_docs}")
        t0_pretok = time.perf_counter()
        for i, doc_id in enumerate(retriever.corpus_ids, 1):
            txt = corpus_texts.get(doc_id, "")
            ce_doc_tokens[doc_id] = shared_ce_tok.encode(txt or "", add_special_tokens=False)
            if (i % 10000) == 0 or i == N_docs:
                elapsed = time.perf_counter() - t0_pretok
                print(f"[PRETOK] {i:>7}/{N_docs} docs tokenized | {elapsed:.1f}s elapsed")

        moe.shared_tok = shared_ce_tok
        moe.doc_tok_map = ce_doc_tokens

    # Optional warmup
    if bool(int(os.getenv("WARMUP_MODELS", "1"))):
        warm_q = "בדיקת מערכת • warmup"
        warm_passages = [f"warmup passage {i} • טקסט בדיקה" for i in range(max(8, RERANK_BATCH))]
        try:
            _ = retriever.embed_texts([warm_q], is_query=True, batch_size=1)
            if faiss_index is not None:
                qe = retriever.embed_texts([warm_q], is_query=True, batch_size=1).astype(np.float32)
                faiss_index.search(qe, 1)
        except Exception as e:
            print("[WARMUP] E5/FAISS skipped:", e)

        if moe is not None:
            try:
                _ = moe.router.probs(warm_q)
            except Exception as e:
                print("[WARMUP] Router probs failed:", e)
            for lab, expert in getattr(moe, "experts", {}).items():
                try:
                    _ = expert.score(warm_q, warm_passages, bs=max(8, RERANK_BATCH))
                    print(f"[WARMUP] Expert '{lab}' ok.")
                except Exception as e:
                    print(f"[WARMUP] Expert '{lab}' failed:", e)
            if moe.general is not None:
                try:
                    _ = moe.general.score(warm_q, warm_passages, bs=max(8, RERANK_BATCH))
                    print("[WARMUP] general_ce ok.")
                except Exception as e:
                    print("[WARMUP] general_ce failed:", e)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("[WARMUP] Done.")

    print("✓ Preprocess complete.")
    print(f"✓ E5 emb: {retriever.corpus_embeddings.shape} | LEX: {lex_type} | MoE: {bool(moe)}")
    return {
        "retriever": retriever,
        "moe": moe,
        "corpus_ids": retriever.corpus_ids,
        "corpus_embeddings": retriever.corpus_embeddings,
        "corpus_texts": corpus_texts,
        "faiss_index": faiss_index,
        "lex_engine": lex_engine,
        "num_documents": len(corpus_dict),
        "ce_doc_tokens": ce_doc_tokens,
        "ce_tokenizer_name": ce_tok_name,
    }

# ----------------------------- #
# Stage-1 retrieval — fuse E5 + BM25/TFIDF
# ----------------------------- #
def _alloc_counts(total: int, avail_keys: List[str]) -> Dict[str, int]:
    if total <= 0 or not avail_keys:
        return {k: 0 for k in avail_keys}
    wsum = sum(SOURCE_WEIGHTS[k] for k in avail_keys)
    fracs = [(k, SOURCE_WEIGHTS[k] / wsum) for k in avail_keys]
    raw = [(k, total * f) for k, f in fracs]
    ints = {k: int(v) for k, v in raw}
    rem = total - sum(ints.values())
    frac_order = sorted(((k, v - int(v)) for k, v in raw), key=lambda x: x[1], reverse=True)
    for i in range(rem):
        ints[frac_order[i % len(frac_order)][0]] += 1
    return ints

def stage1_retrieve_ids(query_text: str, pre: dict, k: int = 100) -> List[str]:
    retr = pre["retriever"]
    lex = pre.get("lex_engine")
    qe_e5 = retr.embed_texts([query_text], is_query=True, batch_size=1).astype(np.float32)
    lex_scores = lex.scores_all(query_text) if lex is not None else None

    avail: List[str] = []
    if qe_e5 is not None:
        avail.append("e5")
    if lex_scores is not None:
        avail.append("lex")
    if not avail:
        return []

    alloc = _alloc_counts(k, avail)
    pairs: List[Tuple[int, float]] = []

    # E5 global
    k_e5 = alloc.get("e5", 0)
    if k_e5 > 0:
        idx = pre.get("faiss_index")
        if idx is not None:
            D, I = idx.search(qe_e5, min(k_e5, len(pre["corpus_ids"])))
            pairs += [(int(i), float(d)) for d, i in zip(D[0], I[0]) if i >= 0]
        else:
            sims = np.dot(qe_e5, pre["corpus_embeddings"].astype(np.float32).T)[0]
            ords = np.argsort(sims)[::-1][:k_e5]
            pairs += [(int(i), float(sims[i])) for i in ords]

    # Lexical global
    k_lex = alloc.get("lex", 0)
    if k_lex > 0 and lex_scores is not None:
        N = len(pre["corpus_ids"])
        take = min(k_lex, N)
        part = np.argpartition(lex_scores, -take)[-take:]
        ords = part[np.argsort(lex_scores[part])[::-1]]
        pairs += [(int(i), float(lex_scores[i])) for i in ords]

    # Deduplicate; keep best score
    best: Dict[int, float] = {}
    for gi, sc in pairs:
        if gi in best:
            if sc > best[gi]:
                best[gi] = sc
        else:
            best[gi] = sc

    cand_sorted = sorted(best.items(), key=lambda x: x[1], reverse=True)[:k]
    return [pre["corpus_ids"][gi] for gi, _ in cand_sorted]

# ----------------------------- #
# predict(): Stage-1 fused + MoE rerank (if available)
# ----------------------------- #
def predict(query: Dict[str, str], preprocessed_data: dict):
    global retriever, moe, corpus_texts

    query_text = query.get("query", "")
    if not query_text:
        return []

    start_ts = time.perf_counter()
    print(f"[QUERY] {query_text}")

    if retriever is None:
        retriever = preprocessed_data.get("retriever")
        moe = preprocessed_data.get("moe")
        corpus_texts = preprocessed_data.get("corpus_texts", {})
    if retriever is None:
        print("Error: Missing retriever in preprocessed data")
        elapsed = time.perf_counter() - start_ts
        print(f"[TIMING] predict elapsed: {elapsed:.3f}s | query: {query_text}")
        return []

    try:
        # Stage 1: retrieval
        candidate_ids = stage1_retrieve_ids(query_text, preprocessed_data, k=K_TOTAL)
        candidate_passages = [corpus_texts.get(doc_id, "") for doc_id in candidate_ids]

        # Stage 2: rerank via MoE (if available), else just use Stage-1 scores
        if preprocessed_data.get("moe") is not None and candidate_ids:
            pairs, _probs, _mode = preprocessed_data["moe"].rerank(
                query_text, candidate_passages, candidate_ids, top_k=20
            )
            if not pairs:
                # Fallback: use Stage-1 scores directly
                qe = retriever.embed_texts([query_text], is_query=True, batch_size=1)
                e5_scores = cosine_similarity(qe, preprocessed_data["corpus_embeddings"])[0]
                id_to_idx = {doc_id: i for i, doc_id in enumerate(preprocessed_data["corpus_ids"])}
                scores = [float(e5_scores[id_to_idx[pid]]) for pid in candidate_ids]
                pairs = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)[:20]
        else:
            # No MoE → global E5 cosine on full corpus, limited to top-20
            qe = retriever.embed_texts([query_text], is_query=True, batch_size=1)
            e5_scores = cosine_similarity(qe, preprocessed_data["corpus_embeddings"])[0]
            top = np.argsort(e5_scores)[::-1][:20]
            pairs = [
                (preprocessed_data["corpus_ids"][i], float(e5_scores[i]))
                for i in top
            ]

        results = [{"paragraph_uuid": pid, "score": float(score)} for pid, score in pairs]
        elapsed = time.perf_counter() - start_ts
        print(f"[TIMING] predict elapsed: {elapsed:.3f}s ")
        return results

    except Exception as e:
        print(f"Error in prediction: {e}")
        # Final fallback: global E5 cosine
        try:
            qe = retriever.embed_texts([query_text], is_query=True, batch_size=1)
            e5_scores = cosine_similarity(qe, preprocessed_data["corpus_embeddings"])[0]
            top = np.argsort(e5_scores)[::-1][:20]
            results = [
                {
                    "paragraph_uuid": preprocessed_data["corpus_ids"][i],
                    "score": float(e5_scores[i]),
                }
                for i in top
            ]
            elapsed = time.perf_counter() - start_ts
            print(f"[TIMING] predict elapsed: {elapsed:.3f}s (fallback) | query: {query_text}")
            return results
        except Exception:
            elapsed = time.perf_counter() - start_ts
            print(f"[TIMING] predict elapsed: {elapsed:.3f}s (fatal) | query: {query_text}")
            return []
