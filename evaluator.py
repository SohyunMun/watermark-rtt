import math
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from sentence_transformers import SentenceTransformer, util

# ── Model names ───────────────────────────────────────────────────────────────
_OPT_MODEL   = "facebook/opt-125m"
_SBERT_MODEL = "all-MiniLM-L6-v2"
_NLI_MODEL   = "cross-encoder/nli-deberta-v3-small"

# ── Watermark hyperparameters (matches eval_rtt_pipeline_v2.ipynb) ────────────
KGW_GAMMA    = 0.25
UNI_FRACTION = 0.5
UNI_SEED     = 42
SID_NGRAM    = 5
SID_NUM_KEYS = 20
WM_KEYS      = list(range(SID_NUM_KEYS))
AWT_SEED     = 42

# ── Lazy-loaded model globals ─────────────────────────────────────────────────
_opt_tokenizer = None
_opt_model     = None
_sbert_model   = None
_nli_tokenizer = None
_nli_model     = None
VOCAB_SIZE     = None

# Pre-computed green sets (initialized on first OPT load)
_UNI_GREEN  = None
_AWT_GREEN  = None


def _load_opt():
    global _opt_tokenizer, _opt_model, VOCAB_SIZE, _UNI_GREEN, _AWT_GREEN
    if _opt_model is None:
        _opt_tokenizer = AutoTokenizer.from_pretrained(_OPT_MODEL)
        if _opt_tokenizer.pad_token is None:
            _opt_tokenizer.pad_token = _opt_tokenizer.eos_token
        _opt_model = AutoModelForCausalLM.from_pretrained(_OPT_MODEL)
        _opt_model.eval()
        VOCAB_SIZE = len(_opt_tokenizer)

        # Pre-compute Unigram green set (fixed seed, computed once)
        _uni_rng = torch.Generator()
        _uni_rng.manual_seed(UNI_SEED)
        _UNI_GREEN = set(
            torch.randperm(VOCAB_SIZE, generator=_uni_rng)
            [:int(UNI_FRACTION * VOCAB_SIZE)].tolist()
        )

        # Pre-compute AWT green set (fixed seed, half vocab)
        _awt_rng = torch.Generator()
        _awt_rng.manual_seed(AWT_SEED)
        _AWT_GREEN = set(
            torch.randperm(VOCAB_SIZE, generator=_awt_rng)
            [:VOCAB_SIZE // 2].tolist()
        )

    return _opt_tokenizer, _opt_model


def _load_sbert():
    global _sbert_model
    if _sbert_model is None:
        _sbert_model = SentenceTransformer(_SBERT_MODEL)
    return _sbert_model


def _load_nli():
    global _nli_tokenizer, _nli_model
    if _nli_model is None:
        _nli_tokenizer = AutoTokenizer.from_pretrained(_NLI_MODEL)
        _nli_model = AutoModelForSequenceClassification.from_pretrained(
            _NLI_MODEL, torch_dtype=torch.float32
        )
        _nli_model.eval()
    return _nli_tokenizer, _nli_model


def _tokenize(text: str) -> list[int]:
    """Tokenize without special tokens, truncated to 512. Matches notebook."""
    tokenizer, _ = _load_opt()
    return tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=512,
    )


# ── PPL ───────────────────────────────────────────────────────────────────────

def calc_ppl(text: str) -> float:
    """
    PPL using shift approach (matches notebook compute_ppl_batch).
    Padding tokens masked with -100 so they're excluded from loss.
    """
    tokenizer, model = _load_opt()
    enc = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )
    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    with torch.inference_mode():
        out    = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits  # (1, T, V)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    token_loss = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    ).view(shift_labels.size())

    valid_mask  = (shift_labels != -100).float()
    sample_loss = (token_loss * valid_mask).sum() / valid_mask.sum().clamp(min=1)

    return round(math.exp(min(sample_loss.item(), 20)), 4)


# ── Z-Score / WM-Score ────────────────────────────────────────────────────────

def _zscore_kgw(text: str) -> float:
    """KGW z-score. Seed = prev_token_id % 2^31 (matches notebook)."""
    _load_opt()
    ids = _tokenize(text)
    T = len(ids)
    if T < 2:
        return 0.0
    gl_size = int(KGW_GAMMA * VOCAB_SIZE)

    def _green_list(prev_id):
        rng = torch.Generator()
        rng.manual_seed(int(prev_id) % (2 ** 31))
        return set(torch.randperm(VOCAB_SIZE, generator=rng)[:gl_size].tolist())

    gc = sum(1 for i in range(1, T) if ids[i] in _green_list(ids[i - 1]))
    z  = (gc - KGW_GAMMA * (T - 1)) / math.sqrt((T - 1) * KGW_GAMMA * (1 - KGW_GAMMA))
    return round(z, 4)


def _zscore_unigram(text: str) -> float:
    """Unigram z-score. Fixed green set pre-computed at load time."""
    _load_opt()
    ids = _tokenize(text)
    T = len(ids)
    if T == 0:
        return 0.0
    gc = sum(1 for t in ids if t in _UNI_GREEN)
    z  = (gc - UNI_FRACTION * T) / math.sqrt(T * UNI_FRACTION * (1 - UNI_FRACTION))
    return round(z, 4)


def _zscore_synthid(text: str) -> float:
    """
    SynthID z-score (matches notebook zscore_synthid_batch).
    Uses hash(ngram) context key + per-key torch.rand mask.
    """
    _load_opt()
    ids = _tokenize(text)
    T = len(ids)
    if T < SID_NGRAM:
        return 0.0

    g_vals = []
    vs = VOCAB_SIZE
    for i in range(SID_NGRAM - 1, T):
        ngram   = tuple(ids[i - SID_NGRAM + 1 : i])
        ctx_key = hash(ngram) % (2 ** 31)
        for key in WM_KEYS:
            rng = torch.Generator()
            rng.manual_seed((ctx_key + key * 31337) % (2 ** 31))
            mask = (torch.rand(vs, generator=rng) > 0.5).float()
            g_vals.append(mask[ids[i]].item())

    g_bar = float(np.mean(g_vals))
    z     = (g_bar - 0.5) / (0.5 / math.sqrt(len(g_vals)))
    return round(z, 4)


def _wmscore_awt(text: str) -> float:
    """
    AWT wm_score: fraction of tokens in green set (0~1).
    Matches notebook wmscore_awt_batch. Returns ratio, not z-score.
    """
    _load_opt()
    ids = _tokenize(text)
    if not ids:
        return 0.0
    score = sum(1 for t in ids if t in _AWT_GREEN) / len(ids)
    return round(score, 4)


_WM_FN = {
    "KGW":     _zscore_kgw,
    "Unigram": _zscore_unigram,
    "SynthID": _zscore_synthid,
    "AWT":     _wmscore_awt,
}


def calc_wmscore(text: str, method: str) -> float:
    try:
        return _WM_FN.get(method, _zscore_kgw)(text)
    except Exception as e:
        print(f"WM-score error ({method}): {e}")
        return 0.0


# ── SBERT ─────────────────────────────────────────────────────────────────────

def calc_sbert(original: str, translated: str) -> float:
    model = _load_sbert()
    emb_a, emb_b = model.encode([original, translated], convert_to_tensor=True)
    return round(float(max(0.0, min(1.0, util.cos_sim(emb_a, emb_b).item()))), 4)


# ── NLI ───────────────────────────────────────────────────────────────────────

def calc_nli(original: str, translated: str) -> float:
    """
    Entailment probability using AutoModelForSequenceClassification.
    Labels: 0=contradiction, 1=neutral, 2=entailment (matches notebook).
    """
    nli_tok, nli_model = _load_nli()
    inputs = nli_tok(
        original, translated,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )
    with torch.inference_mode():
        logits = nli_model(**inputs).logits  # (1, 3)
    probs = torch.softmax(logits.float(), dim=-1)[0]
    return round(float(probs[1]), 4)  # index 1 = entailment


# ── evaluate ──────────────────────────────────────────────────────────────────

def evaluate(original: str, translated: str, method: str) -> dict:
    return {
        "z_score_orig": calc_wmscore(original, method),
        "z_score_rtt":  calc_wmscore(translated, method),
        "ppl_orig":     calc_ppl(original),
        "ppl_rtt":      calc_ppl(translated),
        "sbert":        calc_sbert(original, translated),
        "nli":          calc_nli(original, translated),
    }