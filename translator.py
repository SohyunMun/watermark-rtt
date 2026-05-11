import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from config import LANGUAGE_CODES, MODEL_NAME, MAX_TOKENS
from segmenter import chunk_text

_tokenizer = None
_model = None

# NLLB-600M practical translation quality limit.
# Technical max is 512 tokens, but translation quality degrades significantly
# beyond ~200 tokens for this model size. Chunking at 200 ensures full output.
RTT_CHUNK_LIMIT = 200


def load_model():
    global _tokenizer, _model
    if _model is None:
        print(f"Loading translation model ({MODEL_NAME})...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def _translate_chunk(text, src_lang, tgt_lang, tokenizer, model):
    # Normalize special characters that confuse NLLB
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('…', '...')
    text = re.sub(r'\(\d+\)', '', text)       # remove citation markers like (1)
    text = re.sub(r'\s+', ' ', text).strip()

    tokenizer.src_lang = src_lang
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_TOKENS,
    )
    tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_id,
            max_new_tokens=512,
            num_beams=4,
            repetition_penalty=1.3,
        )
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]


def translate(text, src_lang, tgt_lang):
    tokenizer, model = load_model()
    token_count = len(tokenizer.tokenize(text))
    if token_count > RTT_CHUNK_LIMIT:
        chunks = chunk_text(text, tokenizer, RTT_CHUNK_LIMIT)
        parts = [_translate_chunk(c, src_lang, tgt_lang, tokenizer, model) for c in chunks]
        return ' '.join(parts)
    return _translate_chunk(text, src_lang, tgt_lang, tokenizer, model)


def _rtt_chunk(text, path):
    langs = ["eng_Latn"] + [LANGUAGE_CODES[lc] for lc in path] + ["eng_Latn"]
    names = ["EN"] + list(path) + ["EN"]
    current = text
    for i in range(len(langs) - 1):
        print(f"  {names[i]} -> {names[i + 1]}...")
        current = translate(current, langs[i], langs[i + 1])
    return current


def rtt(text, path):
    tokenizer, _ = load_model()
    if not path:
        return text
    token_count = len(tokenizer.tokenize(text))
    if token_count > RTT_CHUNK_LIMIT:
        chunks = chunk_text(text, tokenizer, RTT_CHUNK_LIMIT)
        print(f"  [Input {token_count} tokens -> {len(chunks)} chunk(s)]")
        parts = []
        for idx, chunk in enumerate(chunks, 1):
            print(f"  [Chunk {idx}/{len(chunks)} started]")
            parts.append(_rtt_chunk(chunk, path))
            print(f"  [Chunk {idx}/{len(chunks)} translated]")
        return ' '.join(parts)
    return _rtt_chunk(text, path)