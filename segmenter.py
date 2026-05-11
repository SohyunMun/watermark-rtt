import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


def chunk_text(text, tokenizer, max_tokens=200):
    sentences = nltk.sent_tokenize(text)
    limit = max_tokens - 10  # headroom for special tokens

    chunks = []
    current = []
    current_len = 0

    for sent in sentences:
        sent_tokens = tokenizer.tokenize(sent)
        sent_len = len(sent_tokens)

        if sent_len > limit:
            # Single sentence exceeds limit: split by words
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            words = sent.split()
            sub_chunk = []
            sub_len = 0
            for word in words:
                word_len = len(tokenizer.tokenize(word))
                if sub_chunk and sub_len + word_len > limit:
                    chunks.append(" ".join(sub_chunk))
                    sub_chunk = [word]
                    sub_len = word_len
                else:
                    sub_chunk.append(word)
                    sub_len += word_len
            if sub_chunk:
                chunks.append(" ".join(sub_chunk))
        elif current and current_len + sent_len > limit:
            chunks.append(" ".join(current))
            current = [sent]
            current_len = sent_len
        else:
            current.append(sent)
            current_len += sent_len

    if current:
        chunks.append(" ".join(current))

    return chunks