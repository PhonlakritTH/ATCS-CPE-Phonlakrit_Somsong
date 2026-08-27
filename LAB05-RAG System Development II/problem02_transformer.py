from data_loader import load_qa


def bow(text):
    result = {}
    for token in text.split():
        result[token] = result.get(token, 0) + 1
    return result


def with_position(text):
    return [(i, token) for i, token in enumerate(text.split())]


def find_pair(data):
    formal = next(d for d in data if d["lang"] == "ทางการ" and "VPN" in d["question"])
    slang = next(d for d in data if d["lang"] == "แสลง" and "VPN" in d["question"])
    return formal, slang


def run():
    data = load_qa()
    formal, slang = find_pair(data)

    print("Formal-register question:", formal["question"])
    print("Slang-register question :", slang["question"])
    print("BoW (formal):", bow(formal["question"]))
    print("BoW (slang) :", bow(slang["question"]))
    common = set(bow(formal["question"])) & set(bow(slang["question"]))
    print("Exact-token overlap:", common or "None")
    print("-> Both questions are about the same topic (what a VPN does and whether it's needed)")
    print("   but Keyword/BoW barely overlaps because the wording differs (Vocabulary Mismatch)")

    print("\nExample: effect of Token order on meaning (Position):")
    a = " ".join(formal["answer"].split()[:8])
    b = " ".join(reversed(a.split()))
    print("A (original):", with_position(a))
    print("B (reversed order):", with_position(b))
    print("BoW identical:", bow(a) == bow(b))

    print("\nCause: BoW does not capture vocabulary variation and does not preserve word order")
    print("Transformers use Positional Information + Self-Attention and semantic Embeddings")
    print("to match questions that use different wording but share the same intent (e.g. formal vs slang)")