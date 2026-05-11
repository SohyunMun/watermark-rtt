import re
import json
from config import WATERMARK_METHODS
from routes import ROUTES
import translator
import evaluator
import visualizer


def preprocess_input(raw):
    raw = raw.strip()
    # Case 1: valid JSON object — extract watermarked_completion or first string value
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            for key in ("watermarked_completion", "completion", "text", "output"):
                if key in data and isinstance(data[key], str):
                    return data[key].strip()
            for v in data.values():
                if isinstance(v, str):
                    return v.strip()
    except (json.JSONDecodeError, ValueError):
        pass
    # Case 2: inline "key": "value" patterns mixed with plain text
    cleaned = re.sub(r'"[^"]+"\s*:\s*"(?:[^"\\]|\\.)*"', '', raw)
    cleaned = re.sub(r'[{}\[\],]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned if cleaned else raw


def main():
    print("Watermark RTT Remover")
    print("=" * 40)
    print("Available methods:", ", ".join(WATERMARK_METHODS))
    raw = input("Select watermark method: ").strip()
    method = next((m for m in WATERMARK_METHODS if m.lower() == raw.lower()), None)

    if method is None:
        print(f"Error: '{method}' is not a supported method.")
        print(f"Supported: {', '.join(WATERMARK_METHODS)}")
        return

    text = preprocess_input(input("Enter watermarked English text: "))
    if not text:
        print("Error: No text provided.")
        return

    path = ROUTES[method]
    print(f"\nRTT path for {method}: EN -> {' -> '.join(path)} -> EN")

    print("\nRunning Round-Trip Translation...")
    rtt_text = translator.rtt(text, path)

    print("\nEvaluating metrics...")
    metrics = evaluator.evaluate(text, rtt_text, method)

    visualizer.visualize(text, rtt_text, metrics, method, path)


if __name__ == "__main__":
    main()
