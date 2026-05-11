import os
import gradio as gr
from config import WATERMARK_METHODS
from routes import ROUTES
import translator
import evaluator
import visualizer
from main import preprocess_input

_METRICS_DESC = """
**평가 지표 설명**

| 지표 | 설명 | 기준 |
|------|------|------|
| **Z-Score** | 워터마크 탐지 강도 (기법별 green list 기반 통계 검정) | 낮을수록 워터마크 제거 ↑ |
| **SBERT** | 원본과 RTT 결과의 의미적 유사도 (문장 임베딩 코사인 유사도) | 높을수록 의미 보존 ↑ |
| **NLI** | 원본이 RTT 결과를 함의(entailment)하는 확률 | 높을수록 의미 보존 ↑ |
| **PPL** | 언어 모델 기준 텍스트 자연스러움 (Perplexity) | 낮을수록 자연스러운 문장 ↓ |

"""


def run_rtt(method, raw_text):
    text = preprocess_input(raw_text)
    if not text:
        return "No text provided.", "", None

    path = ROUTES[method]
    rtt_text = translator.rtt(text, path)
    metrics = evaluator.evaluate(text, rtt_text, method)

    before = {
        "SBERT":   1.0,
        "NLI":     1.0,
        "PPL":     metrics["ppl_orig"],
        "Z-Score": metrics["z_score_orig"],
    }
    after = {
        "SBERT":   metrics["sbert"],
        "NLI":     metrics["nli"],
        "PPL":     metrics["ppl_rtt"],
        "Z-Score": metrics["z_score_rtt"],
    }
    visualizer.plot_metrics(before, after, method, path)
    visualizer._save_result_txt(text, rtt_text, metrics, method, path)

    metrics_str = (
        f"Z-Score (입력) : {metrics['z_score_orig']:.4f}\n"
        f"Z-Score (RTT 결과)  : {metrics['z_score_rtt']:.4f}\n"
        f"SBERT          : {metrics['sbert']:.4f}\n"
        f"NLI            : {metrics['nli']:.4f}\n"
        f"PPL (입력)     : {metrics['ppl_orig']:.2f}\n"
        f"PPL (RTT 결과)      : {metrics['ppl_rtt']:.2f}"
        
    )

    return rtt_text, metrics_str, os.path.abspath("result.png")


_theme = gr.themes.Soft(
    font=[
        gr.themes.GoogleFont("Inter"),
        "ui-sans-serif",
        "system-ui",
        "sans-serif",
    ],
    font_mono=[
        gr.themes.GoogleFont("JetBrains Mono"),
        "ui-monospace",
        "monospace",
    ],
)

with gr.Blocks(title="Watermark RTT Remover", theme=_theme) as demo:
    gr.Markdown("# Watermark RTT Remover")
    gr.Markdown(
        "Remove LLM text watermarks using Round-Trip Translation (RTT). "
        "Select a watermark method, paste the watermarked text, and click **Run RTT**."
    )

    with gr.Row():
        with gr.Column():
            method_input = gr.Dropdown(
                choices=WATERMARK_METHODS,
                value=WATERMARK_METHODS[0],
                label="Watermark Method",
            )
            text_input = gr.Textbox(
                lines=10,
                placeholder=(
                    "Paste watermarked English text here...\n\n"
                    "Example: The quick brown fox jumps over the lazy dog. "
                    "It was a bright cold day in April, and the clocks were striking thirteen."
                ),
                label="Input Text",
            )
            run_btn = gr.Button("Run RTT", variant="primary", size="lg")

        with gr.Column():
            text_output = gr.Textbox(
                lines=10,
                label="RTT Result",
                interactive=False,
            )
            gr.Markdown(_METRICS_DESC)
            metrics_output = gr.Textbox(
                lines=7,
                label="측정값",
                interactive=False,
            )

    image_output = gr.Image(label="Metric Comparison (Before vs After RTT)")

    run_btn.click(
        fn=run_rtt,
        inputs=[method_input, text_input],
        outputs=[text_output, metrics_output, image_output],
    )


if __name__ == "__main__":
    demo.launch()
