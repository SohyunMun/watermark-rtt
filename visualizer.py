import matplotlib.pyplot as plt
from datetime import datetime


def _save_result_txt(original, rtt_text, metrics, method, path):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path_str = "EN -> " + " -> ".join(path) + " -> EN"
    lines = [
        f"Date/Time : {now}",
        f"Method    : {method}",
        f"RTT Path  : {path_str}",
        "",
        "[ Original Text ]",
        original,
        "",
        "[ RTT Result ]",
        rtt_text,
        "",
        "[ Evaluation Metrics ]",
        f"  SBERT similarity : {metrics['sbert']:.4f}",
        f"  NLI entailment   : {metrics['nli']:.4f}",
        f"  PPL (original)   : {metrics['ppl_orig']:.2f}",
        f"  PPL (RTT Result)        : {metrics['ppl_rtt']:.2f}",
        f"  Z-Score (original)   : {metrics['z_score_orig']:.4f}",
        f"  Z-Score (RTT Result)    : {metrics['z_score_rtt']:.4f}",
    ]
    with open("result.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


_IDEAL_METRICS = {"SBERT", "NLI"}


def plot_metrics(before, after, method, path):
    path_str = "EN -> " + " -> ".join(path) + " -> EN"
    color_before = "#BBBBBB"
    color_after  = "#4A90D9"
    color_ideal  = "#D0E8FF"   # light blue for ideal baseline
    edge_ideal   = "#7AAAD0"   # dashed border color

    metric_labels = ["SBERT", "NLI", "PPL", "Z-Score"]
    directions    = [
        "↑ higher is better",
        "↑ higher is better",
        "↓ lower is better",
        "↓ lower is better",
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Watermark RTT Removal  |  Method: {method}  |  Path: {path_str}",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    for ax, metric, direction in zip(axes, metric_labels, directions):
        vals = [before.get(metric, 0), after.get(metric, 0)]
        is_ideal = metric in _IDEAL_METRICS
        x_labels = (
            ["ideal\n(self-similarity)", "After"] if is_ideal else ["Before", "After"]
        )
        bars = ax.bar(
            x_labels,
            vals,
            color=[color_ideal if is_ideal else color_before, color_after],
            width=0.45,
            edgecolor="white",
        )
        if is_ideal:
            bars[0].set_edgecolor(edge_ideal)
            bars[0].set_linewidth(1.5)
            bars[0].set_linestyle("--")

        ax.set_title(metric, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel(direction, fontsize=10, color="#555555")
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=11)

        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.2f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=11,
                fontweight="bold",
            )

        lo, hi = min(vals), max(vals)
        margin = (hi - lo) * 0.2 if hi != lo else abs(hi) * 0.2 or 1.0
        ax.set_ylim(lo - margin, hi + margin)

    plt.tight_layout()
    plt.savefig("result.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def print_results(original, rtt_text, metrics, method, path):
    path_str = "EN -> " + " -> ".join(path) + " -> EN"
    print("\n" + "=" * 55)
    print(f"Method : {method}  |  Path : {path_str}")
    print("=" * 55)
    print("[Original]")
    print(original)
    print("\n[RTT Result]")
    print(rtt_text)
    print("\n" + "-" * 55)
    print("Evaluation Metrics:")
    print("-" * 55)
    print(f"  SBERT similarity : {metrics['sbert']:.4f}   ↑ higher is better")
    print(f"  NLI entailment   : {metrics['nli']:.4f}   ↑ higher is better")
    print(f"  PPL (original)   : {metrics['ppl_orig']:.2f}   ↓ lower is better")
    print(f"  PPL (RTT)        : {metrics['ppl_rtt']:.2f}   ↓ lower is better")
    print(f"  Z-Score (orig)   : {metrics['z_score_orig']:.4f}   ↓ lower is better")
    print(f"  Z-Score (RTT)    : {metrics['z_score_rtt']:.4f}   ↓ lower is better")
    print("-" * 55)


def visualize(original, rtt_text, metrics, method, path):
    print_results(original, rtt_text, metrics, method, path)
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
    plot_metrics(before, after, method, path)
    print("Graph saved to result.png")
    _save_result_txt(original, rtt_text, metrics, method, path)
    print("Result saved to result.txt")
