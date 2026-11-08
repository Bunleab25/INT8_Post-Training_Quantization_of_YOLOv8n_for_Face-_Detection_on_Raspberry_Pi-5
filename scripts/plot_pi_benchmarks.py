#!/usr/bin/env python3
"""Generate publication-grade Raspberry Pi 5 benchmark plots from JSON telemetry.

Reads:
  - reports/pi_benchmark.json (150-run multi-core thread sweep + FP32/INT8 comparison)
  - reports/pi_benchmark_stress_1000runs.json (1,000-run sustained thermal stress test)

Outputs:
  - reports/pi_thread_scaling_thermal.png
  - reports/pi_burst_vs_endurance.png
  - reports/pi_stage_breakdown.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scripts.common import REPORTS as REPORTS_DIR, ROOT, load_json
except ImportError:
    from common import REPORTS as REPORTS_DIR, ROOT, load_json


def plot_thread_scaling_and_thermal(bench_150: dict, out_path: Path) -> None:
    sweep = bench_150["int8_thread_sweep"]
    threads = [item["threads"] for item in sweep]
    fps_med = [item["fps_median"] for item in sweep]
    lat_med = [item["total_ms_median"] for item in sweep]
    lat_p95 = [item["total_ms_p95"] for item in sweep]
    temp_start = [item["temp_start_c"] for item in sweep]
    temp_end = [item["temp_end_c"] for item in sweep]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=200)
    fig.patch.set_facecolor("white")

    # --- Panel A: Thread Scaling (Throughput & Latency) ---
    x = np.arange(len(threads))
    width = 0.36

    bars_fps = ax1.bar(
        x - width / 2,
        fps_med,
        width,
        color=["#4C72B0", "#4C72B0", "#2CA02C", "#D62728"],
        edgecolor="#1a1a1a",
        linewidth=0.8,
        label="Throughput (Median FPS)",
        zorder=3,
    )
    ax1.set_xlabel("ONNX Runtime Intra-Op CPU Threads (Cortex-A76 Cores)", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Throughput (FPS)  [Higher is Better]", fontsize=10.5, fontweight="bold", color="#1f4e78")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{t} Thread{'s' if t > 1 else ''}" for t in threads], fontsize=10)
    ax1.set_ylim(0, 22)
    ax1.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)

    for bar, fps in zip(bars_fps, fps_med):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.45,
            f"{fps:.1f} FPS",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#1a1a1a",
        )

    ax1_lat = ax1.twinx()
    ax1_lat.plot(
        x + width / 2,
        lat_med,
        marker="o",
        markersize=7,
        linewidth=2.2,
        color="#FF7F0E",
        label="End-to-End Median (ms)",
        zorder=4,
    )
    ax1_lat.plot(
        x + width / 2,
        lat_p95,
        marker="s",
        markersize=6,
        linewidth=1.6,
        linestyle="--",
        color="#9467BD",
        label="End-to-End p95 (ms)",
        zorder=4,
    )
    ax1_lat.set_ylabel("End-to-End Latency (ms)  [Lower is Better]", fontsize=10.5, fontweight="bold", color="#b85a00")
    ax1_lat.set_ylim(40, 100)

    for xi, med in zip(x + width / 2, lat_med):
        ax1_lat.annotate(
            f"{med:.1f} ms",
            (xi, med),
            textcoords="offset points",
            xytext=(0, -15),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color="#b85a00",
        )

    opt_idx = int(np.argmax(fps_med))
    ax1.annotate(
        f"Optimal Sweet Spot\n({fps_med[opt_idx]:.1f} FPS / {lat_med[opt_idx]:.1f} ms)",
        xy=(opt_idx - width / 2, fps_med[opt_idx]),
        xytext=(1.15, 19.3),
        arrowprops=dict(arrowstyle="->", color="#2CA02C", lw=1.5),
        fontsize=9,
        fontweight="bold",
        color="#1b691b",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f8e8", edgecolor="#2CA02C", lw=1),
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_lat.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8.5, framealpha=0.92)
    ax1.set_title("(a) Multi-Core Thread Scaling (best_int8.onnx)", fontsize=11.5, fontweight="bold", pad=10)

    # --- Panel B: Thermal Trajectory Across Thread Sweep ---
    ax2.plot(
        x,
        temp_start,
        marker="^",
        markersize=7,
        linewidth=2.0,
        linestyle="--",
        color="#1F77B4",
        label="Run Start Temp (°C)",
        zorder=4,
    )
    ax2.plot(
        x,
        temp_end,
        marker="o",
        markersize=8,
        linewidth=2.5,
        color="#D62728",
        label="Run End Temp (°C)",
        zorder=5,
    )

    for i in range(len(threads)):
        ax2.vlines(x[i], temp_start[i], temp_end[i], color="#D62728", linestyle=":", alpha=0.6, linewidth=1.8)
        start_offset_y = -2.4 if i < 3 else -2.8
        ax2.text(
            x[i] - 0.06,
            temp_start[i] + start_offset_y,
            f"{temp_start[i]:.1f}°C",
            ha="center",
            fontsize=8.5,
            color="#1F77B4",
        )
        end_offset_x = -0.18 if i == 2 else 0.04
        end_offset_y = 0.9 if i == 2 else 1.2
        ax2.text(
            x[i] + end_offset_x,
            temp_end[i] + end_offset_y,
            f"{temp_end[i]:.1f}°C",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#B2182B",
        )

    ax2.axhline(80.0, color="#FF7F0E", linestyle="--", linewidth=1.5, label="BCM2712 Soft Thermal Limit (80°C)")
    ax2.axhline(85.0, color="#8C000F", linestyle="-.", linewidth=1.5, label="BCM2712 Hard Throttle Limit (85°C)")
    ax2.axhspan(80.0, 91.5, color="#FFE5E5", alpha=0.45, zorder=1)

    throttled_str = bench_150.get("throttled_state") or "throttled"
    ax2.annotate(
        f"{threads[-1]}T starts at {temp_start[-1]:.2f}°C -> Throttled!\n(vcgencmd: {throttled_str})",
        xy=(len(threads) - 1, temp_end[-1]),
        xytext=(0.65, 86.6),
        arrowprops=dict(arrowstyle="->", color="#8C000F", lw=1.5, connectionstyle="arc3,rad=-0.12"),
        fontsize=8.8,
        fontweight="bold",
        color="#8C000F",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff0f0", edgecolor="#8C000F", lw=1),
    )

    ax2.set_xlabel("Benchmark Execution Sequence (1T -> 2T -> 3T -> 4T)", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("SoC Temperature (°C)", fontsize=10.5, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{t} Thread{'s' if t > 1 else ''}" for t in threads], fontsize=10)
    ax2.set_ylim(55, 91.5)
    ax2.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax2.legend(loc="lower right", fontsize=8.5, framealpha=0.92)
    ax2.set_title("(b) SoC Thermal Accumulation & Throttling Thresholds", fontsize=11.5, fontweight="bold", pad=10)

    fig.suptitle(
        "Raspberry Pi 5 (BCM2712 Cortex-A76): INT8 Multi-Core Scaling vs. Thermal Throttling",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_burst_vs_endurance(bench_150: dict, bench_1000: dict, out_path: Path) -> None:
    fp32_150 = bench_150["fp32_baseline"]
    int8_150 = bench_150["int8_primary"]
    fp32_1000 = bench_1000["fp32_baseline"]
    int8_1000 = bench_1000["int8_primary"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.5), dpi=200)
    fig.patch.set_facecolor("white")

    # --- Panel A: End-to-End Latency (p50, p95, p99) ---
    categories = ["150-Run Burst\n(Short Workload)", "1,000-Run Endurance\n(Sustained Stress)"]
    x = np.arange(len(categories))
    width = 0.34

    fp32_meds = [fp32_150["total_ms_median"], fp32_1000["total_ms_median"]]
    int8_meds = [int8_150["total_ms_median"], int8_1000["total_ms_median"]]

    fp32_p95s = [fp32_150["total_ms_p95"], fp32_1000["total_ms_p95"]]
    int8_p95s = [int8_150["total_ms_p95"], int8_1000["total_ms_p95"]]

    fp32_p99s = [fp32_150["total_ms_p99"], fp32_1000["total_ms_p99"]]
    int8_p99s = [int8_150["total_ms_p99"], int8_1000["total_ms_p99"]]

    b1 = ax1.bar(
        x - width / 2,
        fp32_meds,
        width,
        color="#E15759",
        edgecolor="#1a1a1a",
        linewidth=0.8,
        label="FP32 ONNX (best.onnx) — Median",
        zorder=3,
    )
    b2 = ax1.bar(
        x + width / 2,
        int8_meds,
        width,
        color="#4E79A7",
        edgecolor="#1a1a1a",
        linewidth=0.8,
        label="INT8 ONNX (best_int8.onnx) — Median",
        zorder=3,
    )

    # Plot p95 and p99 markers
    ax1.scatter(
        x - width / 2,
        fp32_p95s,
        marker="D",
        s=45,
        color="#8C000F",
        zorder=5,
        label="p95 Tail Latency",
    )
    ax1.scatter(
        x + width / 2,
        int8_p95s,
        marker="D",
        s=45,
        color="#8C000F",
        zorder=5,
    )
    ax1.scatter(
        x - width / 2,
        fp32_p99s,
        marker="^",
        s=55,
        color="#2B0000",
        zorder=5,
        label="p99 Tail Latency",
    )
    ax1.scatter(
        x + width / 2,
        int8_p99s,
        marker="^",
        s=55,
        color="#2B0000",
        zorder=5,
    )

    for i in range(2):
        ax1.vlines(x[i] - width / 2, fp32_meds[i], fp32_p99s[i], color="#2B0000", linestyle="--", linewidth=1.2, zorder=4)
        ax1.vlines(x[i] + width / 2, int8_meds[i], int8_p99s[i], color="#2B0000", linestyle="--", linewidth=1.2, zorder=4)

    for bar, med, p95 in zip(b1, fp32_meds, fp32_p95s):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            med / 2,
            f"p50:\n{med:.1f} ms",
            ha="center",
            va="center",
            fontsize=8.8,
            fontweight="bold",
            color="white",
        )
    for bar, med, p95 in zip(b2, int8_meds, int8_p95s):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            med / 2,
            f"p50:\n{med:.1f} ms",
            ha="center",
            va="center",
            fontsize=8.8,
            fontweight="bold",
            color="white",
        )

    speedup_150 = fp32_meds[0] / int8_meds[0] if int8_meds[0] else 1.0
    speedup_1000 = fp32_meds[1] / int8_meds[1] if int8_meds[1] else 1.0
    ax1.annotate(
        f"{speedup_150:.2f}x Faster\n(Burst)",
        xy=(0 + width / 2 + 0.05, int8_meds[0]),
        xytext=(0.14, 96),
        arrowprops=dict(arrowstyle="->", color="#1f4e78", lw=1.4),
        fontsize=9,
        fontweight="bold",
        color="#1f4e78",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#eef5fc", edgecolor="#4E79A7"),
    )
    ax1.annotate(
        f"{speedup_1000:.2f}x Faster!\n(Sustained Stress)",
        xy=(1 + width / 2 + 0.08, int8_meds[1]),
        xytext=(1.02, 106),
        arrowprops=dict(arrowstyle="->", color="#1b691b", lw=1.6, connectionstyle="arc3,rad=-0.15"),
        fontsize=9.5,
        fontweight="bold",
        color="#1b691b",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f8e8", edgecolor="#2CA02C"),
    )

    ax1.set_ylabel("End-to-End Latency (ms)  [Lower is Better]", fontsize=10.5, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 148)
    ax1.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.92)
    ax1.set_title("(a) End-to-End Latency & Tail Jitter (p50 / p95 / p99)", fontsize=11.5, fontweight="bold", pad=10)

    # --- Panel B: Throughput Collapse vs. Peak Temperature ---
    fp32_fps = [fp32_150["fps_median"], fp32_1000["fps_median"]]
    int8_fps = [int8_150["fps_median"], int8_1000["fps_median"]]
    fp32_temp = [fp32_150["temp_end_c"], fp32_1000["temp_end_c"]]
    int8_temp = [int8_150["temp_end_c"], int8_1000["temp_end_c"]]

    b3 = ax2.bar(
        x - width / 2,
        fp32_fps,
        width,
        color="#E15759",
        edgecolor="#1a1a1a",
        linewidth=0.8,
        label="FP32 Throughput (FPS)",
        zorder=3,
    )
    b4 = ax2.bar(
        x + width / 2,
        int8_fps,
        width,
        color="#4E79A7",
        edgecolor="#1a1a1a",
        linewidth=0.8,
        label="INT8 Throughput (FPS)",
        zorder=3,
    )

    for bar, fps, temp in zip(b3, fp32_fps, fp32_temp):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{fps:.1f} FPS\n({temp:.1f}°C)",
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
            color="#9e1b1e",
        )
    for bar, fps, temp in zip(b4, int8_fps, int8_temp):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{fps:.1f} FPS\n({temp:.1f}°C)",
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
            color="#1f4e78",
        )

    fp32_drop_pct = ((fp32_fps[1] - fp32_fps[0]) / fp32_fps[0] * 100.0) if fp32_fps[0] else 0.0
    int8_gain_pct = round(((int8_fps[1] - fp32_fps[1]) / fp32_fps[1]) * 100.0) if fp32_fps[1] else 0
    ax2.annotate(
        f"FP32 drops {fp32_drop_pct:.1f}% FPS\n({fp32_temp[1]:.2f}°C Severe Throttle)",
        xy=(1 - width / 2, fp32_fps[1] - 0.5),
        xytext=(0.08, 4.0),
        arrowprops=dict(arrowstyle="->", color="#9e1b1e", lw=1.5),
        fontsize=8.8,
        fontweight="bold",
        color="#9e1b1e",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff0f0", edgecolor="#E15759"),
    )
    ax2.annotate(
        f"INT8 holds {int8_fps[1]:.1f} FPS\n(+{int8_gain_pct}% over FP32)",
        xy=(1 + width / 2, int8_fps[1] + 2.4),
        xytext=(0.52, 18.8),
        arrowprops=dict(arrowstyle="->", color="#1f4e78", lw=1.5),
        fontsize=8.8,
        fontweight="bold",
        color="#1f4e78",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#eef5fc", edgecolor="#4E79A7"),
    )

    ax2.set_ylabel("Throughput (Median FPS)  [Higher is Better]", fontsize=10.5, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 22)
    ax2.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
    ax2.legend(loc="upper left", fontsize=8.8, framealpha=0.92)
    ax2.set_title("(b) Sustained Throughput & Peak SoC Temperature (4 Threads)", fontsize=11.5, fontweight="bold", pad=10)

    fig.suptitle(
        "Raspberry Pi 5 Burst (150 Runs) vs. Sustained Endurance (1,000 Runs): FP32 vs. INT8",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_stage_and_footprint(bench_150: dict, out_path: Path) -> None:
    int8_150 = bench_150["int8_primary"]
    fp32_150 = bench_150["fp32_baseline"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.8), dpi=200)
    fig.patch.set_facecolor("white")

    # --- Panel A: Stage-by-Stage Latency Comparison (FP32 vs INT8) ---
    models = ["FP32 ONNX\n(best.onnx)", "INT8 ONNX\n(best_int8.onnx)"]
    pre = np.array([fp32_150["preprocess_ms_median"], int8_150["preprocess_ms_median"]])
    inf = np.array([fp32_150["inference_ms_median"], int8_150["inference_ms_median"]])
    post = np.array([fp32_150["postprocess_ms_median"], int8_150["postprocess_ms_median"]])

    y = np.arange(len(models))
    height = 0.42

    ax1.barh(y, pre, height, color="#76B7B2", edgecolor="#1a1a1a", label="1. Preprocess (letterbox + norm)")
    ax1.barh(y, inf, height, left=pre, color="#4E79A7", edgecolor="#1a1a1a", label="2. ONNX Runtime Inference")
    ax1.barh(y, post, height, left=pre + inf, color="#F28E2B", edgecolor="#1a1a1a", label="3. Postprocess (NumPy NMS)")

    for i in range(len(models)):
        total = pre[i] + inf[i] + post[i]
        ax1.text(
            pre[i] + inf[i] / 2,
            y[i],
            f"Inference: {inf[i]:.2f} ms ({inf[i]/total*100:.1f}%)",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=9.2,
        )
        ax1.text(
            total + 1.2,
            y[i],
            f"Total: {total:.2f} ms\n(Pre: {pre[i]:.2f} ms, NMS: {post[i]:.2f} ms)",
            ha="left",
            va="center",
            color="#1a1a1a",
            fontweight="bold",
            fontsize=8.5,
        )

    ax1.set_yticks(y)
    ax1.set_yticklabels(models, fontsize=10, fontweight="bold")
    ax1.set_xlabel("Median Stage Latency (ms) on Raspberry Pi 5", fontsize=10.5, fontweight="bold")
    ax1.set_xlim(0, 128)
    ax1.set_ylim(-0.45, 1.55)
    ax1.grid(axis="x", linestyle="--", alpha=0.45, zorder=0)
    ax1.legend(loc="center right", bbox_to_anchor=(0.99, 0.50), fontsize=8.5, framealpha=0.95)
    ax1.set_title("(a) Pipeline Stage Breakdown (96.7% Compute-Bound)", fontsize=11.5, fontweight="bold", pad=10)

    # --- Panel B: Model Compression & Edge Memory Footprint ---
    metrics = ["Serialized Model Size\n(MB)", "Peak Process RSS RAM\n(MB / 10)"]
    fp32_vals = [fp32_150["model_size_mb"], fp32_150["peak_rss_mb"] / 10.0]
    int8_vals = [int8_150["model_size_mb"], int8_150["peak_rss_mb"] / 10.0]

    x = np.arange(len(metrics))
    width = 0.34
    b1 = ax2.bar(x - width / 2, fp32_vals, width, color="#E15759", edgecolor="#1a1a1a", label="FP32 ONNX (best.onnx)")
    b2 = ax2.bar(x + width / 2, int8_vals, width, color="#59A14F", edgecolor="#1a1a1a", label="INT8 ONNX (best_int8.onnx)")

    size_ratio = fp32_150["model_size_mb"] / int8_150["model_size_mb"] if int8_150["model_size_mb"] else 1.0
    ax2.text(
        b1[0].get_x() + width / 2,
        b1[0].get_height() + 0.35,
        f"{fp32_150['model_size_mb']:.2f} MB",
        ha="center",
        fontweight="bold",
        fontsize=8.8,
    )
    ax2.text(
        b2[0].get_x() + width / 2,
        b2[0].get_height() + 0.35,
        f"{int8_150['model_size_mb']:.2f} MB\n({size_ratio:.2f}x smaller)",
        ha="center",
        fontweight="bold",
        fontsize=8.8,
        color="#1b691b",
    )

    ax2.text(
        b1[1].get_x() + width / 2 - 0.04,
        b1[1].get_height() + 0.35,
        f"{fp32_150['peak_rss_mb']:.1f} MB\n(FP32 RSS)",
        ha="center",
        fontweight="bold",
        fontsize=8.5,
    )
    ax2.text(
        b2[1].get_x() + width / 2 + 0.04,
        b2[1].get_height() + 0.35,
        f"{int8_150['peak_rss_mb']:.1f} MB\n(Zero-PyTorch)",
        ha="center",
        fontweight="bold",
        fontsize=8.5,
        color="#1b691b",
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels(["Model Size on Disk (MB)", "Process RSS Memory (Scaled ÷10 MB)"], fontsize=10, fontweight="bold")
    ax2.set_ylabel("Megabytes (MB)", fontsize=10.5, fontweight="bold")
    ax2.set_ylim(0, 16)
    ax2.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
    ax2.legend(loc="upper left", fontsize=8.8, framealpha=0.92)
    ax2.set_title("(b) Storage Compression & Zero-PyTorch RAM Footprint", fontsize=11.5, fontweight="bold", pad=10)

    fig.suptitle(
        "Raspberry Pi 5 Edge Pipeline Efficiency: Stage Latency, Model Compression & RAM",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    bench_150 = load_json(REPORTS_DIR / "pi_benchmark.json")
    bench_1000 = load_json(REPORTS_DIR / "pi_benchmark_stress_1000runs.json")

    out1 = REPORTS_DIR / "pi_thread_scaling_thermal.png"
    out2 = REPORTS_DIR / "pi_burst_vs_endurance.png"
    out3 = REPORTS_DIR / "pi_stage_breakdown.png"

    plot_thread_scaling_and_thermal(bench_150, out1)
    plot_burst_vs_endurance(bench_150, bench_1000, out2)
    plot_stage_and_footprint(bench_150, out3)

    print(f"Generated: {out1}")
    print(f"Generated: {out2}")
    print(f"Generated: {out3}")


if __name__ == "__main__":
    main()
