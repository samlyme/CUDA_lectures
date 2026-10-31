#!/usr/bin/env python3
"""Plot CPU matrix-multiplication latency from the benchmark results."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "assets" / "benchmark_results" / "results.csv"
CPU_OUTPUT = ROOT / "assets" / "images" / "matmul_timing.png"
COMPARISON_OUTPUT = ROOT / "assets" / "images" / "cpu_naive_latency.png"


def load_results(benchmark):
    """Return matrix sizes and latencies for one benchmark."""
    with RESULTS.open(newline="", encoding="utf-8") as results_file:
        rows = [
            row
            for row in csv.DictReader(results_file)
            if row["benchmark"] == benchmark
        ]

    rows.sort(key=lambda row: int(row["size"]))
    sizes = [int(row["size"]) for row in rows]
    latencies = [float(row["seconds"]) for row in rows]
    return sizes, latencies


def style_latency_axis(axis, title):
    """Apply shared formatting to a latency plot."""
    axis.set_xscale("log", base=2)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Matrix dimension, n")
    axis.set_ylabel("Latency (seconds)")
    axis.set_title(title)
    axis.set_xticks([128, 256, 512, 1024, 2048, 4096, 8192])
    axis.xaxis.set_major_formatter(ScalarFormatter())
    axis.grid(True, which="major", alpha=0.3)
    axis.legend()


def main():
    cpu_sizes, cpu_latencies = load_results("cpu")
    naive_sizes, naive_latencies = load_results("1")

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        cpu_sizes,
        cpu_latencies,
        "o-",
        color="#2563a6",
        linewidth=2.5,
        markersize=6,
        label="CPU (blocked OpenMP)",
    )
    style_latency_axis(axis, "CPU Matrix Multiplication Latency")
    fig.tight_layout()
    fig.savefig(CPU_OUTPUT, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {CPU_OUTPUT}")

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        cpu_sizes,
        cpu_latencies,
        "o-",
        color="#2563a6",
        linewidth=2.5,
        markersize=6,
        label="CPU (blocked OpenMP)",
    )
    axis.plot(
        naive_sizes,
        naive_latencies,
        "s-",
        color="#d97706",
        linewidth=2.5,
        markersize=5,
        label="Naive CUDA kernel",
    )
    style_latency_axis(axis, "CPU vs. Naive CUDA Latency")
    fig.tight_layout()
    fig.savefig(COMPARISON_OUTPUT, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {COMPARISON_OUTPUT}")


if __name__ == "__main__":
    main()
