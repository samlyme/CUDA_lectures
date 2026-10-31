#!/usr/bin/env python3
"""Plot CPU matrix-multiplication latency from the benchmark results."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "assets" / "benchmark_results" / "results.csv"
BLOCK_TILED_CONFIG_RESULTS = (
    ROOT / "assets" / "benchmark_results" / "blocktiling_config_results.csv"
)
CPU_OUTPUT = ROOT / "assets" / "images" / "matmul_timing.png"
COMPARISON_OUTPUT = ROOT / "assets" / "images" / "cpu_naive_latency.png"
TILED_COMPARISON_OUTPUT = (
    ROOT / "assets" / "images" / "cpu_naive_tiled_latency.png"
)
BLOCK_TILED_COMPARISON_OUTPUT = (
    ROOT / "assets" / "images" / "basic_block_tiled_latency.png"
)
ROOFLINE_OUTPUT = ROOT / "assets" / "images" / "rtx3090_roofline.png"


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


def load_block_tiled_configs():
    """Return block-tiled configuration measurements in config order."""
    with BLOCK_TILED_CONFIG_RESULTS.open(
        newline="", encoding="utf-8"
    ) as results_file:
        rows = list(csv.DictReader(results_file))

    return sorted(rows, key=lambda row: int(row["config"]))


def style_latency_axis(axis, title):
    """Apply shared formatting to a latency plot."""
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Matrix dimension, n")
    axis.set_ylabel("Latency (seconds)")
    axis.set_title(title)
    axis.set_xticks(range(0, 8193, 1024))
    axis.grid(True, which="major", alpha=0.3)
    axis.legend()


def main():
    cpu_sizes, cpu_latencies = load_results("cpu")
    naive_sizes, naive_latencies = load_results("1")
    tiled_sizes, tiled_latencies = load_results("3")
    block_tiled_configs = load_block_tiled_configs()

    # Least-squares fit constrained to t = c*n^3, the expected complexity of
    # the three nested matrix-multiplication loops.
    cubes = [size**3 for size in cpu_sizes]
    cubic_coefficient = sum(
        cube * latency for cube, latency in zip(cubes, cpu_latencies)
    ) / sum(cube**2 for cube in cubes)
    fit_sizes = list(range(0, max(cpu_sizes) + 1, 32))
    cubic_fit = [cubic_coefficient * size**3 for size in fit_sizes]

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
        fit_sizes,
        cubic_fit,
        "--",
        color="#b91c1c",
        linewidth=2.2,
        label=r"Cubic fit: $c n^3$",
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
    axis.plot(
        tiled_sizes,
        tiled_latencies,
        "^-",
        color="#15803d",
        linewidth=2.5,
        markersize=6,
        label="Tiled CUDA kernel",
    )
    style_latency_axis(axis, "CPU vs. Naive vs. Tiled CUDA Latency")
    fig.tight_layout()
    fig.savefig(TILED_COMPARISON_OUTPUT, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {TILED_COMPARISON_OUTPUT}")

    # The configuration sweep was measured at n=8192, so compare all kernels
    # at that common size instead of implying full size sweeps for each config.
    config_labels = ["Basic tiled"]
    config_latencies_ms = [tiled_latencies[-1] * 1000]
    for config in block_tiled_configs:
        config_labels.append(
            f"Config {config['config']}: "
            f"{config['BM']}×{config['BN']}×{config['BK']} block, "
            f"{config['TM']}×{config['TN']}/thread"
        )
        config_latencies_ms.append(float(config["seconds"]) * 1000)

    fig, axis = plt.subplots(figsize=(8, 5))
    bars = axis.barh(
        config_labels,
        config_latencies_ms,
        color=["#15803d"] + ["#7c3aed"] * len(block_tiled_configs),
    )
    axis.invert_yaxis()
    axis.bar_label(bars, fmt="%.1f ms", padding=4, fontsize=8)
    axis.set_xlim(0, max(config_latencies_ms) * 1.15)
    axis.set_xlabel("Latency at n = 8192 (milliseconds)")
    axis.grid(True, axis="x", alpha=0.3)
    axis.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(BLOCK_TILED_COMPARISON_OUTPUT, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {BLOCK_TILED_COMPARISON_OUTPUT}")

    peak_tflops = 35.6
    bandwidth_tb_s = 0.936
    ridge_point = peak_tflops / bandwidth_tb_s
    arithmetic_intensities = [2 ** (step / 16) for step in range(-32, 129)]
    roofline = [
        min(peak_tflops, bandwidth_tb_s * intensity)
        for intensity in arithmetic_intensities
    ]
    tiled_tflops = [
        2 * size**3 / latency / 1e12
        for size, latency in zip(tiled_sizes, tiled_latencies)
    ]
    tiled_intensity = 8.0

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.loglog(
        arithmetic_intensities,
        roofline,
        color="#2563a6",
        linewidth=3,
        label="RTX 3090 roofline",
    )
    axis.axvline(
        ridge_point,
        color="#6b7280",
        linestyle="--",
        linewidth=1.5,
        label=f"Ridge point ({ridge_point:.1f} FLOP/byte)",
    )
    axis.scatter(
        [tiled_intensity],
        [tiled_tflops[-1]],
        s=70,
        marker="o",
        color="#d97706",
        edgecolor="black",
        linewidth=0.7,
        zorder=3,
        label="Basic tiled run",
    )
    axis.set_xlim(0.25, 256)
    axis.set_ylim(0.1, 64)
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.set_xlabel("Arithmetic intensity (FLOP/byte)")
    axis.set_ylabel("Throughput (TFLOP/s)")
    axis.set_title("RTX 3090 FP32 Roofline")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ROOFLINE_OUTPUT, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {ROOFLINE_OUTPUT}")


if __name__ == "__main__":
    main()
