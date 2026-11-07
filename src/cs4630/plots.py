#!/usr/bin/env python3
"""Plot CPU matrix-multiplication latency from the benchmark results."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "assets" / "benchmark_results" / "results.csv"
RTX3090_RESULTS = ROOT / "my_sgemm" / "benchmarks" / "sweep.csv"
BLOCK_TILING_RESULTS = (
    ROOT / "assets" / "benchmark_results" / "blocktiling_config_results.csv"
)
GENERATED_IMAGES = ROOT / "assets" / "images" / "generated"
CPU_OUTPUT = GENERATED_IMAGES / "matmul_timing.png"
COMPARISON_OUTPUT = GENERATED_IMAGES / "cpu_naive_latency.png"
NAIVE_VARIANT_LATENCY_OUTPUT = GENERATED_IMAGES / "naive_variant_latency.png"
TILED_COMPARISON_OUTPUT = GENERATED_IMAGES / "cpu_naive_tiled_latency.png"
TILED_THROUGHPUT_OUTPUT = GENERATED_IMAGES / "cpu_naive_tiled_throughput.png"
TILED_PEAK_THROUGHPUT_OUTPUT = (
    GENERATED_IMAGES / "cpu_naive_tiled_peak_throughput.png"
)
TILED_THROUGHPUT_SCALING_OUTPUT = (
    GENERATED_IMAGES / "tiled_throughput_scaling.png"
)
BLOCK_TILED_COMPARISON_OUTPUT = GENERATED_IMAGES / "basic_block_tiled_latency.png"
CUBLAS_BLOCK_TILED_COMPARISON_OUTPUT = (
    GENERATED_IMAGES / "cublas_block_tiled_latency.png"
)
BLOCK_TILED_THROUGHPUT_OUTPUT = (
    GENERATED_IMAGES / "block_tiled_throughput.png"
)
CUBLAS_BLOCK_TILED_THROUGHPUT_OUTPUT = (
    GENERATED_IMAGES / "cublas_block_tiled_throughput.png"
)
ALL_IMPLEMENTATIONS_LATENCY_OUTPUT = (
    GENERATED_IMAGES / "all_implementations_latency.png"
)
ROOFLINE_OUTPUT = GENERATED_IMAGES / "rtx3090_roofline.png"
BLOCK_TILING_ROOFLINE_OUTPUT = (
    GENERATED_IMAGES / "blocktiling_config_roofline.png"
)
CUBLAS_BLOCK_TILING_ROOFLINE_OUTPUT = (
    GENERATED_IMAGES / "cublas_blocktiling_config_roofline.png"
)
BLOCK_TILING_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "*"]
CUBLAS_COLOR = "#eab308"
CUBLAS_MARKER = "p"


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


def load_rtx3090_results(kernel):
    """Return matrix sizes and latencies for a kernel in the RTX 3090 results."""
    with RTX3090_RESULTS.open(newline="", encoding="utf-8") as results_file:
        rows = [
            row
            for row in csv.DictReader(results_file)
            if row["kernel"] == kernel
        ]

    rows.sort(key=lambda row: int(row["size"]))
    sizes = [int(row["size"]) for row in rows]
    latencies = [float(row["time_ms"]) / 1000 for row in rows]
    return sizes, latencies


def load_block_tiling_results():
    """Return block-tiling benchmark rows ordered by config and matrix size."""
    with BLOCK_TILING_RESULTS.open(newline="", encoding="utf-8") as results_file:
        rows = list(csv.DictReader(results_file))

    rows.sort(key=lambda row: (int(row["config"]), int(row["size"])))
    return rows


def plot_block_tiling_latency(
    tiled_sizes,
    tiled_latencies,
    block_tiling_configs,
    output,
    cublas_results=None,
):
    """Plot basic and block-tiled latency, optionally including cuBLAS."""
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        tiled_sizes,
        tiled_latencies,
        marker="h",
        color="black",
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1.5,
        linewidth=2.8,
        markersize=7,
        label="Basic tiled: 32×32 tile",
    )
    colors = plt.get_cmap("tab10").colors
    for (config, rows), marker, color in zip(
        sorted(block_tiling_configs.items()), BLOCK_TILING_MARKERS, colors
    ):
        first = rows[0]
        label = (
            f'C{config}: '
            f'{first["BM"]}×{first["BN"]}×{first["BK"]} / '
            f'{first["TM"]}×{first["TN"]}'
        )
        axis.plot(
            [int(row["size"]) for row in rows],
            [float(row["seconds"]) for row in rows],
            marker=marker,
            color=color,
            linewidth=1.8,
            markersize=4.5,
            label=label,
        )

    title = "Basic Tiled vs. Block-Tiled CUDA Latency"
    if cublas_results is not None:
        cublas_sizes, cublas_latencies = cublas_results
        axis.plot(
            cublas_sizes,
            cublas_latencies,
            marker=CUBLAS_MARKER,
            color=CUBLAS_COLOR,
            markeredgecolor="black",
            markeredgewidth=0.6,
            linewidth=2.8,
            markersize=6,
            zorder=5,
            label="cuBLAS SGEMM",
        )
        title = "Basic Tiled vs. Block-Tiled vs. cuBLAS Latency"

    style_latency_axis(axis, title)
    axis.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {output}")


def plot_block_tiling_throughput(
    tiled_sizes,
    tiled_latencies,
    block_tiling_configs,
    output,
    cublas_results=None,
):
    """Plot useful SGEMM throughput for the tiled implementations."""
    fig, axis = plt.subplots(figsize=(8, 5))
    tiled_throughput = [
        2 * size**3 / latency / 1e12
        for size, latency in zip(tiled_sizes, tiled_latencies)
    ]
    axis.plot(
        tiled_sizes,
        tiled_throughput,
        marker="h",
        color="black",
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1.5,
        linewidth=2.8,
        markersize=7,
        label="Basic tiled: 32×32 tile",
    )
    colors = plt.get_cmap("tab10").colors
    for (config, rows), marker, color in zip(
        sorted(block_tiling_configs.items()), BLOCK_TILING_MARKERS, colors
    ):
        first = rows[0]
        axis.plot(
            [int(row["size"]) for row in rows],
            [float(row["gflops"]) / 1000 for row in rows],
            marker=marker,
            color=color,
            linewidth=1.8,
            markersize=4.5,
            label=(
                f'C{config}: {first["BM"]}×{first["BN"]}×{first["BK"]} / '
                f'{first["TM"]}×{first["TN"]}'
            ),
        )

    title = "Basic Tiled vs. Block-Tiled CUDA Throughput"
    if cublas_results is not None:
        cublas_sizes, cublas_latencies = cublas_results
        axis.plot(
            cublas_sizes,
            [
                2 * size**3 / latency / 1e12
                for size, latency in zip(cublas_sizes, cublas_latencies)
            ],
            marker=CUBLAS_MARKER,
            color=CUBLAS_COLOR,
            markeredgecolor="black",
            markeredgewidth=0.6,
            linewidth=2.8,
            markersize=6,
            zorder=5,
            label="cuBLAS SGEMM",
        )
        title = "Basic Tiled vs. Block-Tiled vs. cuBLAS Throughput"

    axis.axhline(
        35.6,
        color="#6b7280",
        linestyle="--",
        linewidth=1.4,
        label="RTX 3090 FP32 peak",
    )
    axis.set_xlim(left=0)
    axis.set_ylim(0, 38)
    axis.set_xticks(range(0, 8193, 1024))
    axis.set_xlabel("Matrix dimension, n")
    axis.set_ylabel("Throughput (TFLOP/s)")
    axis.set_title(title)
    axis.grid(True, which="major", alpha=0.3)
    axis.legend(loc="upper left", fontsize=7.2, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {output}")


def plot_naive_roofline_point(axis, intensity, throughput):
    """Add the naive CUDA result with consistent styling."""
    axis.scatter(
        intensity,
        throughput,
        s=75,
        marker="d",
        facecolor="white",
        edgecolor="#d97706",
        linewidth=1.5,
        zorder=4,
        label="Naive CUDA kernel",
    )


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
    GENERATED_IMAGES.mkdir(parents=True, exist_ok=True)

    cpu_sizes, cpu_latencies = load_results("cpu")
    naive_sizes, naive_latencies = load_results("1")
    intro_naive_sizes, intro_naive_latencies = load_rtx3090_results("naive-32x32")
    tiled_sizes, tiled_latencies = load_results("3")
    cublas_sizes, cublas_latencies = load_results("0")
    block_tiling_rows = load_block_tiling_results()
    block_tiling_configs = {}
    for row in block_tiling_rows:
        block_tiling_configs.setdefault(int(row["config"]), []).append(row)

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
        intro_naive_sizes,
        intro_naive_latencies,
        "s-",
        color="#d97706",
        linewidth=2.5,
        markersize=5,
        label="Naive CUDA kernel (RTX 3090)",
    )
    style_latency_axis(axis, "CPU vs. Naive CUDA Latency")
    final_size = intro_naive_sizes[-1]
    final_latency = intro_naive_latencies[-1]
    axis.annotate(
        f"Final latency: {final_latency:.2f} s\nat n={final_size}",
        xy=(final_size, final_latency),
        xytext=(-105, 30),
        textcoords="offset points",
        fontsize=8,
        ha="center",
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#d97706",
        },
        arrowprops={
            "arrowstyle": "->",
            "color": "#d97706",
            "linewidth": 1.5,
        },
    )
    fig.tight_layout()
    fig.savefig(COMPARISON_OUTPUT, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {COMPARISON_OUTPUT}")

    naive_variants = {
        "8×8 block": load_rtx3090_results("naive-8x8"),
        "16×16 block": load_rtx3090_results("naive-16x16"),
        "32×32 block": load_rtx3090_results("naive-32x32"),
    }
    fig, axis = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^"]
    for (label, (sizes, latencies)), marker in zip(
        naive_variants.items(), markers
    ):
        axis.plot(
            sizes,
            latencies,
            marker=marker,
            linewidth=2,
            markersize=5,
            label=label,
        )
    style_latency_axis(axis, "Naive CUDA Latency by Block Shape")
    fig.tight_layout()
    fig.savefig(NAIVE_VARIANT_LATENCY_OUTPUT, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {NAIVE_VARIANT_LATENCY_OUTPUT}")

    naive_32_sizes, naive_32_latencies = naive_variants["32×32 block"]
    tiled_32_sizes, tiled_32_latencies = load_rtx3090_results("tiled-basic-32")
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        naive_32_sizes,
        naive_32_latencies,
        "s-",
        color="#d97706",
        linewidth=2.5,
        markersize=6,
        label="Naive 32×32 block",
    )
    axis.plot(
        tiled_32_sizes,
        tiled_32_latencies,
        "^-",
        color="#15803d",
        linewidth=2.5,
        markersize=6,
        label="Basic tiled 32×32 block",
    )
    style_latency_axis(axis, "Naive vs. Basic Tiled CUDA Latency")
    fig.tight_layout()
    fig.savefig(TILED_COMPARISON_OUTPUT, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {TILED_COMPARISON_OUTPUT}")

    fig, axis = plt.subplots(figsize=(8, 5))
    throughput_series = (
        (cpu_sizes, cpu_latencies, "o-", "#2563a6", "CPU (blocked OpenMP)"),
        (naive_sizes, naive_latencies, "s-", "#d97706", "Naive CUDA kernel"),
        (tiled_sizes, tiled_latencies, "^-", "#15803d", "Tiled CUDA kernel"),
    )
    for sizes, latencies, style, color, label in throughput_series:
        axis.plot(
            sizes,
            [
                2 * size**3 / latency / 1e12
                for size, latency in zip(sizes, latencies)
            ],
            style,
            color=color,
            linewidth=2.5,
            markersize=6,
            label=label,
        )
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    axis.set_xticks(range(0, 8193, 1024))
    axis.set_xlabel("Matrix dimension, n")
    axis.set_ylabel("Throughput (TFLOP/s)")
    axis.set_title("CPU vs. Naive vs. Tiled Matrix Multiplication Throughput")
    axis.grid(True, which="major", alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(TILED_THROUGHPUT_OUTPUT, dpi=200)
    print(f"Saved plot to {TILED_THROUGHPUT_OUTPUT}")

    for line in list(axis.lines)[:-1]:
        line.remove()
    axis.get_legend().remove()

    peak_tflops = 35.6
    final_tiled_throughput = 2 * tiled_sizes[-1] ** 3 / tiled_latencies[-1] / 1e12
    percent_of_peak = 100 * final_tiled_throughput / peak_tflops
    axis.axhline(
        peak_tflops,
        color="#b91c1c",
        linestyle="--",
        linewidth=2,
        label="RTX 3090 FP32 peak",
    )
    axis.set_ylim(0, 38)
    axis.set_title("Tiled Kernel Throughput vs. RTX 3090 Peak Throughput")
    axis.text(
        200,
        peak_tflops - 0.8,
        "RTX 3090 FP32 peak (35.6 TFLOP/s)",
        color="#b91c1c",
        fontsize=9,
        va="top",
    )
    axis.annotate(
        f"Tiled kernel: {final_tiled_throughput:.2f} TFLOP/s\n"
        f"{percent_of_peak:.1f}% of peak",
        xy=(tiled_sizes[-1], final_tiled_throughput),
        xytext=(5200, 12),
        fontsize=9,
        ha="center",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "#15803d",
        },
        arrowprops={
            "arrowstyle": "->",
            "color": "#15803d",
            "linewidth": 1.5,
        },
    )
    fig.tight_layout()
    fig.savefig(TILED_PEAK_THROUGHPUT_OUTPUT, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {TILED_PEAK_THROUGHPUT_OUTPUT}")

    tiled_throughput = [
        2 * size**3 / latency / 1e12
        for size, latency in zip(tiled_sizes, tiled_latencies)
    ]
    plateau_throughput = sum(
        throughput
        for size, throughput in zip(tiled_sizes, tiled_throughput)
        if size >= 2048
    ) / sum(size >= 2048 for size in tiled_sizes)
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.axvspan(0, 1024, color="#fef3c7", alpha=0.8)
    axis.axvspan(2048, 8192, color="#dcfce7", alpha=0.65)
    axis.plot(
        tiled_sizes,
        tiled_throughput,
        "^-",
        color="#15803d",
        linewidth=2.8,
        markersize=7,
    )
    axis.axhline(
        plateau_throughput,
        color="#166534",
        linestyle="--",
        linewidth=1.5,
    )
    axis.text(
        520,
        0.18,
        "Startup\nregion",
        color="#92400e",
        fontsize=11,
        ha="center",
        va="bottom",
    )
    axis.text(
        5100,
        plateau_throughput - 0.12,
        f"Steady-state plateau: {plateau_throughput:.2f} TFLOP/s",
        color="#166534",
        fontsize=11,
        ha="center",
        va="top",
    )
    axis.set_xlim(0, 8192)
    axis.set_ylim(0, 3.4)
    axis.set_xticks(range(0, 8193, 2048))
    axis.set_xlabel("Matrix dimension, n", fontsize=11)
    axis.set_ylabel("Throughput (TFLOP/s)", fontsize=11)
    axis.set_title("Tiled Kernel Throughput Approaches a Plateau", fontsize=14)
    axis.tick_params(labelsize=9)
    axis.grid(True, which="major", alpha=0.3)
    fig.tight_layout()
    fig.savefig(TILED_THROUGHPUT_SCALING_OUTPUT, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {TILED_THROUGHPUT_SCALING_OUTPUT}")

    plot_block_tiling_latency(
        tiled_sizes,
        tiled_latencies,
        block_tiling_configs,
        BLOCK_TILED_COMPARISON_OUTPUT,
    )
    plot_block_tiling_latency(
        tiled_sizes,
        tiled_latencies,
        block_tiling_configs,
        CUBLAS_BLOCK_TILED_COMPARISON_OUTPUT,
        (cublas_sizes, cublas_latencies),
    )
    plot_block_tiling_throughput(
        tiled_sizes,
        tiled_latencies,
        block_tiling_configs,
        BLOCK_TILED_THROUGHPUT_OUTPUT,
    )
    plot_block_tiling_throughput(
        tiled_sizes,
        tiled_latencies,
        block_tiling_configs,
        CUBLAS_BLOCK_TILED_THROUGHPUT_OUTPUT,
        (cublas_sizes, cublas_latencies),
    )

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        cpu_sizes,
        cpu_latencies,
        "o-",
        color="#2563a6",
        linewidth=2.8,
        markersize=5,
        label="CPU (blocked OpenMP)",
    )
    axis.plot(
        naive_sizes,
        naive_latencies,
        marker="d",
        color="#d97706",
        markerfacecolor="white",
        markeredgewidth=1.2,
        linewidth=2.5,
        markersize=5,
        label="Naive CUDA kernel",
    )
    axis.plot(
        tiled_sizes,
        tiled_latencies,
        marker="h",
        color="black",
        markerfacecolor="white",
        markeredgewidth=1.2,
        linewidth=2.5,
        markersize=6,
        label="Basic tiled: 32×32 tile",
    )
    best_block_rows = block_tiling_configs[4]
    axis.plot(
        [int(row["size"]) for row in best_block_rows],
        [float(row["seconds"]) for row in best_block_rows],
        marker=BLOCK_TILING_MARKERS[4],
        color=plt.get_cmap("tab10").colors[4],
        linewidth=2.5,
        markersize=5,
        label="Block-tiled: C4",
    )
    axis.plot(
        cublas_sizes,
        cublas_latencies,
        marker=CUBLAS_MARKER,
        color=CUBLAS_COLOR,
        markeredgecolor="black",
        markeredgewidth=0.6,
        linewidth=2.5,
        markersize=6,
        label="cuBLAS SGEMM",
    )
    style_latency_axis(axis, "Matrix Multiplication Latency: All Implementations")
    axis.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(ALL_IMPLEMENTATIONS_LATENCY_OUTPUT, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {ALL_IMPLEMENTATIONS_LATENCY_OUTPUT}")

    peak_tflops = 35.6
    bandwidth_tb_s = 0.936
    ridge_point = peak_tflops / bandwidth_tb_s
    roofline_intensities = [
        2 ** (step / 32)
        for step in range(int(32 * -2), int(32 * 5.6) + 1)
    ]
    roofline = [
        min(peak_tflops, bandwidth_tb_s * intensity)
        for intensity in roofline_intensities
    ]
    tiled_tflops = [
        2 * size**3 / latency / 1e12
        for size, latency in zip(tiled_sizes, tiled_latencies)
    ]
    naive_tflops = [
        2 * size**3 / latency / 1e12
        for size, latency in zip(naive_sizes, naive_latencies)
    ]
    cublas_tflops = [
        2 * size**3 / latency / 1e12
        for size, latency in zip(cublas_sizes, cublas_latencies)
    ]
    tiled_intensity = 8.0
    naive_size = naive_sizes[-1]
    naive_intensity = (2 * naive_size**3) / (
        4 * (2 * naive_size**3 + naive_size**2)
    )
    # Nsight identifies a 256×128 output tile for this cuBLAS SGEMM kernel.
    cublas_size = cublas_sizes[-1]
    cublas_bm, cublas_bn = 256, 128
    cublas_intensity = (2 * cublas_bm * cublas_bn * cublas_size) / (
        4
        * (
            cublas_size * (cublas_bm + cublas_bn)
            + 2 * cublas_bm * cublas_bn
        )
    )

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        roofline_intensities,
        roofline,
        color="#2563a6",
        linewidth=2.5,
        label="RTX 3090 roofline",
    )
    axis.axvline(
        ridge_point,
        color="#6b7280",
        linestyle="--",
        linewidth=1.5,
        label=f"Ridge point ({ridge_point:.1f})",
    )
    axis.scatter(
        tiled_intensity,
        tiled_tflops[-1],
        s=85,
        marker="h",
        facecolor="white",
        edgecolor="black",
        linewidth=1.5,
        zorder=4,
        label="Basic tiled: 32×32 tile",
    )
    plot_naive_roofline_point(axis, naive_intensity, naive_tflops[-1])
    axis.set_xlim(-1, 44)
    axis.set_xticks(range(0, 45, 5))
    axis.set_ylim(0, 38)
    axis.set_xlabel("Arithmetic intensity (FLOP/byte)")
    axis.set_ylabel("Throughput (TFLOP/s)")
    axis.set_title("Naive and Basic Tiled SGEMM")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7.5)
    fig.tight_layout()
    fig.savefig(ROOFLINE_OUTPUT, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {ROOFLINE_OUTPUT}")

    configurations = [rows[-1] for _, rows in sorted(block_tiling_configs.items())]

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        roofline_intensities,
        roofline,
        color="#2563a6",
        linewidth=2.5,
        label="RTX 3090 roofline",
    )
    axis.axvline(
        ridge_point,
        color="#6b7280",
        linestyle="--",
        linewidth=1.5,
        label=f"Ridge point ({ridge_point:.1f})",
    )
    axis.scatter(
        tiled_intensity,
        tiled_tflops[-1],
        s=85,
        marker="h",
        facecolor="white",
        edgecolor="black",
        linewidth=1.5,
        zorder=4,
        label="Basic tiled: 32×32 tile",
    )
    plot_naive_roofline_point(axis, naive_intensity, naive_tflops[-1])

    colors = plt.get_cmap("tab10").colors
    for row, marker, color in zip(configurations, BLOCK_TILING_MARKERS, colors):
        intensity = float(row["arithmetic_intensity_flop_per_byte"])
        throughput = float(row["gflops"]) / 1000
        config_label = (
            f'C{row["config"]}: '
            f'{row["BM"]}×{row["BN"]}×{row["BK"]} / '
            f'{row["TM"]}×{row["TN"]}'
        )
        axis.scatter(
            intensity,
            throughput,
            s=65,
            marker=marker,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
            label=config_label,
        )

    axis.set_xlim(-1, 44)
    axis.set_xticks(range(0, 45, 5))
    axis.set_ylim(0, 38)
    axis.set_xlabel("Arithmetic intensity (FLOP/byte)")
    axis.set_ylabel("Throughput (TFLOP/s)")
    axis.set_title("Naive, Tiled, and Block-Tiled SGEMM")
    axis.grid(True, alpha=0.25)
    axis.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=7.5,
        title="Block config: BM×BN×BK / TM×TN",
        title_fontsize=7.5,
    )
    fig.tight_layout()
    fig.savefig(BLOCK_TILING_ROOFLINE_OUTPUT, dpi=200)
    print(f"Saved plot to {BLOCK_TILING_ROOFLINE_OUTPUT}")

    axis.scatter(
        cublas_intensity,
        cublas_tflops[-1],
        s=90,
        marker=CUBLAS_MARKER,
        color=CUBLAS_COLOR,
        edgecolor="black",
        linewidth=0.7,
        zorder=5,
        label="cuBLAS SGEMM",
    )
    axis.set_title("Naive, Tiled, Block-Tiled, and cuBLAS SGEMM")
    axis.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=7.5,
        title="Block config: BM×BN×BK / TM×TN",
        title_fontsize=7.5,
    )
    fig.tight_layout()
    fig.savefig(CUBLAS_BLOCK_TILING_ROOFLINE_OUTPUT, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {CUBLAS_BLOCK_TILING_ROOFLINE_OUTPUT}")


if __name__ == "__main__":
    main()
