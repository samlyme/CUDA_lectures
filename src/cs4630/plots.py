#!/usr/bin/env python3
"""Benchmark the naive matrix multiplication and plot its cubic scaling."""

from pathlib import Path
import subprocess
import tempfile

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]


def run_benchmark():
    """Compile matmul.cpp, run it, and return n and average time in microseconds."""
    with tempfile.TemporaryDirectory() as directory:
        executable = Path(directory) / "matmul"
        subprocess.run(
            [
                "c++",
                "-std=c++17",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Wpedantic",
                str(ROOT / "benchmarks" / "matmul.cpp"),
                "-o",
                str(executable),
            ],
            check=True,
        )
        result = subprocess.run(
            [str(executable)], capture_output=True, text=True, check=True
        )

    sizes = []
    average_times = []
    for line in result.stdout.splitlines():
        columns = line.split()
        if columns and columns[0].isdigit():
            sizes.append(int(columns[0]))
            # Columns are: n, repetitions, total (ms), average (us), GFLOP/s.
            average_times.append(float(columns[3]))

    return sizes, average_times


def main():
    sizes, average_times = run_benchmark()

    # Least-squares fit of t = c*n^3. This reference curve demonstrates the
    # expected cubic running time of the three nested loops.
    cubes = [n**3 for n in sizes]
    coefficient = sum(x * y for x, y in zip(cubes, average_times)) / sum(
        x * x for x in cubes
    )
    cubic_fit = [coefficient * n**3 for n in sizes]

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(sizes, average_times, "o-", linewidth=2, label="Measured average")
    axis.plot(sizes, cubic_fit, "--", linewidth=2, label=r"Cubic fit: $c n^3$")
    axis.set_xlabel("Matrix size, n")
    axis.set_ylabel("Average multiplication time (µs)")
    axis.set_title("Naive Matrix Multiplication Is $O(n^3)$")
    axis.set_xlim(0, 264)
    axis.set_xticks(range(0, 257, 32))
    axis.set_xticks(range(0, 257, 8), minor=True)
    axis.grid(True, which="major", alpha=0.3)
    axis.grid(True, which="minor", axis="x", alpha=0.1)
    axis.legend()
    fig.tight_layout()

    output = ROOT / "assets" / "images" / "matmul_timing.png"
    fig.savefig(output, dpi=200)
    print(f"Saved plot to {output}")
    plt.show()


if __name__ == "__main__":
    main()
