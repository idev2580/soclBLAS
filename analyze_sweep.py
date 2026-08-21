#!/usr/bin/env python3

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path


MEASUREMENT_PATTERN = re.compile(
    r"^\((?P<configuration>[0-9 ]+)\)"
    r"\[(?P<run_index>[0-9]+)\]:"
    r"\((?P<tflops>[0-9]+(?:\.[0-9]+)?)\)$"
)
EXPECTED_RUN_INDICES = set(range(10))


def parse_measurements(log_path: Path) -> dict[str, dict[int, float]]:
    measurements: dict[str, dict[int, float]] = defaultdict(dict)

    with log_path.open(encoding="utf-8") as log_file:
        for line_number, line in enumerate(log_file, start=1):
            match = MEASUREMENT_PATTERN.fullmatch(line.strip())
            if match is None:
                continue

            configuration = f"({match.group('configuration')})"
            run_index = int(match.group("run_index"))
            tflops = float(match.group("tflops"))

            if run_index not in EXPECTED_RUN_INDICES:
                continue
            if run_index in measurements[configuration]:
                raise ValueError(
                    f"line {line_number}: duplicate run index {run_index} "
                    f"for configuration {configuration}"
                )

            measurements[configuration][run_index] = tflops

    return measurements


def calculate_averages(
    measurements: dict[str, dict[int, float]],
) -> tuple[list[tuple[str, float]], list[str]]:
    averages: list[tuple[str, float]] = []
    incomplete_configurations: list[str] = []

    for configuration, runs in measurements.items():
        if set(runs) != EXPECTED_RUN_INDICES:
            incomplete_configurations.append(configuration)
            continue

        average_tflops = sum(runs.values()) / len(EXPECTED_RUN_INDICES)
        averages.append((configuration, average_tflops))

    averages.sort(key=lambda result: result[1], reverse=True)
    return averages, incomplete_configurations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print configurations with the highest average TFLOPs."
    )
    parser.add_argument(
        "log_path",
        nargs="?",
        type=Path,
        default=Path("RTX4060LaptopNaiveSweep.txt"),
        help="sweep log path (default: RTX4060LaptopNaiveSweep.txt)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="number of configurations to print (default: 10)",
    )
    args = parser.parse_args()

    if args.top <= 0:
        parser.error("--top must be greater than zero")

    try:
        measurements = parse_measurements(args.log_path)
        averages, incomplete_configurations = calculate_averages(measurements)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if incomplete_configurations:
        print(
            f"warning: skipped {len(incomplete_configurations)} configuration(s) "
            "without all 10 runs",
            file=sys.stderr,
        )

    if not averages:
        print("error: no configuration with all 10 runs was found", file=sys.stderr)
        return 1

    print("Rank  Average TFLOPs  Configuration")
    for rank, (configuration, average_tflops) in enumerate(
        averages[: args.top], start=1
    ):
        print(f"{rank:>4}  {average_tflops:>14.6f}  {configuration}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
