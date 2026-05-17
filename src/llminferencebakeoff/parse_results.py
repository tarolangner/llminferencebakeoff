"""Parse UI-copied experiment results and output CSV for plotting.

How to use: edit INPUT and OUTPUT below, then run:
    uv run python src/llminferencebakeoff/parse_results.py
"""

import csv
import re
from pathlib import Path

# ── Edit these ────────────────────────────────────────────────
INPUT = "experiment2_results_l4.txt"  # path to text file with pasted results
OUTPUT = "experiment2_parsed_l4.csv"  # path for the output CSV
# ─────────────────────────────────────────────────────────────


def _parse_ttft_line(line: str) -> dict:
    results = {}
    for match in re.finditer(r"(\w+(?:\s+\w+)*):\s*([\d.]+)ms", line):
        results[match.group(1).strip()] = {"time_to_first_token_ms": float(match.group(2))}
    return results


def _parse_throughput_line(line: str) -> dict:
    results = {}
    for match in re.finditer(r"(\w+(?:\s+\w+)*):\s*([\d.]+)\s*tok/s", line):
        results[match.group(1).strip()] = {"decode_throughput": float(match.group(2))}
    return results


def _parse_response_line(line: str) -> dict | None:
    m = re.match(r"([\w\s]+):\s*Tokens:\s*(\d+)", line)
    if not m:
        return None
    result = {"backend": m.group(1).strip(), "token_count": int(m.group(2))}
    ttft = re.search(r"TTFT:\s*([\d.]+)ms", line)
    if ttft:
        result["time_to_first_token_ms"] = float(ttft.group(1))
    speed = re.search(r"Decode Speed:\s*([\d.]+)\s*tok/s", line)
    if speed:
        result["decode_throughput"] = float(speed.group(1))
    total = re.search(r"Total Time:\s*([\d.]+)s", line)
    if total:
        result["total_time_s"] = float(total.group(1))
    return result


def parse_file(path: str) -> list[dict]:
    text = Path(path).read_text()
    rows = []

    for block in re.split(r"\n\s*\n", text):
        lines = [line.strip() for line in block.strip().split("\n") if line.strip()]
        if not lines:
            continue

        row = _parse_response_line(lines[0])
        if row:
            rows.append(row)
            continue

        ttft = None
        throughput = None
        for line in lines:
            if line.startswith("TTFT"):
                ttft = _parse_ttft_line(line)
            elif line.startswith("Throughput"):
                throughput = _parse_throughput_line(line)

        if ttft or throughput:
            all_backends = set()
            if ttft:
                all_backends |= set(ttft.keys())
            if throughput:
                all_backends |= set(throughput.keys())

            for backend in all_backends:
                row = {"backend": backend}
                if ttft and backend in ttft:
                    row.update(ttft[backend])
                if throughput and backend in throughput:
                    row.update(throughput[backend])
                rows.append(row)

    return rows


def main():
    rows = parse_file(INPUT)
    if not rows:
        print("No results found in input file.")
        return

    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
