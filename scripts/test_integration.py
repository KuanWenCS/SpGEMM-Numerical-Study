import csv
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import struct
import sys
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def typed(value, precision):
    return struct.unpack("f", struct.pack("f", value))[0] if precision == "fp32" else value


def rational(significand, exponent):
    return Fraction(int(significand)) * Fraction(2) ** int(exponent)


def close_decimal(actual, expected):
    with localcontext() as context:
        context.prec = 100
        wanted = Decimal(expected.numerator) / Decimal(expected.denominator)
        observed = Decimal(actual)
        assert abs(observed - wanted) <= abs(wanted) * Decimal("1e-23") + Decimal("1e-90"), (actual, wanted)


def main():
    cuda = sys.argv[1:] == ["--cuda"]
    if sys.argv[1:] and not cuda:
        raise ValueError("Supported option: --cuda")
    binary = ROOT / "build" / ("study_cuda" if cuda else "study")
    (ROOT / "build").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "build", prefix="integration-") as temporary:
        root = Path(temporary)
        rng = random.Random(731)
        choices = [0.0, 1.0, -1.0, 33554432.0, 2.0 ** -30, 1.1, -0.3]
        a = [[rng.choice(choices) for _ in range(5)] for _ in range(5)]
        p = [[rng.choice(choices) for _ in range(4)] for _ in range(5)]
        for name, matrix in (("A", a), ("P", p)):
            entries = [(i + 1, j + 1, value) for i, row in enumerate(matrix) for j, value in enumerate(row)]
            rng.shuffle(entries)
            text = f"%%MatrixMarket matrix coordinate real general\n{len(matrix)} {len(matrix[0])} {len(entries)}\n"
            text += "".join(f"{i} {j} {value:.17g}\n" for i, j, value in entries)
            (root / f"{name}.mtx").write_text(text)
        out = root / "study"
        command = [str(binary), "--a", str(root / "A.mtx"), "--p", str(root / "P.mtx"),
                   "--out", str(out), "--samples", "all", "--repeat", "3", "--threads", "4", "--trace", "0,0"]
        if cuda:
            command += ["--backend", "both"]
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert (out / "status.txt").read_text().strip() == "complete"
        references = {}
        for precision in ("fp32", "fp64"):
            for entry in read_csv(out / f"reference_{precision}.csv"):
                i, j = int(entry["row"]), int(entry["column"])
                products = [Fraction.from_float(typed(a[i][k], precision)) *
                            Fraction.from_float(typed(p[k][j], precision)) for k in range(5)]
                expected = sum(products, Fraction())
                actual = rational(entry["reference_significand"], entry["reference_exponent"])
                assert actual == expected, (precision, i, j, actual, expected)
                rounded = sum((Fraction.from_float(typed(typed(a[i][k], precision) * typed(p[k][j], precision), precision))
                               for k in range(5)), Fraction())
                assert rational(entry["rounded_sum_significand"], entry["rounded_sum_exponent"]) == rounded
                assert int(entry["products"]) == 5
                assert int(entry["positive_products"]) == sum(v > 0 for v in products)
                assert int(entry["negative_products"]) == sum(v < 0 for v in products)
                assert int(entry["zero_products"]) == sum(v == 0 for v in products)
                assert int(entry["a_row_entries"]) == 5 and int(entry["p_column_entries"]) == 5
                references[precision, i, j] = (expected, rounded)
        for entry in read_csv(out / "errors.csv"):
            precision = entry["precision"]
            i, j = int(entry["row"]), int(entry["column"])
            size, fmt = (4, "!f") if precision == "fp32" else (8, "!d")
            value = struct.unpack(fmt, int(entry["value_bits"]).to_bytes(size, "big"))[0]
            reference, rounded = references[precision, i, j]
            observed = Fraction.from_float(value)
            close_decimal(entry["signed_error"], observed - reference)
            close_decimal(entry["abs_error"], abs(observed - reference))
            close_decimal(entry["product_error"], rounded - reference)
            close_decimal(entry["accumulation_error"], observed - rounded)
            if reference:
                close_decimal(entry["relative_error"], abs((observed - reference) / reference))
            else:
                assert entry["relative_error"] == "NA"
        summaries = read_csv(out / "summary.csv")
        assert len(summaries) == (28 if cuda else 14)
        for entry in summaries:
            assert int(entry["sampled_entries"]) == 20
            if entry["strategy"] == "serial":
                assert int(entry["bitwise_diff_cpu_serial"]) == 0
        assert all(int(row["replay_verified"]) == 1 for row in read_csv(out / "trace_summary.csv"))
        assert all(int(row["entries_changed_from_first"]) == 0 for row in read_csv(out / "variability.csv") if row["strategy"] == "serial")
        duplicate = subprocess.run(command, capture_output=True, text=True)
        assert duplicate.returncode != 0 and "already exists" in duplicate.stderr
        wrapper_out = root / "wrapped"
        wrapped = subprocess.run(["python3", str(ROOT / "scripts/run_experiment.py"), "--binary", str(binary), "--a", str(root / "A.mtx"),
            "--p", str(root / "P.mtx"), "--out", str(wrapper_out), "--samples", "2", "--repeat", "1", "--strategy", "serial"],
            cwd=ROOT, capture_output=True, text=True)
        assert wrapped.returncode == 0, (wrapped.stdout, wrapped.stderr)
        manifest = json.loads((wrapper_out / "manifest.json").read_text())
        assert manifest["status"] == "complete"
        assert manifest["inputs"]["A"]["sha256"] == hashlib.sha256((root / "A.mtx").read_bytes()).hexdigest()
        assert all(int(row["sampled_entries"]) == 2 for row in read_csv(wrapper_out / "artifacts/summary.csv"))
        failed_out = root / "bad"
        failure = subprocess.run([str(binary), "--a", str(root / "A.mtx"), "--p", str(root / "P.mtx"),
            "--out", str(failed_out), "--trace", "999,0", "--repeat", "1"], capture_output=True, text=True)
        assert failure.returncode != 0 and (failed_out / "status.txt").read_text().startswith("failed:")
        none_out = root / "none"
        none = subprocess.run([str(binary), "--a", str(root / "A.mtx"), "--p", str(root / "P.mtx"),
            "--out", str(none_out), "--samples", "0", "--repeat", "1"], capture_output=True, text=True)
        assert none.returncode == 0, none.stderr
        assert all(int(row["sampled_entries"]) == 0 for row in read_csv(none_out / "summary.csv"))
    print(f"{'CUDA' if cuda else 'CPU'} integration: PASS (Fraction oracle, rounding decomposition, trace, sampling, repeats, manifest, failures)")


if __name__ == "__main__":
    main()
