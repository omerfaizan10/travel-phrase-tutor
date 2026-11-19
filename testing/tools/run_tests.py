"""
Run the test sets and build one report.

    .venv/bin/python testing/tools/run_tests.py                      # quick: unit + slm_contract
    .venv/bin/python testing/tools/run_tests.py --sets all           # + ui + slm_quality (live Ollama)
    .venv/bin/python testing/tools/run_tests.py --sets slm_quality --repeats 5 --model llama3
    .venv/bin/python testing/tools/run_tests.py --sets all --language macedonian   # one reviewer's language

Sets:
  unit          pure logic                          pytest -m unit
  slm_contract  code around the SLM, fake SLM       pytest -m slm_contract
  ui            Streamlit AppTest, fake SLM         pytest -m ui
  slm_quality   real SLM conversations              testing/tools/collect_transcripts.py
                (then a judge agent scores them: see testing/skills/test-app/SKILL.md)

Output: testing/reports/<timestamp>/  with results.json, summary.md and report.html.
Exit code 0 when every set that ran passed (slm_quality counts only its code checks
until judged).
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # the repository root
TESTING = ROOT / "testing"
PY = sys.executable
PYTEST_SETS = ("unit", "slm_contract", "ui")
ALL_SETS = PYTEST_SETS + ("slm_quality",)
QUICK = ("unit", "slm_contract")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", default="quick", help="quick, all, or a comma list of " + ", ".join(ALL_SETS))
    ap.add_argument("--repeats", type=int, help="slm_quality runs per scenario")
    ap.add_argument("--model", help="Ollama model for slm_quality")
    ap.add_argument("--only", help="slm_quality: scenario ids or kinds")
    ap.add_argument("--language", choices=["spanish", "macedonian"],
                    help="slm_quality: one language only (code tests always run for both)")
    ap.add_argument("--out", help="run folder (default testing/reports/<timestamp>)")
    args = ap.parse_args()

    sets = {"quick": QUICK, "all": ALL_SETS}.get(args.sets) or tuple(
        s.strip() for s in args.sets.split(","))
    unknown = set(sets) - set(ALL_SETS)
    if unknown:
        ap.error(f"unknown set(s): {', '.join(sorted(unknown))}")

    out = Path(args.out) if args.out else TESTING / "reports" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "sets.txt").write_text("\n".join(sets))
    print(f"Run folder: {out}")
    # The test catalog (What / Why / If it fails of every test) for the report.
    subprocess.run([PY, "-m", "pytest", str(TESTING / "tests"), "--collect-only", "-q", "-p", "no:cacheprovider",
                    f"--catalog={out / 'catalog.json'}"], cwd=ROOT, capture_output=True)

    for name in sets:
        start = time.time()
        if name in PYTEST_SETS:
            cmd = [PY, "-m", "pytest", str(TESTING / "tests"), "-m", name, "-q", "-p", "no:cacheprovider",
                   f"--junitxml={out / f'pytest-{name}.xml'}"]
        else:
            cmd = [PY, str(TESTING / "tools" / "collect_transcripts.py"), "--out", str(out)]
            for flag in ("repeats", "model", "only", "language"):
                if getattr(args, flag):
                    cmd += [f"--{flag}", str(getattr(args, flag))]
        print(f"\n=== {name} ===", flush=True)
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        log = proc.stdout + proc.stderr
        (out / f"{name}.log").write_text(log)
        tail = [line for line in log.strip().splitlines() if "Warning" not in line][-4:]
        print("\n".join(tail))
        print(f"({time.time() - start:.0f}s)")

    proc = subprocess.run([PY, str(TESTING / "tools" / "build_report.py"), str(out)], cwd=ROOT)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
