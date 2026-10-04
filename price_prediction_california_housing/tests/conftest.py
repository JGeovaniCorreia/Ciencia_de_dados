"""Pytest configuration for the California Housing project.

The XGBoost pipeline was serialized from a notebook where the transformer
classes lived in __main__. Under pytest, __main__ is pytest.__main__, so
joblib.load fails to reconstruct them. This conftest patches __main__ with
the classes from src.transformers before any test runs.

At the end of each session, writes RESULTADOS_TESTS_UNITARIOS.txt inside
this directory, overwriting any previous run.
"""

import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.transformers import WinsorizacaoTransformer, CaliforniaHousingTransformer  # noqa: E402

sys.modules["__main__"].WinsorizacaoTransformer = WinsorizacaoTransformer
sys.modules["__main__"].CaliforniaHousingTransformer = CaliforniaHousingTransformer

_results: list[dict] = []


def pytest_runtest_logreport(report):
    if report.when == "call":
        _results.append({
            "name": report.nodeid.split("::")[-1],
            "status": "PASSOU" if report.passed else "FALHOU",
            "duration": report.duration,
        })


def pytest_sessionfinish(session, exitstatus):
    passed = sum(1 for r in _results if r["status"] == "PASSOU")
    failed = sum(1 for r in _results if r["status"] == "FALHOU")
    total = len(_results)

    lines = [
        "=" * 62,
        "  RESULTADOS — TESTES UNITÁRIOS",
        "  California Housing · src/predict.py",
        f"  Executado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 62,
        "",
        f"  Total: {total}   Passaram: {passed}   Falharam: {failed}",
        "",
        "-" * 62,
    ]

    for r in _results:
        marca = "OK  " if r["status"] == "PASSOU" else "FAIL"
        lines.append(f"  [{marca}]  {r['name']:<45}  {r['duration']:.3f}s")

    lines += [
        "-" * 62,
        "",
    ]

    if failed == 0:
        lines.append("  STATUS FINAL: VERDE — todos os testes passaram.")
    else:
        lines.append(f"  STATUS FINAL: VERMELHO — {failed} teste(s) falharam.")

    lines.append("=" * 62)

    out = Path(__file__).parent / "RESULTADOS_TESTS_UNITARIOS.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nResultados salvos em: {out}")
