# ruff: noqa: E501
"""TEMPORARY diagnostic (removed before review): isolate a Windows heap corruption."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

HEADER = """
import sys, gc, numpy as np
sys.path.insert(0, {tests!r})
import test_parametric_solver as t
from crc_sdk.parametric import _solver
def churn():
    for _ in range(200):
        a = np.random.default_rng(0).normal(size=(50, 50)); a @ a; gc.collect()
"""
SCENARIOS = {
    "qp-stall": "p = t._design_qp(t.STALLS)\nfor _ in range(3):\n    print(_solver._run(p, _solver.QP_OPTIONS, None).status); churn()",
    "qp-unbounded": "p = t._design_qp(t.FALSE_UNBOUNDED)\nfor _ in range(3):\n    print(_solver._run(p, _solver.QP_OPTIONS, None).status); churn()",
    "qp-stall-iterations-only": "p = t._design_qp(t.STALLS)\nfor _ in range(3):\n    print(_solver._run(p, {'qp_regularization_value': 1e-12, 'qp_iteration_limit': 100000}, None).status); churn()",
    "qp-stall-time-only": "p = t._design_qp(t.STALLS)\nprint(_solver._run(p, {'qp_regularization_value': 1e-12, 'time_limit': 1.0}, None).status); churn()",
    "qp-unbounded-no-limits": "p = t._design_qp(t.FALSE_UNBOUNDED)\nfor _ in range(3):\n    print(_solver._run(p, {'qp_regularization_value': 1e-12}, None).status); churn()",
    "qp-control": "p = t._qp(np.eye(2) * 2.0, [-2.0, -2.0], [[1.0, 1.0]], [1.0])\nfor _ in range(3):\n    print(_solver._run(p, _solver.QP_OPTIONS, None).status); churn()",
    "start-lp": "p = t._design_qp(t.STALLS)\nfor _ in range(3):\n    print(_solver._start(p).status); churn()",
    "active-set-only": "p = t._design_qp(t.STALLS)\nfor _ in range(3):\n    print(_solver._active_set(p, 1000).status); churn()",
    "descent-lp": "g = np.array([1.0, -1.0, 0.5])\nfor rows in (np.zeros((0, 3)), np.eye(3)[:2]):\n    print(_solver._descent(g, rows, 1.0) is not None); churn()",
    "solve-qp": "p = t._design_qp(t.STALLS)\nfor _ in range(3):\n    print(_solver.solve_qp(p).status); churn()",
}


@pytest.mark.skipif(sys.platform != "win32", reason="a Windows diagnostic")
def test_windows_probe() -> None:
    tests = str(Path(__file__).parent)
    report = []
    for name, body in SCENARIOS.items():
        code = HEADER.format(tests=tests) + body + "\nprint('done')\n"
        result = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-c", code],
            capture_output=True,
            text=True,
            timeout=300,
        )
        report.append(
            f"{name}: rc={result.returncode} out={result.stdout.split()!r} "
            f"err={result.stderr[-300:]!r}"
        )
    raise AssertionError("\n" + "\n".join(report))
