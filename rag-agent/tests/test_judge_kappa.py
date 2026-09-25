import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from judge_kappa import kappa, report  # noqa: E402


def test_kappa_known_value():
    # 일치율 .75, 우연 일치 .5*.25 + .5*.75 = .5 -> κ = (.75 - .5) / (1 - .5) = .5
    a = ["예", "예", "아니오", "아니오"]
    b = ["예", "아니오", "아니오", "아니오"]
    assert kappa(a, b) == 0.5
    r = report(a, b, n_boot=200)
    assert r["agree"] == 3 and r["table_A_by_B"] == {"예|예": 1, "예|아니오": 1, "아니오|예": 0, "아니오|아니오": 2}
