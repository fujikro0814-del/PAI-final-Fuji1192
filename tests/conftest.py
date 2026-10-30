"""提出用リポジトリのテストの設定（書き出しの道具が作った。開発用のリポジトリにはない）。

開発の作業記録の文書（このリポジトリには入れていない）を読むテストを、理由を付けて飛ばす。
"""
import pytest

SKIP = [
    "tests/test_s4_diag_D.py",
    "tests/test_s4_bundle4.py::test_band_rejection",
    "tests/test_s4_bundle4.py::test_ledger_reserved_and_planned_rows_block_but_not_the_band_itself",
    "tests/test_s4_scoring_fix.py::test_audit_bundle1_plan_is_readable",
    "tests/test_s4_test1_eval.py::test_doc_smoke_health_and_expect_values"
]


def pytest_collection_modifyitems(config, items):
    mark = pytest.mark.skip(reason="公開していない開発の作業記録の文書を読むテスト")
    for item in items:
        nid = item.nodeid.replace("\\", "/")
        if any(nid == s or nid.startswith(s + "::") for s in SKIP):
            item.add_marker(mark)
