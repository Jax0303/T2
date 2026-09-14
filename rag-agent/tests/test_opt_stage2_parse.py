from scripts.opt_stage2 import parse


def test_parse_rules():
    assert parse('["a", "b"]') == (["a", "b"], 0)
    assert parse('Queries: ["a", " a ", "b"] done') == (["a", "b"], 1)    # strip 후 완전 중복 제거
    assert parse('["", "  "]') == (None, 0)                                # 남는 문자열 없음 → 대체
    assert parse("no list here") == (None, 0)
    assert parse('[1, "a"]') == (None, 0)                                  # 문자열만 허용
    assert parse('["a"] then ["b"]') == (None, 0)                          # 첫 [ ~ 마지막 ] 가 JSON 이 아님
