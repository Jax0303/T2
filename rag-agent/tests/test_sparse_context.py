from rag_agent.serialization.sparse_context import VisibleCell, render_context, build_visible_lookup, resolve_ids

def test_keeps_duplicate_and_conflicting_cells_and_sparse_holes():
    context = ["a", "b", "c", "d"]
    lookup = {"a": VisibleCell("T", "R1", "C1", "1.00"),
              "b": VisibleCell("T", "R1", "C1", "2"),
              "c": VisibleCell("T", "R2", "C2", "3"),
              "d": VisibleCell("T", "R1", "C1", "1.00")}
    text, audit = render_context(context, lookup, "sparse")
    assert audit["n_preserved_lines"] == 4
    assert "1.00 [#1]<br>2 [#2]<br>1.00 [#4]" in text
    assert "| R2 | — | 3 [#3] |" in text

def test_unknown_and_ambiguous_lines_stay_verbatim():
    text, audit = render_context(["unknown", "ambiguous"], {"ambiguous": None}, "sparse")
    assert "[#1] unknown" in text and "[#2] ambiguous" in text
    assert audit["fallback"] == 2

def test_mt2net_has_no_invented_title():
    text, _ = render_context(["x"], {"x": VisibleCell("", "a of b", "c, d", "7")}, "sparse")
    assert 'Table:' not in text
    assert "a of b" in text and "c, d" in text

def test_escaping_does_not_create_extra_columns():
    text, _ = render_context(["x"], {"x": VisibleCell("T", "a|b", "c\nd", "0.5")}, "sparse")
    assert "a&#124;b" in text and "c&#10;d" in text
    assert "0.5 [#1]" in text

def test_original_is_byte_identical():
    text, _ = render_context(["a", "b"], {}, "original")
    assert text == "a\nb"

def test_grouped_retains_body_and_every_duplicate():
    s = "In the table 'T', among R, the value of C is 1."
    text, audit = render_context([s, s], {s: VisibleCell("T","R","C","1")}, "grouped")
    assert text.count("among R, the value of C is 1.") == 2
    assert audit["n_preserved_lines"] == 2


def test_hierarchy_separator_is_not_html_encoded():
    text, _ = render_context(["x"], {"x": VisibleCell("T", "A > B", "C > D", "1")}, "sparse")
    assert "A > B" in text and "C > D" in text


def test_cellid_layout_numbers_every_line_and_preserves_it():
    context = [f"In the table 't', among r{i}, the value of c is {i}." for i in range(1, 21)]
    text, info = render_context(context, {}, "cellid")
    assert info["n_preserved_lines"] == 20
    for rank, line in enumerate(context, 1):
        assert f"[#{rank}] {line}" in text


def test_resolve_ids_returns_the_stored_value_not_the_models_text():
    cells = [VisibleCell("t", f"r{i}", "c", f"{i}.50") for i in range(1, 4)]
    context = [f"line {i}" for i in range(1, 4)]
    lookup = dict(zip(context, cells))
    assert resolve_ids("[#2]", context, lookup) == ("2.50", {"ids": [2], "n_ids": 1})
    # several cells -> the same comma rule BASE uses for several values
    value, audit = resolve_ids("[#1], [#3]", context, lookup)
    assert value == "1.50, 3.50" and audit["n_ids"] == 2
    # the model writing a value instead of a rank is a breakage, not an answer
    assert resolve_ids("42", context, lookup)[1]["failure"] == "no_id"
    assert resolve_ids("[#99]", context, lookup)[1]["failure"] == "out_of_range"
    # a line the corpus dictionary cannot decompose has no value to return
    assert resolve_ids("[#1]", context, {})[1]["failure"] == "unresolved_value"
