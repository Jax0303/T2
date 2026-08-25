"""A reader spec has to reach the loader, and has to survive into the name.

Both halves matter for the same reason. `--reader local:...?quantization=8bit`
was accepted and silently ignored, so an 8-bit run was a 4-bit run; and even
once it is honoured, 4-bit and 8-bit of the same weights are different readers,
so guard_resume must refuse to join their records.
"""
from rag_agent.llm.factory import _split_spec


def test_query_suffix_becomes_kwargs():
    assert _split_spec("Qwen/Qwen2.5-7B-Instruct?quantization=8bit") == (
        "Qwen/Qwen2.5-7B-Instruct", {"quantization": "8bit"})


def test_several_and_ints():
    name, kw = _split_spec("Qwen/Qwen2.5-Math-7B-Instruct"
                           "?quantization=4bit&default_max_tokens=512")
    assert name == "Qwen/Qwen2.5-Math-7B-Instruct"
    assert kw == {"quantization": "4bit", "default_max_tokens": 512}


def test_bare_spec_is_untouched():
    assert _split_spec("Qwen/Qwen2.5-7B-Instruct") == ("Qwen/Qwen2.5-7B-Instruct", {})
    assert _split_spec("openai/gpt-oss-120b") == ("openai/gpt-oss-120b", {})


def test_quantization_is_part_of_the_reader_name():
    # constructing LocalQwenLLM would download weights, so check the one line
    # that builds the name rather than the loader around it
    import inspect

    from rag_agent.llm.local_qwen import LocalQwenLLM
    src = inspect.getsource(LocalQwenLLM.__init__)
    assert 'f"?quantization={quantization}"' in src, \
        "the reader name must carry the quantization or resume joins 4-bit onto 8-bit"


if __name__ == "__main__":
    test_query_suffix_becomes_kwargs()
    test_several_and_ints()
    test_bare_spec_is_untouched()
    test_quantization_is_part_of_the_reader_name()
    print("ok")
