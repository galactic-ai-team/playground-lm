import playground_lm


def test_version():
    assert playground_lm.__version__


def test_tokenizer_exported():
    assert callable(playground_lm.get_tokenizer)
