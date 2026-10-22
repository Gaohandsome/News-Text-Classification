import pytest

from news_clf.registry import MODEL_SPECS, list_models, resolve_key


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("rf", "tfidf-rf"),
        ("Random-Forest", "tfidf-rf"),
        ("ft", "fasttext"),
        ("int8", "bert-quantized"),
        ("distilled", "bilstm-distilled"),
        ("pruned", "bert-pruned"),
        ("LLM", "llm"),
    ],
)
def test_aliases_resolve(alias, expected):
    assert resolve_key(alias) == expected


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="未知模型"):
        resolve_key("transformer-xl")


def test_all_registered_models_are_listed():
    items = list_models()
    assert len(items) == len(MODEL_SPECS) == 7
    assert all({"key", "display_name", "available", "status"} <= set(item) for item in items)
