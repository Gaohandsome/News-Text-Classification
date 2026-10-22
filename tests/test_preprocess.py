from pathlib import Path

from news_clf.data.loader import read_examples
from news_clf.data.preprocess import join_tokens, to_fasttext_line

LABELS = [f"c{i}" for i in range(10)]


def test_fasttext_line_prefix():
    line = to_fasttext_line("国足战胜对手", "sports", granularity="word")
    assert line.startswith("__label__sports ")
    assert line.endswith("\n")


def test_char_granularity_keeps_every_char():
    tokens = join_tokens("国足2:1", granularity="char").split()
    assert "国" in tokens and ":" in tokens


def test_read_examples_handles_tab_in_text(tmp_path: Path):
    path = tmp_path / "sample.txt"
    # 正文里带制表符，标签在最后一个 \t 之后
    path.write_text("a\tb\t3\n正常文本\t7\n\n", encoding="utf-8")

    examples = read_examples(path, LABELS)

    assert len(examples) == 2
    assert examples[0].text == "a\tb"
    assert examples[0].label == 3
    assert examples[1].label == 7


def test_read_examples_rejects_out_of_range_label(tmp_path: Path):
    path = tmp_path / "sample.txt"
    path.write_text("文本\t99\n", encoding="utf-8")

    try:
        read_examples(path, LABELS)
    except ValueError as exc:
        assert "为空或全部解析失败" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("越界标签应导致读取失败")
