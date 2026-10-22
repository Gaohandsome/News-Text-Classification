from news_clf.labels import canonical_label

LABELS = ("finance", "realty", "sports", "game", "entertainment")


def test_exact_match():
    assert canonical_label("sports", LABELS) == "sports"
    assert canonical_label("  SPORTS \n", LABELS) == "sports"


def test_prefix_and_punctuation():
    assert canonical_label("类别：finance。", LABELS) == "finance"
    assert canonical_label("`game`", LABELS) == "game"


def test_longest_substring_wins():
    # "entertainment" 同时包含 "game" 之外的干扰项时，应取最长匹配
    assert canonical_label("这个属于 entertainment 类别", LABELS) == "entertainment"


def test_unknown_returns_none():
    assert canonical_label("unknown", LABELS) is None
    assert canonical_label("", LABELS) is None
