from src.visualization.run_loss_landscape import parse_seeds


def test_parse_seeds_from_comma_string():
    assert parse_seeds('42,123,456', 7) == [42, 123, 456]


def test_parse_seeds_fallback_to_seed():
    assert parse_seeds(None, 7) == [7]
    assert parse_seeds('', 11) == [11]
