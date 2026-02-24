from poker_ai.utils.rng import RNG


def test_rng_is_deterministic() -> None:
    r1 = RNG(123).make()
    r2 = RNG(123).make()
    assert [r1.random() for _ in range(5)] == [r2.random() for _ in range(5)]
