import pytest

from app.model import ToxicityModel


@pytest.mark.unit
def test_same_seed_produces_same_scores():
    left = ToxicityModel(seed=7)
    right = ToxicityModel(seed=7)
    left.load()
    right.load()

    assert left.score("neutral text") == pytest.approx(right.score("neutral text"))
    assert left.score("you are an idiot") == pytest.approx(
        right.score("you are an idiot")
    )


@pytest.mark.unit
def test_seeded_score_matches_expected_value_with_high_precision():
    model = ToxicityModel(seed=7)
    model.load()

    # With a fixed seed, we can verify tiny floating-point jitter precisely.
    assert model.score("neutral text") == pytest.approx(0.0000323832, rel=0, abs=1e-10)


@pytest.mark.unit
def test_keyword_detection_is_true_for_obvious_toxic_text():
    model = ToxicityModel(seed=17)
    model.load()

    assert model.predict("you are stupid")
    assert not model.predict("thank you and have a nice day")
