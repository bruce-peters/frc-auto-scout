import pytest

from autoscout import attribution


@pytest.fixture(autouse=True)
def hand_tuned_credit_rule():
    """Unit tests exercise the hand-tuned credit rule; the learned model (models/attr_model.json, the runtime
    default) is opted into explicitly by the tests that cover it."""
    old = attribution.ATTR_MODEL
    attribution.ATTR_MODEL = None
    yield
    attribution.ATTR_MODEL = old
