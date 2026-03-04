import random

import pytest


@pytest.fixture(autouse=True)
def fixed_seed() -> None:
    random.seed(12345)
