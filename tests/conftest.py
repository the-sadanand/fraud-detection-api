import os

# Must be set before `app` is imported: don't try to reach MLflow in tests.
os.environ["LOAD_MODEL_ON_STARTUP"] = "false"

import numpy as np
import pytest


class FakeModel:
    """Mimics a sklearn classifier: risky locations -> high fraud probability."""

    def predict_proba(self, df):
        p = df["location_risk_score"].to_numpy(dtype=float)
        return np.column_stack([1 - p, p])

    def predict(self, df):
        return (self.predict_proba(df)[:, 1] >= 0.5).astype(int)


@pytest.fixture
def fake_model():
    return FakeModel()
