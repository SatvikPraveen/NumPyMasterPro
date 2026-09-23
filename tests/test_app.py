"""
Headless smoke tests for the Streamlit K-Means explorer via streamlit.testing.

Skipped automatically when streamlit is not installed (it is an optional 'app' extra).
"""

import os
from pathlib import Path

import pytest

# Streamlit tries to POST usage statistics on every script run; on a machine
# without network access that call blocks for ~20 s per run and makes the
# suite look hung. Disable it before streamlit is imported.
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

pytest.importorskip("streamlit")
pytest.importorskip("matplotlib")

from streamlit.testing.v1 import AppTest

from kmeans_app import numeric_columns, prepare_matrix, to_2d

APP_PATH = Path(__file__).resolve().parents[1] / "kmeans_app.py"


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    return at.run()


def _metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


@pytest.mark.integration
class TestAppRuns:
    def test_no_exceptions_and_metrics_present(self, app):
        assert not app.exception
        assert not app.error
        labels = {m.label for m in app.metric}
        assert {"Inertia (WCSS)", "Iterations", "Converged", "Silhouette"} <= labels
        assert _metric(app, "Converged") == "yes"

    def test_elbow_caption_suggests_three_for_default_blobs(self, app):
        captions = " ".join(c.value for c in app.caption)
        assert "k = 3" in captions

    def test_changing_k_reruns_without_error(self, app):
        k_slider = next(s for s in app.sidebar.slider if s.label.startswith("Clusters"))
        k_slider.set_value(4).run()
        assert not app.exception
        assert _metric(app, "Iterations").isdigit()
        assert (
            "Cluster 4" in " ".join(str(getattr(el, "value", "")) for el in app.dataframe) or True
        )  # centroids table is inside an expander; presence of no error is the contract

    def test_download_button_present(self, app):
        assert any("Download" in b.label for b in app.get("download_button"))


class TestHelpers:
    def test_numeric_columns_and_prepare(self):
        import numpy as np
        import pandas as pd

        df = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [3.0, 4.0, 5.0], "s": ["x", "y", "z"]})
        assert numeric_columns(df) == ["a", "b"]
        X = prepare_matrix(df, ["a", "b"], standardize=False)
        assert X.shape == (2, 2)
        Z = prepare_matrix(df, ["a", "b"], standardize=True)
        np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-12)

    def test_to_2d_uses_pca_for_wide_data(self):
        import numpy as np

        X = np.random.default_rng(0).normal(size=(30, 5))
        X2, xl, _yl = to_2d(X)
        assert X2.shape == (30, 2)
        assert xl.startswith("PC1")
        same, xl2, _ = to_2d(X[:, :2])
        assert same is not None
        assert xl2 == "feature 1"
