import numpy as np
import pandas as pd

from monitoring.drift import compute_drift_report, simulate_drift


def _make_reference(n_rows=200, n_cols=10, seed=42):
    """Crée un DataFrame de référence reproductible."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_rows, n_cols))
    cols = [f"feat_{i}" for i in range(n_cols)]
    return pd.DataFrame(data, columns=cols)


def test_no_drift_identical_data():
    ref = _make_reference()
    report = compute_drift_report(ref, ref)
    assert report["drift_detected"].sum() == 0


def test_simulate_drift_none():
    ref = _make_reference()
    result = simulate_drift(ref, drift_type="none")
    pd.testing.assert_frame_equal(result, ref)


def test_simulate_drift_gradual():
    ref = _make_reference()
    result = simulate_drift(ref, drift_type="gradual", intensity=0.5)
    assert result.shape == ref.shape
    assert not ref.equals(result)


def test_simulate_drift_sudden():
    ref = _make_reference()
    result = simulate_drift(ref, drift_type="sudden", intensity=0.5)
    assert result.shape == ref.shape


def test_simulate_drift_feature_shift():
    ref = _make_reference()
    result = simulate_drift(ref, drift_type="feature_shift", intensity=0.5)
    assert result.shape == ref.shape


def test_drift_detected_with_strong_shift():
    ref = _make_reference(n_rows=500)
    drifted = simulate_drift(ref, drift_type="sudden", intensity=1.0)
    report = compute_drift_report(ref, drifted, top_n=10)
    assert report["drift_detected"].any()


def test_drift_report_columns():
    ref = _make_reference()
    report = compute_drift_report(ref, ref)
    assert list(report.columns) == ["feature", "ks_statistic", "p_value", "drift_detected"]


def test_drift_report_top_n():
    ref = _make_reference(n_cols=50)
    report = compute_drift_report(ref, ref, top_n=10)
    assert len(report) <= 10
