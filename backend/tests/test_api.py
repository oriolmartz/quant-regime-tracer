from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import API_VERSION, app

client = TestClient(app)


def test_health_endpoint_reports_current_version() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == API_VERSION == "0.10.0"


def test_data_sources_endpoint_documents_real_first_policy() -> None:
    response = client.get("/data-sources")
    assert response.status_code == 200
    body = response.json()
    assert body["default_mode"] == "real"
    assert "real" in body["modes"]


def test_time_windows_endpoint_includes_two_year_preset() -> None:
    response = client.get("/time-windows")
    assert response.status_code == 200
    assert "2Y" in response.json()["windows"]


def test_analyze_endpoint_returns_guarded_report() -> None:
    response = client.post("/analyze", json={"asset": "SPY", "interval": "1Y", "data_mode": "sample", "language": "en"})
    assert response.status_code == 200
    body = response.json()
    assert body["api_version"] == "0.10.0"
    assert body["asset"] == "SPY"
    assert body["source_report"]["mode"] == "sample"
    assert body["source_report"]["is_real_data"] is False
    assert body["time_series"]
    assert body["time_series"][-1].get("posterior_entropy") is not None
    assert body["transition_matrix"]
    assert body["risk_metrics"]["annualization_factor"] == 252.0
    assert body["model_evaluation"]["status"] in {"ok", "unavailable", "failed", "insufficient_window"}
    assert body["current_traceback"]
    assert body["traceback_points"]
    assert body["current_traceback"]["feature_evidence"]
    assert body["current_traceback"]["baseline_votes"]
    assert "temporal holdout" in body["report_markdown"].lower()
    assert "financial advice" in body["report_markdown"].lower()


def test_compare_endpoint_deduplicates_assets_and_limits_output() -> None:
    response = client.post("/compare", json={"assets": ["SPY", "spy", "QQQ"], "interval": "6M", "data_mode": "sample"})
    assert response.status_code == 200
    body = response.json()
    assert body["assets_requested"] == ["SPY", "QQQ"]
    assert len(body["summaries"]) == 2
    assert body["highest_risk_asset"] in {"SPY", "QQQ"}
