"""
tests/test_api.py
-----------------
End-to-end API tests using FastAPI TestClient.

TestClient sends real HTTP requests to the app
but runs entirely in-memory — no server needed.

These tests verify:
- Correct HTTP status codes
- Response body structure
- Validation rejection of bad inputs
- Database logging side effects
- Business logic correctness
"""

import pytest
import time


# ── Root Endpoints ───────────────────────────────────────────────────

class TestRootEndpoints:
    """Tests for / and /health."""

    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_endpoints(self, client):
        """Root response acts as API navigation."""
        response = client.get("/")
        data = response.json()
        assert "endpoints" in data
        assert "predict" in data["endpoints"]

    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_health_contains_environment(self, client):
        response = client.get("/health")
        assert "environment" in response.json()


# ── Model Endpoints ──────────────────────────────────────────────────

class TestModelEndpoints:
    """Tests for /model/* endpoints."""

    def test_model_health_returns_healthy(self, client):
        response = client.get("/model/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_model_info_returns_200(self, client):
        response = client.get("/model/info")
        assert response.status_code == 200

    def test_model_info_has_required_fields(self, client):
        """All fields the dashboard depends on must be present."""
        response = client.get("/model/info")
        data = response.json()

        for field in ["model_name", "version", "trained_at",
                      "features", "metrics", "feature_importance"]:
            assert field in data, f"Missing field in /model/info: {field}"

    def test_model_info_metrics_are_numbers(self, client):
        """Metrics must be numeric — not strings or None."""
        metrics = client.get("/model/info").json()["metrics"]
        assert isinstance(metrics["accuracy"], float)
        assert isinstance(metrics["f1_weighted"], float)

    def test_model_retrain_returns_202(self, client):
        """Retrain endpoint accepts request and returns 202."""
        response = client.post("/model/retrain")
        assert response.status_code == 202

    def test_model_retrain_response_structure(self, client):
        response = client.post("/model/retrain")
        data = response.json()
        assert "status" in data
        assert "message" in data


# ── Prediction Endpoints ─────────────────────────────────────────────

class TestPredictionEndpoints:
    """Tests for /predict/* endpoints."""

    def test_valid_prediction_returns_200(
        self, client, rural_commune_data
    ):
        response = client.post(
            "/predict/commune",
            json=rural_commune_data,
        )
        assert response.status_code == 200

    def test_prediction_response_structure(
        self, client, rural_commune_data
    ):
        """Response contains all fields frontend depends on."""
        response = client.post(
            "/predict/commune",
            json=rural_commune_data,
        )
        data = response.json()

        required = [
            "risk_level", "risk_label", "risk_color",
            "confidence", "probabilities", "recommendations",
        ]
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_prediction_risk_level_valid(
        self, client, rural_commune_data
    ):
        response = client.post(
            "/predict/commune",
            json=rural_commune_data,
        )
        assert response.json()["risk_level"] in [0, 1, 2]

    def test_probabilities_sum_to_one(
        self, client, rural_commune_data
    ):
        response = client.post(
            "/predict/commune",
            json=rural_commune_data,
        )
        probs = response.json()["probabilities"]
        total = probs["low"] + probs["medium"] + probs["high"]
        assert total == pytest.approx(1.0, abs=0.01)

    def test_rural_gets_higher_risk_than_urban(
        self, client, rural_commune_data, urban_commune_data
    ):
        """
        Core domain logic test.
        Rural Creuse must always score higher risk than Paris.
        If this fails, stop everything and investigate the model.
        """
        rural = client.post(
            "/predict/commune", json=rural_commune_data
        ).json()
        urban = client.post(
            "/predict/commune", json=urban_commune_data
        ).json()

        assert rural["risk_level"] >= urban["risk_level"], (
            f"Rural ({rural['risk_level']}) should be >= "
            f"Urban ({urban['risk_level']})"
        )

    def test_commune_name_in_response(
        self, client, rural_commune_data
    ):
        """Commune identifiers are passed through to response."""
        response = client.post(
            "/predict/commune",
            json=rural_commune_data,
        )
        assert response.json()["commune_name"] == "Ahun"

    def test_prediction_has_timing_header(
        self, client, rural_commune_data
    ):
        """Middleware adds timing header to every response."""
        response = client.post(
            "/predict/commune",
            json=rural_commune_data,
        )
        assert "x-process-time-ms" in response.headers

    # ── Validation Tests ───────────────────────────────────────────

    def test_missing_required_field_returns_422(self, client):
        """Pydantic validation rejects incomplete input."""
        incomplete = {
            "commune_name": "Test",
            # population_log missing — required field
        }
        response = client.post("/predict/commune", json=incomplete)
        assert response.status_code == 422

    def test_invalid_urban_score_returns_422(
        self, client, rural_commune_data
    ):
        """urban_score must be 0-3. Value of 5 must be rejected."""
        invalid = {**rural_commune_data, "urban_score": 5}
        response = client.post("/predict/commune", json=invalid)
        assert response.status_code == 422

    def test_invalid_elderly_ratio_returns_422(
        self, client, rural_commune_data
    ):
        """elderly_ratio must be 0-1. Value of 1.5 must be rejected."""
        invalid = {**rural_commune_data, "elderly_ratio": 1.5}
        response = client.post("/predict/commune", json=invalid)
        assert response.status_code == 422

    def test_negative_gp_count_returns_422(
        self, client, rural_commune_data
    ):
        """gp_count cannot be negative."""
        invalid = {**rural_commune_data, "gp_count": -1}
        response = client.post("/predict/commune", json=invalid)
        assert response.status_code == 422

    def test_empty_body_returns_422(self, client):
        """Empty body must return validation error."""
        response = client.post("/predict/commune", json={})
        assert response.status_code == 422

    def test_validation_error_has_readable_format(self, client):
        """
        Our custom error handler formats validation errors cleanly.
        Check the shape matches what we defined in main.py.
        """
        response = client.post("/predict/commune", json={})
        data = response.json()
        assert "detail" in data
        assert "errors" in data
        assert isinstance(data["errors"], list)

    # ── Batch Tests ────────────────────────────────────────────────

    def test_batch_with_two_communes(
        self, client, rural_commune_data, urban_commune_data
    ):
        """Batch endpoint returns correct count."""
        response = client.post(
            "/predict/batch",
            json={"communes": [rural_commune_data, urban_commune_data]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert len(data["results"]) == 2

    def test_batch_empty_list_returns_400(self, client):
        """Empty batch must be rejected."""
        response = client.post(
            "/predict/batch",
            json={"communes": []},
        )
        assert response.status_code == 400

    def test_batch_results_have_commune_names(
        self, client, rural_commune_data, urban_commune_data
    ):
        """Commune names are passed through in batch results."""
        response = client.post(
            "/predict/batch",
            json={"communes": [rural_commune_data, urban_commune_data]},
        )
        results = response.json()["results"]
        names = [r["commune_name"] for r in results]
        assert "Ahun" in names
        assert "Paris 8e" in names


# ── Communes / Stats Endpoints ───────────────────────────────────────

class TestCommunesEndpoints:
    """Tests for /communes/* endpoints."""

    def test_stats_returns_200(self, client):
        response = client.get("/communes/stats")
        assert response.status_code == 200

    def test_stats_empty_database(self, client):
        """
        Before any predictions, stats should return zeros.
        Not an error — just empty state.
        """
        data = client.get("/communes/stats").json()
        assert data["total_predictions"] == 0
        assert data["high_risk_count"] == 0

    def test_stats_after_prediction(
        self, client, rural_commune_data
    ):
        """
        After making a prediction, stats should reflect it.
        This tests the full cycle:
        predict → log to DB → stats reads from DB.
        """
        # Make a prediction first
        client.post("/predict/commune", json=rural_commune_data)

        # Small wait — background task needs to complete
        time.sleep(0.1)

        # Stats should now show 1 prediction
        stats = client.get("/communes/stats").json()
        assert stats["total_predictions"] == 1

    def test_dataset_returns_200(self, client):
        response = client.get("/communes/dataset")
        assert response.status_code == 200

    def test_dataset_has_pagination_fields(self, client):
        """Dataset response includes pagination metadata."""
        data = client.get("/communes/dataset?limit=10").json()
        for field in ["total", "returned", "skip", "limit", "data"]:
            assert field in data, f"Missing pagination field: {field}"

    def test_dataset_limit_is_respected(self, client):
        """Limit parameter controls how many rows come back."""
        response = client.get("/communes/dataset?limit=5")
        data = response.json()
        assert data["returned"] <= 5

    def test_dataset_department_filter(self, client):
        """Filtering by department returns only that department."""
        response = client.get("/communes/dataset?department=23")
        data = response.json()
        if data["returned"] > 0:
            dept_codes = [
                row["department_code"] for row in data["data"]
            ]
            assert all(d == "23" for d in dept_codes)

    def test_predictions_log_returns_200(self, client):
        response = client.get("/communes/predictions")
        assert response.status_code == 200

    def test_predictions_log_empty_initially(self, client):
        """Prediction log starts empty for fresh test DB."""
        data = client.get("/communes/predictions").json()
        assert data["total"] == 0

    def test_predictions_log_after_prediction(
        self, client, rural_commune_data
    ):
        """Prediction appears in log after being made."""
        client.post("/predict/commune", json=rural_commune_data)
        time.sleep(0.1)

        data = client.get("/communes/predictions").json()
        assert data["total"] == 1
        assert data["predictions"][0]["commune_name"] == "Ahun"