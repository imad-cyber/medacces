"""
tests/test_predict.py
---------------------
Tests for the ML inference module (ml/predict.py).

These tests verify:
- Model loads correctly
- Predictions return correct structure
- Prediction values are in valid ranges
- Batch prediction works
- Rural communes score higher risk than urban ones
"""

import pytest
from ml.predict import load_model, predict_one, predict_batch, get_metadata


# ── Setup ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def loaded_model():
    """
    Load model once for all tests in this file.
    scope="module" → runs once per file, not per test.
    Model loading takes ~200ms — no need to repeat it.
    """
    load_model()


# ── Model Loading Tests ──────────────────────────────────────────────

class TestModelLoading:
    """Tests that the model loads correctly."""

    def test_model_loads_without_error(self, loaded_model):
        """Model loads and metadata is available."""
        metadata = get_metadata()
        assert metadata is not None

    def test_metadata_has_required_fields(self, loaded_model):
        """Metadata contains all fields the API depends on."""
        metadata = get_metadata()

        required_fields = [
            "model_name", "version", "trained_at",
            "features", "metrics", "feature_importance",
        ]
        for field in required_fields:
            assert field in metadata, f"Missing field: {field}"

    def test_metadata_metrics_are_valid(self, loaded_model):
        """Model metrics are in valid ranges."""
        metrics = get_metadata()["metrics"]

        # Accuracy must be between 0 and 1
        assert 0.0 <= metrics["accuracy"] <= 1.0

        # Must beat random guessing (33% for 3 classes)
        assert metrics["accuracy"] > 0.33, (
            f"Model accuracy {metrics['accuracy']} "
            "is no better than random guessing"
        )

    def test_correct_number_of_features(self, loaded_model):
        """Model expects exactly 10 features."""
        features = get_metadata()["features"]
        assert len(features) == 10


# ── Single Prediction Tests ──────────────────────────────────────────

class TestPredictOne:
    """Tests for the predict_one() function."""

    @pytest.fixture
    def rural_features(self):
        return {
            "population_log":         7.1,
            "urban_score":            0,
            "elderly_ratio":          0.38,
            "gp_count":               1,
            "specialist_density":     25.0,
            "pharmacy_score":         1.5,
            "wealth_index":           0.32,
            "population_growth_rate": -0.02,
            "avg_gp_age":             61.0,
            "teleconsult_score":      1.5,
        }

    @pytest.fixture
    def urban_features(self):
        return {
            "population_log":         11.5,
            "urban_score":            3,
            "elderly_ratio":          0.16,
            "gp_count":               85,
            "specialist_density":     180.0,
            "pharmacy_score":         4.8,
            "wealth_index":           0.92,
            "population_growth_rate": 0.02,
            "avg_gp_age":             49.0,
            "teleconsult_score":      4.5,
        }

    def test_returns_required_keys(self, loaded_model, rural_features):
        """Prediction result contains all expected keys."""
        result = predict_one(rural_features)

        required_keys = [
            "risk_level", "risk_label", "risk_color",
            "confidence", "probabilities", "recommendations",
            "model_version",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_risk_level_is_valid(self, loaded_model, rural_features):
        """Risk level is always 0, 1, or 2."""
        result = predict_one(rural_features)
        assert result["risk_level"] in [0, 1, 2]

    def test_risk_label_matches_level(self, loaded_model, rural_features):
        """Risk label matches the risk level."""
        label_map = {0: "Low", 1: "Medium", 2: "High"}
        result = predict_one(rural_features)
        expected_label = label_map[result["risk_level"]]
        assert result["risk_label"] == expected_label

    def test_confidence_is_valid_probability(self, loaded_model, rural_features):
        """Confidence is between 0 and 1."""
        result = predict_one(rural_features)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_probabilities_sum_to_one(self, loaded_model, rural_features):
        """All class probabilities must sum to 1.0."""
        result = predict_one(rural_features)
        probs = result["probabilities"]
        total = probs["low"] + probs["medium"] + probs["high"]
        # Use pytest.approx for floating point comparison
        assert total == pytest.approx(1.0, abs=0.01)

    def test_recommendations_not_empty(self, loaded_model, rural_features):
        """Every prediction includes at least one recommendation."""
        result = predict_one(rural_features)
        assert len(result["recommendations"]) >= 1

    def test_rural_scores_higher_than_urban(
        self, loaded_model, rural_features, urban_features
    ):
        """
        Core business logic test:
        Rural Creuse commune must score higher risk than Paris.

        If this fails, the model has learned something wrong.
        This is called a sanity test — validates domain knowledge.
        """
        rural_result = predict_one(rural_features)
        urban_result = predict_one(urban_features)

        assert rural_result["risk_level"] >= urban_result["risk_level"], (
            f"Rural risk ({rural_result['risk_level']}) should be >= "
            f"urban risk ({urban_result['risk_level']})"
        )

    def test_rural_high_confidence(self, loaded_model, rural_features):
        """
        Creuse is France's most well-known medical desert.
        The model should be confident about this.
        """
        result = predict_one(rural_features)
        assert result["confidence"] >= 0.70, (
            f"Model should be confident about Creuse risk "
            f"but confidence is only {result['confidence']:.1%}"
        )


# ── Batch Prediction Tests ───────────────────────────────────────────

class TestPredictBatch:
    """Tests for the predict_batch() function."""

    @pytest.fixture
    def batch_records(self):
        return [
            {   # rural
                "population_log": 7.1, "urban_score": 0,
                "elderly_ratio": 0.38, "gp_count": 1,
                "specialist_density": 25.0, "pharmacy_score": 1.5,
                "wealth_index": 0.32, "population_growth_rate": -0.02,
                "avg_gp_age": 61.0, "teleconsult_score": 1.5,
            },
            {   # urban
                "population_log": 11.5, "urban_score": 3,
                "elderly_ratio": 0.16, "gp_count": 85,
                "specialist_density": 180.0, "pharmacy_score": 4.8,
                "wealth_index": 0.92, "population_growth_rate": 0.02,
                "avg_gp_age": 49.0, "teleconsult_score": 4.5,
            },
        ]

    def test_batch_returns_correct_count(self, loaded_model, batch_records):
        """Batch returns same number of results as inputs."""
        results = predict_batch(batch_records)
        assert len(results) == len(batch_records)

    def test_batch_each_result_is_valid(self, loaded_model, batch_records):
        """Every result in batch has valid structure."""
        results = predict_batch(batch_records)
        for result in results:
            assert result["risk_level"] in [0, 1, 2]
            assert 0.0 <= result["confidence"] <= 1.0

    def test_single_item_batch(self, loaded_model, batch_records):
        """Batch works with just one item."""
        results = predict_batch([batch_records[0]])
        assert len(results) == 1