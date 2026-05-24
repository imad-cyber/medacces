"""
tests/conftest.py
-----------------
Shared fixtures for all tests.

conftest.py is automatically loaded by pytest — no imports needed.
Fixtures defined here are available in every test file.

Key decisions:
- Use SQLite in-memory database for tests (fast, isolated, no cleanup)
- Override the get_db dependency so tests use test DB not production DB
- Fresh database state for every test function (no bleed-over between tests)
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.database import Base, get_db
from app.models.db_models import PredictionLog


# ── Test Database Setup ─────────────────────────────────────────────
# In-memory SQLite — lives only during the test run
# StaticPool → same connection reused (required for in-memory SQLite)
TEST_DATABASE_URL = "sqlite://"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,    # critical for in-memory SQLite
)

TestingSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=test_engine,
)


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="function")
def test_db():
    """
    Creates fresh database tables before each test.
    Drops everything after each test.

    scope="function" means this runs for EVERY test function.
    Each test gets a clean, empty database — no leftover data.

    Why drop and recreate?
    Test A creates a prediction. Test B counts predictions.
    Without cleanup, Test B sees Test A's data → flaky tests.
    """
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture(scope="function")
def db_session(test_db):
    """
    Provides a database session for tests that
    need to query the DB directly (not via API).
    """
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="function")
def client(test_db):
    """
    Provides a FastAPI TestClient with test database injected.

    The key here is dependency_overrides:
    We replace get_db (the production DB function)
    with get_test_db (our test DB function).
    The API thinks it's using the real database
    but it's actually using the isolated test database.

    This is the correct way to test FastAPI apps —
    never point tests at your real database.
    """

    def get_test_db():
        """Test version of get_db — uses in-memory SQLite."""
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    # Override the dependency
    app.dependency_overrides[get_db] = get_test_db

    with TestClient(app) as test_client:
        yield test_client

    # Clean up — restore real dependency after test
    app.dependency_overrides.clear()


# ── Shared Test Data ─────────────────────────────────────────────────

@pytest.fixture
def rural_commune_data():
    """
    A typical high-risk rural commune (Creuse department).
    Used across multiple tests — defined once here.
    """
    return {
        "commune_name":           "Ahun",
        "commune_code":           "23001",
        "department_code":        "23",
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
def urban_commune_data():
    """
    A typical low-risk urban commune (Paris).
    """
    return {
        "commune_name":           "Paris 8e",
        "commune_code":           "75108",
        "department_code":        "75",
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