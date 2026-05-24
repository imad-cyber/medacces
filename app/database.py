from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import settings

# ── Engine ───────────────────────────────────────────────────
# The engine is the actual connection to the database.
# create_engine does NOT open a connection immediately —
# it just configures HOW to connect when needed.
engine = create_engine(
    settings.database_url,
    
    # SQLite-specific argument:
    # SQLite doesn't support multiple threads by default.
    # This tells it to allow it (safe with SQLAlchemy's session handling).
    # For PostgreSQL this line is ignored automatically.
    connect_args=(
        {"check_same_thread": False}
        if "sqlite" in settings.database_url
        else {}
    ),

    # echo=True prints every SQL query to the terminal — useful for debugging.
    # Turn off in production.
    echo=settings.environment == "development",
)


# ── Session Factory ──────────────────────────────────────────
# SessionLocal is a factory — calling SessionLocal() creates a new DB session.
# A session = a single conversation with the database.
# autocommit=False → we commit manually (gives us control + rollback ability)
# autoflush=False  → we control when changes are sent to DB

SessionLocal = sessionmaker(
    autocommit = False,
    autoflush = False,
    bind = engine
)

# ── Base ─────────────────────────────────────────────────────
# All SQLAlchemy models inherit from Base.
# Base keeps track of all models so create_all() can find them.
Base = declarative_base()

# ── Dependency ───────────────────────────────────────────────
def get_db():
    """
    FastAPI dependency that provides a database session to routes.

    The 'yield' makes this a generator — FastAPI calls it like this:
    1. Open a session (db = SessionLocal())
    2. Give it to the route function (yield db)
    3. Route runs and does its work
    4. After route finishes (or crashes), the finally block closes the session

    This guarantees sessions are ALWAYS closed — no connection leaks.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()