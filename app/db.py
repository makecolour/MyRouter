"""Async SQLAlchemy engine/session plus startup schema management."""

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from .config import settings

logger = logging.getLogger("ai-sidecar.db")


class Base(DeclarativeBase):
    pass


# pool_pre_ping/pool_recycle: MySQL silently drops idle connections.
engine = create_async_engine(
    settings.database_url, pool_pre_ping=True, pool_recycle=3600
)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

# Columns added to older api_keys tables; create_all does not ALTER
# existing tables, so these are applied idempotently at startup.
_API_KEYS_MIGRATIONS = {
    "enabled": "ADD COLUMN enabled TINYINT(1) NOT NULL DEFAULT 1",
    "label": "ADD COLUMN label VARCHAR(255) NULL",
    "created_at": "ADD COLUMN created_at DATETIME NULL",
    "last_used_at": "ADD COLUMN last_used_at DATETIME NULL",
    "request_count": "ADD COLUMN request_count BIGINT NOT NULL DEFAULT 0",
    # v3.5 key kinds: google (profile-bound) vs comfy (instance-bound)
    "key_type": "ADD COLUMN key_type VARCHAR(16) NOT NULL DEFAULT 'google'",
    "comfy_instance": "ADD COLUMN comfy_instance VARCHAR(64) NULL",
}


# Same idempotent pattern for columns added to gemini_conversations later.
_GEMINI_CONV_MIGRATIONS = {
    "title": "ADD COLUMN title VARCHAR(255) NULL",
}


async def _table_columns(conn, table: str) -> dict:
    rows = (
        await conn.execute(
            text(
                "SELECT column_name, is_nullable "
                "FROM information_schema.columns "
                "WHERE table_schema = DATABASE() AND table_name = :table"
            ),
            {"table": table},
        )
    ).all()
    return {name.lower(): nullable for name, nullable in rows}


async def ensure_schema() -> None:
    from . import models  # noqa: F401  — register all mappings on Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        existing = await _table_columns(conn, "api_keys")
        for column, ddl in _API_KEYS_MIGRATIONS.items():
            if column not in existing:
                logger.info("Migrating api_keys: %s", ddl)
                await conn.execute(text(f"ALTER TABLE api_keys {ddl}"))

        # Comfy keys have no Google profile — profile_name must be NULLable.
        if existing.get("profile_name", "YES").upper() == "NO":
            logger.info("Migrating api_keys: profile_name -> NULLable")
            await conn.execute(
                text("ALTER TABLE api_keys MODIFY profile_name VARCHAR(255) NULL")
            )

        existing = await _table_columns(conn, "gemini_conversations")
        for column, ddl in _GEMINI_CONV_MIGRATIONS.items():
            if existing and column not in existing:
                logger.info("Migrating gemini_conversations: %s", ddl)
                await conn.execute(
                    text(f"ALTER TABLE gemini_conversations {ddl}")
                )
