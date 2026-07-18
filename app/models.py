"""SQLAlchemy models: api keys, Google profiles, ComfyUI instances, request logs."""

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import JSON, BigInteger, Boolean, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


def utcnow() -> datetime:
    """Naive UTC timestamp (MySQL DATETIME is timezone-naive)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class ApiKey(Base):
    """Bearer keys, two kinds:

    * key_type="google" — bound to a Google profile; valid for Gemini AND
      NotebookLM (they share the profile's cookies).
    * key_type="comfy"  — bound to exactly one ComfyUI instance; the only
      kind accepted by the image endpoints.
    """

    __tablename__ = "api_keys"

    key_string: Mapped[str] = mapped_column(String(255), primary_key=True)
    key_type: Mapped[str] = mapped_column(String(16), nullable=False, default="google")
    profile_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    comfy_instance: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    label: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True, default=utcnow
    )
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    request_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)

    def __str__(self) -> str:
        target = (
            f"comfy:{self.comfy_instance}"
            if self.key_type == "comfy"
            else self.profile_name
        )
        return f"{self.label or self.key_string[:14] + '…'} → {target}"


class GoogleProfile(Base):
    """DB is the source of truth for Google auth (Playwright storage state)."""

    __tablename__ = "google_profiles"

    profile_name: Mapped[str] = mapped_column(String(255), primary_key=True)
    storage_state: Mapped[Any] = mapped_column(JSON, nullable=False, default=dict)
    state_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    # active | expired | pending_login | error
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_synced_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    def __str__(self) -> str:
        return f"{self.profile_name} ({self.status})"


class ComfyInstance(Base):
    __tablename__ = "comfy_instances"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    base_url: Mapped[str] = mapped_column(String(512), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    def __str__(self) -> str:
        return self.name


class GeminiConversation(Base):
    """Server-side Gemini chat threads: [cid, rid, rcid] metadata per id."""

    __tablename__ = "gemini_conversations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    profile_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    # First user message excerpt — the sidecar's own history listing
    # (Gemini's web sidebar does not show API-created conversations).
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    # "metadata" is reserved on Declarative classes, hence the attribute name.
    chat_metadata: Mapped[Any] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True, default=utcnow
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True, default=utcnow
    )


class RequestLog(Base):
    __tablename__ = "request_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=utcnow, index=True
    )
    api_key: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    profile: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    endpoint: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    status: Mapped[int] = mapped_column(Integer, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
