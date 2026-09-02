"""NotebookLM collections (google keys) — account-level notebook grouping.

New in notebooklm-py 0.8.1. A collection groups notebooks at the account
level, so a key's notebooks can be organized without touching their contents.

  GET    /v1/notebooklm/collections                     list collections
  POST   /v1/notebooklm/collections                     create one
  GET    /v1/notebooklm/collections/{id}                one collection
  GET    /v1/notebooklm/collections/{id}/notebooks      its notebooks
  POST   /v1/notebooklm/collections/{id}/rename         rename it
  POST   /v1/notebooklm/collections/{id}/notebooks      add notebooks
  DELETE /v1/notebooklm/collections/{id}/notebooks      remove notebooks
  DELETE /v1/notebooklm/collections/{id}                delete it
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import List

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict

from ..pool import get_notebook_client
from ..schemas import openai_error
from ..security import AuthContext, describe_error, log_request, require_google_auth
from .notebooklm import _to_dict

logger = logging.getLogger("ai-sidecar.notebooklm.collections")
router = APIRouter()


class CollectionCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str


class CollectionRenameRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str


class CollectionMembersRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    notebook_ids: List[str]


@asynccontextmanager
async def _tracked(ctx: AuthContext, endpoint: str, target: str):
    """Same timing/status/error logging the other route modules do inline.

    There are eight near-identical handlers here, so the shared try/finally
    lives in one place rather than being copy-pasted eight times; the status
    and error semantics are unchanged.
    """
    started = time.perf_counter()
    state = {"status": 500, "error": None}
    try:
        yield state
        state["status"] = 200
    except Exception as exc:
        state["status"] = getattr(exc, "status_code", 500)
        state["error"] = describe_error(exc)
        raise
    finally:
        log_request(ctx, endpoint, target, state["status"], started, state["error"])


async def _collection_or_404(client, collection_id: str):
    """Fetch a collection, shaping the miss as our own 404.

    Uses get_or_none() rather than get(): 0.8.0 made get() raise, and
    get_or_none is the sanctioned None-on-miss path — which lets the error
    come out in the same OpenAI shape as every other route here.
    """
    collection = await client.collections.get_or_none(collection_id)
    if collection is None:
        raise openai_error(
            404, f"Collection '{collection_id}' was not found.", code="not_found"
        )
    return collection


@router.get("/v1/notebooklm/collections")
async def list_collections(ctx: AuthContext = Depends(require_google_auth)):
    async with _tracked(ctx, "/v1/notebooklm/collections", "-"):
        client = await get_notebook_client(ctx.profile_name)
        # NOTE: never inspect.signature() collections.list — on Python 3.14
        # the `list` method shadows the builtin in its own annotations and
        # introspection raises. Calling it is fine.
        collections = await client.collections.list()
        return {"collections": [_to_dict(c) for c in collections]}


@router.post("/v1/notebooklm/collections")
async def create_collection(
    request: CollectionCreateRequest,
    ctx: AuthContext = Depends(require_google_auth),
):
    async with _tracked(ctx, "/v1/notebooklm/collections/create", request.name):
        client = await get_notebook_client(ctx.profile_name)
        return _to_dict(await client.collections.create(request.name))


@router.get("/v1/notebooklm/collections/{collection_id}")
async def get_collection(
    collection_id: str, ctx: AuthContext = Depends(require_google_auth)
):
    async with _tracked(ctx, "/v1/notebooklm/collections/get", collection_id):
        client = await get_notebook_client(ctx.profile_name)
        return _to_dict(await _collection_or_404(client, collection_id))


@router.get("/v1/notebooklm/collections/{collection_id}/notebooks")
async def collection_notebooks(
    collection_id: str, ctx: AuthContext = Depends(require_google_auth)
):
    async with _tracked(ctx, "/v1/notebooklm/collections/notebooks", collection_id):
        client = await get_notebook_client(ctx.profile_name)
        await _collection_or_404(client, collection_id)
        notebooks = await client.collections.notebooks(collection_id)
        return {
            "collection_id": collection_id,
            "notebooks": [_to_dict(n) for n in notebooks],
        }


@router.post("/v1/notebooklm/collections/{collection_id}/rename")
async def rename_collection(
    collection_id: str,
    request: CollectionRenameRequest,
    ctx: AuthContext = Depends(require_google_auth),
):
    async with _tracked(ctx, "/v1/notebooklm/collections/rename", collection_id):
        client = await get_notebook_client(ctx.profile_name)
        await _collection_or_404(client, collection_id)
        result = await client.collections.rename(collection_id, request.name)
        return {"collection_id": collection_id, "renamed": True, **_to_dict(result)}


@router.post("/v1/notebooklm/collections/{collection_id}/notebooks")
async def add_notebooks(
    collection_id: str,
    request: CollectionMembersRequest,
    ctx: AuthContext = Depends(require_google_auth),
):
    async with _tracked(ctx, "/v1/notebooklm/collections/add", collection_id):
        client = await get_notebook_client(ctx.profile_name)
        await _collection_or_404(client, collection_id)
        result = await client.collections.add_notebooks(
            collection_id, request.notebook_ids
        )
        return {
            "collection_id": collection_id,
            "added": request.notebook_ids,
            **_to_dict(result),
        }


@router.delete("/v1/notebooklm/collections/{collection_id}/notebooks")
async def remove_notebooks(
    collection_id: str,
    request: CollectionMembersRequest,
    ctx: AuthContext = Depends(require_google_auth),
):
    async with _tracked(ctx, "/v1/notebooklm/collections/remove", collection_id):
        client = await get_notebook_client(ctx.profile_name)
        await _collection_or_404(client, collection_id)
        result = await client.collections.remove_notebooks(
            collection_id, request.notebook_ids
        )
        return {
            "collection_id": collection_id,
            "removed": request.notebook_ids,
            **_to_dict(result),
        }


@router.delete("/v1/notebooklm/collections/{collection_id}")
async def delete_collection(
    collection_id: str, ctx: AuthContext = Depends(require_google_auth)
):
    async with _tracked(ctx, "/v1/notebooklm/collections/delete", collection_id):
        client = await get_notebook_client(ctx.profile_name)
        await _collection_or_404(client, collection_id)
        # Returns None; success is the absence of an exception.
        await client.collections.delete(collection_id)
        return {"collection_id": collection_id, "deleted": True}
