"""ComfyUI self-provisioning via ComfyUI-Manager V3.41.

Push a workflow to an instance and have the Manager install missing custom
nodes and download missing models/embeddings (queue-based install API):

  POST /api/manager/queue/reset          (clear the queue)
  POST /api/manager/queue/install        (enqueue a custom-node pack)
  POST /api/manager/queue/install_model  (enqueue a model download)
  POST /api/manager/queue/start          (begin processing)
  GET  /api/manager/queue/status         -> {total_count, done_count,
                                             in_progress_count, is_processing}

Node/model detection is GET-only (safe); installs are POSTed only when the
caller explicitly provisions. Model metadata rides the full-graph workflow's
top-level `models` array [{name, url, directory, hash}].
"""

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Set

import httpx

from . import comfy
from .config import settings
from .schemas import openai_error

logger = logging.getLogger("ai-sidecar.comfy-provision")


def _client() -> httpx.AsyncClient:
    if comfy._http is None:  # noqa: SLF001 — shared lifespan client
        raise openai_error(503, "ComfyUI HTTP client not ready.", "server_error")
    return comfy._http


async def _get_json(base: str, path: str, timeout: float = 90.0) -> Any:
    resp = await _client().get(f"{base}{path}", timeout=timeout)
    resp.raise_for_status()
    return resp.json()


async def _post(base: str, path: str, json: Optional[dict] = None) -> httpx.Response:
    resp = await _client().post(f"{base}{path}", json=json)
    resp.raise_for_status()
    return resp


# --------------------------------------------------------------------------
# Detection (GET only — no side effects)
# --------------------------------------------------------------------------


def _iter_nodes(workflow: dict):
    """Yield each node's class_type across API-format and full-graph shapes."""
    if not isinstance(workflow, dict):
        return
    # API format: {"<id>": {"class_type": ..., "inputs": {...}}, ...}
    for value in workflow.values():
        if isinstance(value, dict) and "class_type" in value:
            yield value["class_type"], value.get("inputs", {})
    # Full-graph format: {"nodes": [{"type": ..., ...}], ...}
    for node in workflow.get("nodes", []) if isinstance(workflow.get("nodes"), list) else []:
        if isinstance(node, dict) and node.get("type"):
            yield node["type"], {}


async def installed_node_types(base: str) -> Set[str]:
    data = await _get_json(base, "/object_info")
    return set(data.keys()) if isinstance(data, dict) else set()


def missing_node_types(workflow: dict, installed: Set[str]) -> List[str]:
    wanted = {ct for ct, _ in _iter_nodes(workflow) if ct}
    return sorted(wanted - installed)


def workflow_models(workflow: dict) -> List[dict]:
    """Full-graph top-level `models` array: [{name, url, directory, hash}]."""
    models = workflow.get("models") if isinstance(workflow, dict) else None
    return [m for m in models if isinstance(m, dict) and m.get("url")] if isinstance(models, list) else []


async def installed_model_names(base: str) -> Set[str]:
    """All model filenames the instance already exposes (from object_info enums)."""
    data = await _get_json(base, "/object_info")
    names: Set[str] = set()
    if not isinstance(data, dict):
        return names
    for node in data.values():
        required = ((node or {}).get("input", {}) or {}).get("required", {}) or {}
        optional = ((node or {}).get("input", {}) or {}).get("optional", {}) or {}
        for spec in list(required.values()) + list(optional.values()):
            # A combo input is [ [choices...], {...} ]; model loaders list files.
            if isinstance(spec, list) and spec and isinstance(spec[0], list):
                for choice in spec[0]:
                    if isinstance(choice, str):
                        names.add(choice)
    return names


async def getmappings(base: str) -> Dict[str, Any]:
    try:
        return await _get_json(base, "/customnode/getmappings?mode=cache", timeout=60.0)
    except httpx.HTTPError as exc:
        logger.warning("getmappings failed on %s: %s", base, exc)
        return {}


def node_pack_id(mappings: Dict[str, Any], class_type: str) -> Optional[str]:
    """Find the installable pack id whose node list includes class_type.

    getmappings shape: { "<pack-ref>": [[node_class_names...], {meta}], ... }.
    """
    for pack_ref, value in mappings.items():
        try:
            node_list = value[0]
        except (TypeError, IndexError):
            continue
        if isinstance(node_list, list) and class_type in node_list:
            return pack_ref
    return None


# --------------------------------------------------------------------------
# Queue control (POST — installs; only from provision)
# --------------------------------------------------------------------------


async def _enqueue_node(base: str, pack_id: str) -> None:
    await _post(
        base,
        "/api/manager/queue/install",
        {
            "id": pack_id,
            "version": "latest",
            "selected_version": "latest",
            "channel": "default",
            "mode": "cache",
            "ui_id": uuid.uuid4().hex,
            "skip_post_install": False,
        },
    )


async def _enqueue_model(base: str, model: dict) -> None:
    url = model.get("url")
    filename = model.get("name") or url.rsplit("/", 1)[-1]
    await _post(
        base,
        "/api/manager/queue/install_model",
        {
            "ui_id": uuid.uuid4().hex,
            "name": model.get("name") or filename,
            "url": url,
            "filename": filename,
            "save_path": model.get("directory") or "default",
            "type": model.get("type") or "",
            "base": model.get("base") or "",
        },
    )


async def queue_status(base: str) -> dict:
    return await _get_json(base, "/api/manager/queue/status", timeout=20.0)


async def provision(
    base_url: str,
    workflow: dict,
    wait: bool = True,
    timeout: Optional[float] = None,
) -> dict:
    """Install missing custom nodes + download missing models for a workflow.

    Returns a report; when `wait`, polls the Manager queue until idle.
    """
    base = base_url.rstrip("/")
    timeout = timeout if timeout is not None else settings.comfy_provision_timeout

    try:
        installed = await installed_node_types(base)
        installed_models = await installed_model_names(base)
    except httpx.HTTPError as exc:
        raise openai_error(502, f"ComfyUI object_info failed ({base}): {exc}", "api_error")

    missing_nodes = missing_node_types(workflow, installed)
    all_models = workflow_models(workflow)
    missing_models = [
        m for m in all_models
        if (m.get("name") or m["url"].rsplit("/", 1)[-1]) not in installed_models
    ]

    enqueued_nodes: List[str] = []
    unresolved_nodes: List[str] = list(missing_nodes)
    if missing_nodes:
        mappings = await getmappings(base)
        unresolved_nodes = []
        for class_type in missing_nodes:
            pack = node_pack_id(mappings, class_type)
            if pack:
                try:
                    await _enqueue_node(base, pack)
                    enqueued_nodes.append(f"{class_type} -> {pack}")
                except httpx.HTTPError as exc:
                    logger.warning("enqueue node %s failed: %s", pack, exc)
                    unresolved_nodes.append(class_type)
            else:
                unresolved_nodes.append(class_type)

    enqueued_models: List[str] = []
    for model in missing_models:
        try:
            await _enqueue_model(base, model)
            enqueued_models.append(model.get("name") or model["url"])
        except httpx.HTTPError as exc:
            logger.warning("enqueue model failed: %s", exc)

    report: dict = {
        "instance_base_url": base,
        "missing_nodes": missing_nodes,
        "enqueued_nodes": enqueued_nodes,
        "unresolved_nodes": unresolved_nodes,
        "missing_models": [m.get("name") or m["url"] for m in missing_models],
        "enqueued_models": enqueued_models,
    }

    if not enqueued_nodes and not enqueued_models:
        report["status"] = "nothing_to_install"
        return report

    try:
        await _post(base, "/api/manager/queue/start")
    except httpx.HTTPError as exc:
        raise openai_error(502, f"ComfyUI queue start failed: {exc}", "api_error")

    if not wait:
        report["status"] = "installing"
        return report

    deadline = time.monotonic() + timeout
    last = {}
    while time.monotonic() < deadline:
        await asyncio.sleep(3.0)
        try:
            last = await queue_status(base)
        except httpx.HTTPError:
            continue
        if not last.get("is_processing"):
            break
    report["status"] = "done" if not last.get("is_processing") else "timeout"
    report["final_queue"] = last
    return report
