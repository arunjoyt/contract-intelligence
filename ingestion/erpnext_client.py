"""Async Frappe REST API wrapper for ERPNext.

All ingestion code talks to ERPNext exclusively through this client. It never embeds
business logic about specific doctypes — that lives in `document_parser.py`.
"""

from __future__ import annotations

import json
import os
from types import TracebackType
from typing import Any
from urllib.parse import urlparse

import httpx


class ERPNextError(Exception):
    """Base exception for all ERPNext client errors."""


class ERPNextAuthError(ERPNextError):
    """Raised when ERPNext rejects the API credentials (401/403)."""


class ERPNextNotFoundError(ERPNextError):
    """Raised when a requested document or resource does not exist (404)."""


class ERPNextInvalidFileURLError(ERPNextError):
    """Raised when a `File` record's `file_url` is not a relative site path.

    ERPNext's `File` doctype supports "attach by URL", so a low-privileged
    user (e.g. a `Purchase User` attaching a file to a Purchase Order or
    Contract) can set `file_url` to an arbitrary absolute URL. Since the
    underlying httpx client carries a default `Authorization` header with the
    service account's ERPNext API key/secret, honoring an absolute
    `file_url` verbatim would leak those credentials to an attacker-controlled
    host (and enables SSRF against internal-only services). See issue #63.
    """


class ERPNextClient:
    """Thin async wrapper around the Frappe REST API.

    Credentials and base URL default to the `ERPNEXT_URL` / `ERPNEXT_API_KEY` /
    `ERPNEXT_API_SECRET` environment variables but can be overridden (e.g. in tests).
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        base_url = base_url or os.environ["ERPNEXT_URL"]
        api_key = api_key or os.environ["ERPNEXT_API_KEY"]
        api_secret = api_secret or os.environ["ERPNEXT_API_SECRET"]
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"token {api_key}:{api_secret}"},
            timeout=timeout,
        )

    async def __aenter__(self) -> ERPNextClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def get_list(
        self,
        doctype: str,
        filters: list[list[Any]] | dict[str, Any] | None = None,
        fields: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Fetch documents for `doctype` matching `filters`, returning only `fields`.

        `limit` maps to Frappe's `limit_page_length`; pass `0` for no limit.
        """
        params: dict[str, Any] = {}
        if filters is not None:
            params["filters"] = json.dumps(filters)
        if fields is not None:
            params["fields"] = json.dumps(fields)
        params["limit_page_length"] = limit

        response = await self._client.get(f"/api/resource/{doctype}", params=params)
        self._raise_for_status(response, doctype=doctype)
        return response.json()["data"]

    async def get_doc(self, doctype: str, name: str) -> dict[str, Any]:
        """Fetch a single document by name, including any child tables."""
        response = await self._client.get(f"/api/resource/{doctype}/{name}")
        self._raise_for_status(response, doctype=doctype, name=name)
        return response.json()["data"]

    async def get_attached_files(self, doctype: str, docname: str) -> list[dict[str, Any]]:
        """List `File` records attached to a given document.

        Returns each file's `name`, `file_url`, and `file_name`; callers filter by
        extension and fetch bytes via `get_file_content(file_url)`.
        """
        return await self.get_list(
            "File",
            filters=[["attached_to_doctype", "=", doctype], ["attached_to_name", "=", docname]],
            fields=["name", "file_url", "file_name"],
            limit=0,
        )

    async def get_file_content(self, file_url: str) -> bytes:
        """Download raw bytes for a Frappe `File` record's `file_url`.

        `file_url` must be a path relative to the site (e.g. `/private/files/x.pdf`),
        resolved against the client's base URL. Rejects any `file_url` carrying its
        own scheme or host -- see `ERPNextInvalidFileURLError`.
        """
        parsed = urlparse(file_url)
        if parsed.scheme or parsed.netloc:
            raise ERPNextInvalidFileURLError(
                f"file_url must be a relative path, got {file_url!r}"
            )
        response = await self._client.get(file_url)
        self._raise_for_status(response)
        return response.content

    def _raise_for_status(
        self,
        response: httpx.Response,
        *,
        doctype: str | None = None,
        name: str | None = None,
    ) -> None:
        if response.status_code in (401, 403):
            raise ERPNextAuthError(
                f"ERPNext rejected credentials ({response.status_code}): {response.text}"
            )
        if response.status_code == 404:
            target = f"{doctype}/{name}" if name else doctype
            raise ERPNextNotFoundError(f"{target} not found")
        response.raise_for_status()
