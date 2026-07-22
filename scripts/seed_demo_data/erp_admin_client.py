"""Sync Frappe REST API wrapper with write operations, for demo-data seeding only.

`ingestion.erpnext_client.ERPNextClient` is deliberately read-only and is imported by the
production FastAPI app and webhook handler — bolting create/submit/cancel/delete/upload onto
it would blur that boundary. This mirrors its constructor/header/exception conventions but
adds the write operations `scripts/seed_demo_data` needs, following the same sync-httpx
patterns already used in `tests/test_integration.py`'s ERP helper functions.
"""

from __future__ import annotations

import json
import os
from types import TracebackType
from typing import Any

import httpx


class ERPAdminError(Exception):
    """Base exception for all ERPAdminClient errors."""


class ERPAdminAuthError(ERPAdminError):
    """Raised when ERPNext rejects the API credentials (401/403)."""


class ERPAdminClient:
    """Thin sync wrapper around the Frappe REST API, with write operations.

    Credentials and base URL default to the `ERPNEXT_URL` / `ERPNEXT_API_KEY` /
    `ERPNEXT_API_SECRET` environment variables but can be overridden.
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
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"token {api_key}:{api_secret}"},
            timeout=timeout,
        )

    def __enter__(self) -> ERPAdminClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def get_list(
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

        response = self._client.get(f"/api/resource/{doctype}", params=params)
        self._raise_for_status(response, doctype=doctype)
        return response.json()["data"]

    def get_doc(self, doctype: str, name: str) -> dict[str, Any]:
        """Fetch a single document by name, including any child tables."""
        response = self._client.get(f"/api/resource/{doctype}/{name}")
        self._raise_for_status(response, doctype=doctype, name=name)
        return response.json()["data"]

    def get_attached_files(self, doctype: str, docname: str) -> list[dict[str, Any]]:
        """List `File` records attached to a given document."""
        return self.get_list(
            "File",
            filters=[["attached_to_doctype", "=", doctype], ["attached_to_name", "=", docname]],
            fields=["name", "file_url", "file_name"],
            limit=0,
        )

    def create_doc(self, doctype: str, data: dict[str, Any]) -> dict[str, Any]:
        """Create a new document. Returns the created doc (including its assigned name)."""
        response = self._client.post(
            f"/api/resource/{doctype}", json={"doctype": doctype, **data}
        )
        self._raise_for_status(response, doctype=doctype)
        return response.json()["data"]

    def update_doc(self, doctype: str, name: str, data: dict[str, Any]) -> dict[str, Any]:
        """Update an existing document's fields in place."""
        response = self._client.put(f"/api/resource/{doctype}/{name}", json=data)
        self._raise_for_status(response, doctype=doctype, name=name)
        return response.json()["data"]

    def submit_doc(self, created_doc: dict[str, Any]) -> None:
        """Submit a document (docstatus 0 -> 1).

        Passes the full created doc, not just its name, to `frappe.client.submit` so
        Frappe's optimistic-locking timestamp check passes.
        """
        response = self._client.post(
            "/api/method/frappe.client.submit",
            data={"doc": json.dumps(created_doc)},
        )
        self._raise_for_status(response, doctype=created_doc.get("doctype"))

    def cancel_doc(self, doctype: str, name: str) -> None:
        """Cancel a submitted document (docstatus 1 -> 2)."""
        response = self._client.post(
            "/api/method/frappe.client.cancel",
            data={"doctype": doctype, "name": name},
        )
        self._raise_for_status(response, doctype=doctype, name=name)

    def delete_doc(self, doctype: str, name: str) -> None:
        """Delete a document outright. Only valid for docstatus 0 or 2."""
        response = self._client.delete(f"/api/resource/{doctype}/{name}")
        self._raise_for_status(response, doctype=doctype, name=name)

    def upload_file(
        self,
        doctype: str,
        docname: str,
        filename: str,
        content: bytes,
        is_private: bool = True,
    ) -> dict[str, Any]:
        """Attach a file to an existing document via `/api/method/upload_file`."""
        response = self._client.post(
            "/api/method/upload_file",
            files={"file": (filename, content, "application/pdf")},
            data={
                "doctype": doctype,
                "docname": docname,
                "is_private": 1 if is_private else 0,
            },
        )
        self._raise_for_status(response, doctype=doctype, name=docname)
        return response.json()["message"]

    def _raise_for_status(
        self,
        response: httpx.Response,
        *,
        doctype: str | None = None,
        name: str | None = None,
    ) -> None:
        if response.status_code in (401, 403):
            raise ERPAdminAuthError(
                f"ERPNext rejected credentials ({response.status_code}): {response.text}"
            )
        if response.status_code >= 400:
            target = f"{doctype}/{name}" if name else doctype
            raise ERPAdminError(
                f"ERPNext request failed for {target} ({response.status_code}): {response.text}"
            )
