"""Tests for ingestion.erpnext_client. No network calls — httpx is mocked via respx."""

import httpx
import pytest
import respx

from ingestion.erpnext_client import (
    ERPNextAuthError,
    ERPNextClient,
    ERPNextNotFoundError,
)

BASE_URL = "https://erp.example.com"


@pytest.fixture
def client() -> ERPNextClient:
    return ERPNextClient(base_url=BASE_URL, api_key="key123", api_secret="secret456")


@pytest.mark.asyncio
@respx.mock
async def test_get_list_returns_data(client: ERPNextClient) -> None:
    route = respx.get(f"{BASE_URL}/api/resource/Purchase Order").mock(
        return_value=httpx.Response(200, json={"data": [{"name": "PUR-ORD-2026-00001"}]})
    )

    result = await client.get_list(
        "Purchase Order", filters=[["status", "=", "Completed"]], fields=["name"], limit=5
    )

    assert result == [{"name": "PUR-ORD-2026-00001"}]
    assert route.called
    request = route.calls.last.request
    assert request.headers["Authorization"] == "token key123:secret456"
    assert request.url.params["limit_page_length"] == "5"


@pytest.mark.asyncio
@respx.mock
async def test_get_doc_returns_data(client: ERPNextClient) -> None:
    respx.get(f"{BASE_URL}/api/resource/Purchase Order/PUR-ORD-2026-00001").mock(
        return_value=httpx.Response(200, json={"data": {"name": "PUR-ORD-2026-00001"}})
    )

    result = await client.get_doc("Purchase Order", "PUR-ORD-2026-00001")

    assert result == {"name": "PUR-ORD-2026-00001"}


@pytest.mark.asyncio
@respx.mock
async def test_get_file_content_returns_bytes(client: ERPNextClient) -> None:
    respx.get(f"{BASE_URL}/private/files/contract.pdf").mock(
        return_value=httpx.Response(200, content=b"%PDF-1.4 fake content")
    )

    result = await client.get_file_content("/private/files/contract.pdf")

    assert result == b"%PDF-1.4 fake content"


@pytest.mark.asyncio
@respx.mock
async def test_get_attached_files_returns_data(client: ERPNextClient) -> None:
    route = respx.get(f"{BASE_URL}/api/resource/File").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "name": "FILE-001",
                        "file_url": "/private/files/contract.pdf",
                        "file_name": "contract.pdf",
                    }
                ]
            },
        )
    )

    result = await client.get_attached_files("Contract", "CON-001")

    assert result == [
        {
            "name": "FILE-001",
            "file_url": "/private/files/contract.pdf",
            "file_name": "contract.pdf",
        }
    ]
    assert route.called
    request = route.calls.last.request
    filters = request.url.params["filters"]
    assert '"attached_to_doctype", "=", "Contract"' in filters or "attached_to_doctype" in filters
    assert "CON-001" in filters


@pytest.mark.asyncio
@respx.mock
async def test_get_doc_raises_not_found_on_404(client: ERPNextClient) -> None:
    respx.get(f"{BASE_URL}/api/resource/Contract/CON-9999").mock(
        return_value=httpx.Response(404, json={"exc_type": "DoesNotExistError"})
    )

    with pytest.raises(ERPNextNotFoundError):
        await client.get_doc("Contract", "CON-9999")


@pytest.mark.asyncio
@respx.mock
async def test_get_list_raises_auth_error_on_401(client: ERPNextClient) -> None:
    respx.get(f"{BASE_URL}/api/resource/Purchase Order").mock(
        return_value=httpx.Response(401, text="Unauthorized")
    )

    with pytest.raises(ERPNextAuthError):
        await client.get_list("Purchase Order")


@pytest.mark.asyncio
@respx.mock
async def test_get_list_raises_auth_error_on_403(client: ERPNextClient) -> None:
    respx.get(f"{BASE_URL}/api/resource/Purchase Order").mock(
        return_value=httpx.Response(403, text="Forbidden")
    )

    with pytest.raises(ERPNextAuthError):
        await client.get_list("Purchase Order")


@pytest.mark.asyncio
@respx.mock
async def test_get_list_raises_on_server_error(client: ERPNextClient) -> None:
    respx.get(f"{BASE_URL}/api/resource/Purchase Order").mock(
        return_value=httpx.Response(500, text="Internal Server Error")
    )

    with pytest.raises(httpx.HTTPStatusError):
        await client.get_list("Purchase Order")


@pytest.mark.asyncio
async def test_constructor_reads_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ERPNEXT_URL", BASE_URL)
    monkeypatch.setenv("ERPNEXT_API_KEY", "envkey")
    monkeypatch.setenv("ERPNEXT_API_SECRET", "envsecret")

    client = ERPNextClient()
    try:
        assert client._client.headers["Authorization"] == "token envkey:envsecret"
    finally:
        await client.aclose()
