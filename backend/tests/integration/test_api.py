"""Integration tests for FastAPI endpoints."""
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.core.security import create_access_token

@pytest.fixture(scope="module")
def token():
    return create_access_token({"sub":"test-id","email":"test@example.com"})

@pytest.fixture(scope="module")
def auth_headers(token):
    return {"Authorization": f"Bearer {token}"}

@pytest_asyncio.fixture(scope="module")
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

class TestHealth:
    @pytest.mark.asyncio
    async def test_health(self, client):
        r = await client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_ready(self, client):
        r = await client.get("/health/ready")
        assert r.status_code in (200, 503)
        assert "checks" in r.json()

class TestAuth:
    @pytest.mark.asyncio
    async def test_google_url(self, client):
        r = await client.get("/api/v1/auth/google/url")
        assert r.status_code == 200
        data = r.json()
        assert "auth_url" in data
        assert "code_challenge" in data["auth_url"]  # PKCE
        assert "state" in data

    @pytest.mark.asyncio
    async def test_logout(self, client):
        r = await client.post("/api/v1/auth/logout")
        assert r.status_code == 200

class TestEmailAuth:
    @pytest.mark.asyncio
    async def test_requires_auth(self, client):
        r = await client.get("/api/v1/email/")
        assert r.status_code == 403

    @pytest.mark.asyncio
    async def test_invalid_token(self, client):
        r = await client.get("/api/v1/email/",
            headers={"Authorization":"Bearer invalid.token"})
        assert r.status_code == 401

class TestAgent:
    @pytest.mark.asyncio
    async def test_chat_no_auth(self, client):
        r = await client.post("/api/v1/agent/chat",json={"message":"hi"})
        assert r.status_code == 403