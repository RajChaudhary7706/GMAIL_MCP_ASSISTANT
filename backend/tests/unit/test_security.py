"""Unit tests for security module."""
import pytest
from datetime import datetime, timedelta, timezone
from app.core.security import (
    hash_password, verify_password,
    create_access_token, create_refresh_token, decode_token,
    generate_pkce_pair, TokenVault,
)

class TestPasswordHashing:
    def test_hash_not_plaintext(self):
        assert hash_password("secret") != "secret"

    def test_verify_correct(self):
        h = hash_password("correct")
        assert verify_password("correct", h) is True

    def test_reject_wrong(self):
        h = hash_password("correct")
        assert verify_password("wrong", h) is False

    def test_different_salts(self):
        assert hash_password("pw") != hash_password("pw")

class TestJWT:
    def test_create_decode(self):
        token = create_access_token({"sub": "u1", "email": "a@b.com"})
        payload = decode_token(token)
        assert payload["sub"] == "u1"
        assert payload["type"] == "access"

    def test_expired_rejected(self):
        token = create_access_token({"sub":"u1"},expires_delta=timedelta(seconds=-1))
        with pytest.raises(ValueError): decode_token(token)

    def test_refresh_type(self):
        token = create_refresh_token("u2")
        assert decode_token(token)["type"] == "refresh"

    def test_tampered_rejected(self):
        token = create_access_token({"sub":"u1"})
        with pytest.raises(ValueError): decode_token(token[:-5]+"XXXXX")

class TestPKCE:
    def test_pair_structure(self):
        pkce = generate_pkce_pair()
        assert "code_verifier" in pkce
        assert "code_challenge" in pkce
        assert pkce["code_challenge_method"] == "S256"

    def test_url_safe(self):
        import re
        pkce = generate_pkce_pair()
        assert re.match(r'^[A-Za-z0-9\-_]+$', pkce["code_verifier"])

    def test_uniqueness(self):
        from app.core.security import generate_state_token
        states = {generate_state_token() for _ in range(100)}
        assert len(states) == 100

class TestTokenVault:
    def test_roundtrip(self):
        token = "ya29.google_access_token"
        assert TokenVault.decrypt_token(TokenVault.encrypt_token(token)) == token

    def test_encrypted_not_plaintext(self):
        token = "sensitive_token"
        assert token not in TokenVault.encrypt_token(token)

    def test_expiry_check(self):
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        assert TokenVault.is_token_expired(past) is True
        assert TokenVault.is_token_expired(future) is False