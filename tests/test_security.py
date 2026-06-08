"""Tests for auth password hashing and JWT helpers (pure functions)."""
import pytest

from app.auth import security


def test_hash_password_is_not_plaintext():
    hashed = security.hash_password("s3cret-pw")
    assert hashed != "s3cret-pw"
    assert hashed.startswith("$2")  # bcrypt prefix


def test_verify_password_accepts_correct_and_rejects_wrong():
    hashed = security.hash_password("s3cret-pw")
    assert security.verify_password("s3cret-pw", hashed) is True
    assert security.verify_password("wrong-pw", hashed) is False


def test_access_token_roundtrip_returns_subject():
    token = security.create_access_token(subject="user-123")
    payload = security.decode_access_token(token)
    assert payload["sub"] == "user-123"


def test_decode_rejects_tampered_token():
    token = security.create_access_token(subject="user-123")
    tampered = token[:-2] + ("aa" if not token.endswith("aa") else "bb")
    with pytest.raises(security.TokenError):
        security.decode_access_token(tampered)


def test_decode_rejects_expired_token():
    token = security.create_access_token(subject="user-123", expires_minutes=-1)
    with pytest.raises(security.TokenError):
        security.decode_access_token(token)
