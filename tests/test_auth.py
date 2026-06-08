"""Integration tests for the auth endpoints (signup / login / me)."""


def _signup(client, email="a@b.com", password="password123"):
    return client.post("/auth/signup", json={"email": email, "password": password})


def test_signup_returns_token(client):
    r = _signup(client)
    assert r.status_code == 201
    assert "access_token" in r.json()


def test_signup_duplicate_email_rejected(client):
    _signup(client)
    r = _signup(client)
    assert r.status_code == 400


def test_login_success(client):
    _signup(client)
    r = client.post("/auth/login", json={"email": "a@b.com", "password": "password123"})
    assert r.status_code == 200
    assert "access_token" in r.json()


def test_login_wrong_password_rejected(client):
    _signup(client)
    r = client.post("/auth/login", json={"email": "a@b.com", "password": "nope"})
    assert r.status_code == 401


def test_me_returns_current_user(client):
    token = _signup(client).json()["access_token"]
    r = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["email"] == "a@b.com"


def test_me_without_token_rejected(client):
    r = client.get("/auth/me")
    assert r.status_code in (401, 403)


def test_me_with_garbage_token_rejected(client):
    r = client.get("/auth/me", headers={"Authorization": "Bearer garbage.token.here"})
    assert r.status_code == 401
