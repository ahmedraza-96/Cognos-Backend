"""Integration tests for the documents endpoints (upload / list / delete)."""


def _token(client, email="d@e.com"):
    return client.post(
        "/auth/signup", json={"email": email, "password": "password123"}
    ).json()["access_token"]


def _auth(client, email="d@e.com"):
    return {"Authorization": f"Bearer {_token(client, email)}"}


def test_upload_returns_document_with_chunks(client):
    h = _auth(client)
    files = {"file": ("notes.txt", b"Paris is the capital of France.", "text/plain")}
    r = client.post("/documents", files=files, headers=h)
    assert r.status_code == 201
    body = r.json()
    assert body["filename"] == "notes.txt"
    assert body["chunk_count"] >= 1


def test_list_returns_only_own_documents(client):
    ha = _auth(client, "a@x.com")
    hb = _auth(client, "b@x.com")
    client.post(
        "/documents",
        files={"file": ("a.txt", b"alpha content", "text/plain")},
        headers=ha,
    )
    listed_b = client.get("/documents", headers=hb)
    assert listed_b.status_code == 200
    assert listed_b.json() == []


def test_upload_requires_auth(client):
    files = {"file": ("notes.txt", b"hi", "text/plain")}
    r = client.post("/documents", files=files)
    assert r.status_code in (401, 403)


def test_reject_unsupported_extension(client):
    h = _auth(client)
    files = {"file": ("evil.exe", b"MZ\x90\x00", "application/octet-stream")}
    r = client.post("/documents", files=files, headers=h)
    assert r.status_code == 400


def test_delete_removes_document(client):
    h = _auth(client)
    doc = client.post(
        "/documents",
        files={"file": ("notes.txt", b"some content here to chunk", "text/plain")},
        headers=h,
    ).json()
    d = client.delete(f"/documents/{doc['id']}", headers=h)
    assert d.status_code == 204
    assert client.get("/documents", headers=h).json() == []
