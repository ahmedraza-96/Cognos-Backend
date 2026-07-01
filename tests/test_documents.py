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


def test_chunks_returned_match_count_and_ordered(client):
    h = _auth(client)
    # ~3000 chars forces multiple chunks (chunk_size=1000, overlap=150).
    content = ("Sentence about geography and travel. " * 90).encode()
    doc = client.post(
        "/documents",
        files={"file": ("geo.txt", content, "text/plain")},
        headers=h,
    ).json()
    assert doc["chunk_count"] > 1

    r = client.get(f"/documents/{doc['id']}/chunks", headers=h)
    assert r.status_code == 200
    chunks = r.json()
    # One entry per indexed chunk.
    assert len(chunks) == doc["chunk_count"]
    # Ordered by chunk index, contiguous from 0.
    indices = [c["chunk"] for c in chunks]
    assert indices == list(range(len(chunks)))
    # Shape + non-empty content.
    assert all({"id", "chunk", "content"} <= c.keys() for c in chunks)
    assert all(c["content"].strip() for c in chunks)


def test_chunks_cross_user_returns_404(client):
    ha = _auth(client, "a@x.com")
    hb = _auth(client, "b@x.com")
    doc = client.post(
        "/documents",
        files={"file": ("a.txt", b"alpha content to index", "text/plain")},
        headers=ha,
    ).json()
    r = client.get(f"/documents/{doc['id']}/chunks", headers=hb)
    assert r.status_code == 404


def test_chunks_unknown_id_returns_404(client):
    h = _auth(client)
    r = client.get("/documents/does-not-exist/chunks", headers=h)
    assert r.status_code == 404


def test_chunks_requires_auth(client):
    r = client.get("/documents/anything/chunks")
    assert r.status_code in (401, 403)


def test_file_returns_original_bytes_inline(client):
    h = _auth(client)
    raw = b"Paris is the capital of France.\nA second line of text."
    doc = client.post(
        "/documents",
        files={"file": ("notes.txt", raw, "text/plain")},
        headers=h,
    ).json()

    r = client.get(f"/documents/{doc['id']}/file", headers=h)
    assert r.status_code == 200
    assert r.content == raw
    assert r.headers["content-type"].startswith("text/plain")
    assert "inline" in r.headers.get("content-disposition", "")


def test_file_cross_user_returns_404(client):
    ha = _auth(client, "a@x.com")
    hb = _auth(client, "b@x.com")
    doc = client.post(
        "/documents",
        files={"file": ("a.txt", b"alpha content", "text/plain")},
        headers=ha,
    ).json()
    r = client.get(f"/documents/{doc['id']}/file", headers=hb)
    assert r.status_code == 404


def test_file_unknown_id_returns_404(client):
    h = _auth(client)
    r = client.get("/documents/does-not-exist/file", headers=h)
    assert r.status_code == 404


def test_file_requires_auth(client):
    r = client.get("/documents/anything/file")
    assert r.status_code in (401, 403)
