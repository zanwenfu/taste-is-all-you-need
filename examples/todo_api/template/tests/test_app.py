from app import _reset_state, create_app


def setup_function(_):
    _reset_state()


def test_list_is_empty_initially():
    client = create_app().test_client()
    assert client.get("/items").json == []


def test_create_and_list():
    client = create_app().test_client()
    r = client.post("/items", json={"title": "buy milk"})
    assert r.status_code == 201
    assert r.json["title"] == "buy milk"
    assert r.json["done"] is False
    items = client.get("/items").json
    assert len(items) == 1
    assert items[0]["title"] == "buy milk"


def test_create_rejects_missing_title():
    client = create_app().test_client()
    r = client.post("/items", json={})
    assert r.status_code == 400


def test_create_rejects_blank_title():
    client = create_app().test_client()
    r = client.post("/items", json={"title": "   "})
    assert r.status_code == 400


def test_delete_removes_item():
    client = create_app().test_client()
    r = client.post("/items", json={"title": "task"})
    item_id = r.json["id"]
    d = client.delete(f"/items/{item_id}")
    assert d.status_code == 204
    assert client.get("/items").json == []


def test_delete_missing_item_404():
    client = create_app().test_client()
    assert client.delete("/items/999").status_code == 404
