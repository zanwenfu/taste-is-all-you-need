"""Minimal TODO API used as the starting point for the real-model demo."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from flask import Flask, request


@dataclass
class Item:
    id: int
    title: str
    done: bool = False


_STORE: dict[int, Item] = {}
_NEXT_ID = 1


def _reset_state() -> None:
    """Testing hook: wipe the in-memory store between tests."""
    global _NEXT_ID
    _STORE.clear()
    _NEXT_ID = 1


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/items")
    def list_items():
        return [asdict(x) for x in _STORE.values()]

    @app.post("/items")
    def create_item():
        global _NEXT_ID
        body = request.get_json(silent=True) or {}
        title = body.get("title")
        if not isinstance(title, str) or not title.strip():
            return {"error": "title required"}, 400
        item = Item(id=_NEXT_ID, title=title.strip())
        _STORE[_NEXT_ID] = item
        _NEXT_ID += 1
        return asdict(item), 201

    @app.delete("/items/<int:item_id>")
    def delete_item(item_id: int):
        if item_id not in _STORE:
            return {"error": "not found"}, 404
        del _STORE[item_id]
        return "", 204

    return app
