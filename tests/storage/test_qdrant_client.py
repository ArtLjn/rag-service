"""collection_manager 单测：mock QdrantClient，验证 collection 创建/删除/存在性。"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.core.exceptions import CollectionNotFound
from app.storage import collection_manager


class FakeQdrant:
    def __init__(self) -> None:
        self.collections: dict[str, dict] = {}
        self.create_calls: list[dict] = []
        self.delete_calls: list[str] = []

    def collection_exists(self, name: str) -> bool:
        return name in self.collections

    def create_collection(self, **kwargs):
        self.create_calls.append(kwargs)
        self.collections[kwargs["collection_name"]] = {"vectors_config": kwargs.get("vectors_config")}

    def delete_collection(self, collection_name: str) -> None:
        self.delete_calls.append(collection_name)
        self.collections.pop(collection_name, None)

    def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in self.collections])

    def get_collection(self, name: str):
        return SimpleNamespace(vectors_count=10, points_count=5, status="GREEN")


def test_create_collection_passes_dense_and_sparse_vectors() -> None:
    fake = FakeQdrant()
    with patch.object(collection_manager, "get_client", return_value=fake):
        result = collection_manager.create_collection("c1", vector_dim=768)
    assert result["action"] == "created"
    call = fake.create_calls[0]
    assert "dense" in call["vectors_config"]
    assert "text-sparse" in call["sparse_vectors_config"]


def test_create_collection_idempotent_when_exists() -> None:
    fake = FakeQdrant()
    fake.collections["c1"] = {}
    with patch.object(collection_manager, "get_client", return_value=fake):
        result = collection_manager.create_collection("c1", vector_dim=768)
    assert result["action"] == "noop"


def test_delete_collection_raises_when_missing() -> None:
    fake = FakeQdrant()
    with patch.object(collection_manager, "get_client", return_value=fake):
        with pytest.raises(CollectionNotFound):
            collection_manager.delete_collection("missing")


def test_list_collections_returns_metadata() -> None:
    fake = FakeQdrant()
    fake.collections["a"] = {}
    fake.collections["b"] = {}
    with patch.object(collection_manager, "get_client", return_value=fake):
        listing = collection_manager.list_collections()
    names = [item["name"] for item in listing]
    assert sorted(names) == ["a", "b"]


def test_ensure_collection_or_raise_raises_when_missing() -> None:
    fake = FakeQdrant()
    with patch.object(collection_manager, "get_client", return_value=fake):
        with pytest.raises(CollectionNotFound):
            collection_manager.ensure_collection_or_raise("missing")
