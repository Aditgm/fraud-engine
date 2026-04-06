from __future__ import annotations

import json
from typing import Any

from redis.asyncio import Redis


class InMemoryFeatureStore:
    def __init__(self, window_size: int = 5) -> None:
        self.window_size = window_size
        self._store: dict[str, list[dict[str, Any]]] = {}

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def get_recent_transactions(self, user_id: str) -> list[dict[str, Any]]:
        return list(self._store.get(user_id, []))

    async def append_transaction(
        self, user_id: str, amount: float, timestamp: str
    ) -> None:
        history = self._store.get(user_id, [])
        history.insert(0, {"amount": amount, "timestamp": timestamp})
        self._store[user_id] = history[: self.window_size]

    async def ping(self) -> bool:
        return True


class RedisFeatureStore:
    def __init__(
        self, url: str, window_size: int = 5, key_prefix: str = "fraud_history"
    ) -> None:
        self.url = url
        self.window_size = window_size
        self.key_prefix = key_prefix
        self._redis: Redis | None = None

    async def connect(self) -> None:
        self._redis = Redis.from_url(self.url, decode_responses=True)

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.close()

    @property
    def client(self) -> Redis:
        if self._redis is None:
            raise RuntimeError("Redis client is not connected")
        return self._redis

    def _key(self, user_id: str) -> str:
        return f"{self.key_prefix}:{user_id}"

    async def get_recent_transactions(self, user_id: str) -> list[dict[str, Any]]:
        raw_items = await self.client.lrange(
            self._key(user_id), 0, self.window_size - 1
        )
        output: list[dict[str, Any]] = []
        for item in raw_items:
            try:
                parsed = json.loads(item)
                if isinstance(parsed, dict):
                    output.append(parsed)
            except json.JSONDecodeError:
                continue
        return output

    async def append_transaction(
        self, user_id: str, amount: float, timestamp: str
    ) -> None:
        payload = json.dumps({"amount": amount, "timestamp": timestamp})
        key = self._key(user_id)

        pipe = self.client.pipeline(transaction=True)
        await pipe.lpush(key, payload)
        await pipe.ltrim(key, 0, self.window_size - 1)
        await pipe.execute()

    async def ping(self) -> bool:
        return bool(await self.client.ping())
