from __future__ import annotations

import random
from datetime import datetime, timezone

from locust import HttpUser, between, task


class FraudApiUser(HttpUser):
    wait_time = between(0.01, 0.15)

    def on_start(self) -> None:
        self.headers = {"x-api-key": "local-dev-key"}
        self.user_pool = [f"user_{i}" for i in range(1, 1000)]

    @task
    def score_transaction(self) -> None:
        user_id = random.choice(self.user_pool)
        amount = round(random.lognormvariate(3.0, 1.0), 2)

        payload = {
            "user_id": user_id,
            "amount": amount,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self.client.post("/predict", json=payload, headers=self.headers, name="/predict")
