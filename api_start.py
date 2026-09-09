#!/usr/bin/env python3
"""Start training via API on app9.nextaura.us."""
import requests
import json


def main():
    # Current config from UI (muse-sft source has issues)
    config = {
        "config": "sample-10BT",
        "target_gb": 0.08,
        "model_params": 12_600_000,
        "learning_rate": 3e-4,
        "batch_size": 1024,
        "data_source": "fineweb",
        "total_train_steps": 0,
        "run_goal_title": "Smoke test",
    }

    try:
        resp = requests.post(
            'https://app9.nextaura.us/api/start',
            json=config,
            timeout=30
        )
        print(f"API status: {resp.status_code}")
        print(f"Response: {resp.text}")
    except Exception as e:
        print(f"API error: {e}")


if __name__ == "__main__":
    main()
