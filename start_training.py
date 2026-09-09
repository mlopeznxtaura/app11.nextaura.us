#!/usr/bin/env python3
"""Start training on app9.nextaura.us via API."""
import requests
import json


def main():
    # Start training with minimal config
    config = {
        "config": "sample-10BT",
        "target_gb": 0.1,
        "model_params": 12_000_000,
        "learning_rate": 3e-4,
        "batch_size": 1024,
        "data_source": "fineweb",
        "total_train_steps": 0,
        "run_goal_title": "Test training session",
    }

    try:
        resp = requests.post(
            'https://app9.nextaura.us/api/start',
            json=config,
            timeout=30
        )
        print(f'Start API: {resp.status_code}')
        if resp.status_code == 200:
            data = resp.json()
            print(f"Training started: {data.get('status')}")
            print(f"Model params: {data.get('model_params')}")
        else:
            print(f"Response: {resp.text[:200]}")
    except Exception as e:
        print(f'API error: {e}')


if __name__ == "__main__":
    main()
