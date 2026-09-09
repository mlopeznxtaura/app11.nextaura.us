#!/usr/bin/env python3
"""Start Muse-Glimmer-30B QLoRA SFT on app9.nextaura.us."""
import requests
import sys


sys.stdout.reconfigure(encoding='utf-8')


def main():
    # Get current state
    try:
        resp = requests.get('https://app9.nextaura.us/api/status', timeout=10)
        data = resp.json()
        print(f"Current status: {data.get('status')}")
        print(f"Running: {data.get('running')}")
    except Exception as e:
        print(f"Error checking status: {e}")
        return
    
    # Try to start training with QLoRA SFT config
    config = {
        "config": "sample-10BT",
        "target_gb": 0.5,
        "model_params": 30_000_000_000,
        "learning_rate": 3e-4,
        "batch_size": 64,
        "data_source": "fable-traces",
        "total_train_steps": 1000,
        "run_goal_title": "Muse-Glimmer-30B QLoRA SFT v2",
    }

    try:
        resp = requests.post(
            'https://app9.nextaura.us/api/start',
            json=config,
            timeout=30
        )
        print(f"\nStart API: {resp.status_code}")
        print(f"Response: {resp.text[:500]}")
    except Exception as e:
        print(f"\nAPI error: {e}")


if __name__ == "__main__":
    main()
