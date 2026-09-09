#!/usr/bin/env python3
"""Start Muse-Glimmer-30B QLoRA SFT on app9.nextaura.us."""
import requests


def main():
    # Click the preset button first via API
    # The button "Muse-Glimmer-30B QLoRA SFT v2" should set the config
    config = {
        "config": "sample-10BT",  # Base config
        "target_gb": 0.5,         # Adjusted for QLoRA SFT
        "model_params": 30_000_000_000,  # 30B params
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
        print(f'Start API: {resp.status_code}')
        print(f'Response: {resp.text[:500]}')
    except Exception as e:
        print(f'API error: {e}')


if __name__ == "__main__":
    main()
