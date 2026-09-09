#!/usr/bin/env python3
"""Fully start fineweb training."""
import requests
import json
import time


def main():
    config = {
        'data_source': 'fineweb',
        'mix_local_jsonl': False,
        'mix_local_corpora': False,
        'target_gb': 0.08,
        'model_params': 12600000,
        'learning_rate': 3e-4,
        'batch_size': 1024,
        'total_train_steps': 0,
        'n_layer': 8,
        'n_head': 8,
        'n_embd': 256,
        'block_size': 256,
        'run_goal_title': 'Fineweb smoke',
    }

    print("Starting fineweb training...")
    r = requests.post('https://app9.nextaura.us/api/start', json=config, timeout=30)
    print(f"API Status: {r.status_code}")
    data = r.json()
    print(f"Initial status: {data.get('status')}")

    # Poll for 15 seconds
    for i in range(5):
        time.sleep(3)
        r2 = requests.get('https://app9.nextaura.us/api/status', timeout=5)
        data2 = r2.json()
        print(f"{i+1}: status={data2.get('status')} running={data2.get('running')} bytes={data2.get('size_mb')} MB progress={data2.get('progress_pct')}%")


if __name__ == "__main__":
    main()
