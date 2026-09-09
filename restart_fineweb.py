#!/usr/bin/env python3
"""Start fineweb training without local JSONL."""
import requests
import json


def main():
    # Try with mix_local_jsonl = False
    config = {
        'data_source': 'fineweb',
        'mix_local_jsonl': False,
        'target_gb': 0.08,
        'model_params': 12600000,
        'learning_rate': 3e-4,
        'batch_size': 1024,
        'total_train_steps': 0,
        'n_layer': 8,
        'n_head': 8,
        'run_goal_title': 'Fineweb smoke',
    }

    print("Starting fineweb training...")
    r = requests.post('https://app9.nextaura.us/api/start', json=config, timeout=30)
    print(f"API Status: {r.status_code}")
    
    if r.status_code == 200:
        data = r.json()
        print(f"Status: {data.get('status')}")
        print(f"Data source: {data.get('data_source')}")
        
        # Check after a few seconds
        import time
        time.sleep(3)
        
        r2 = requests.get('https://app9.nextaura.us/api/status', timeout=5)
        data2 = r2.json()
        print(f"\nAfter 3s: status={data2.get('status')} running={data2.get('running')}")
    else:
        print(f"Response: {r.text[:500]}")


if __name__ == "__main__":
    main()
