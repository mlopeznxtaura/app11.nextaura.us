#!/usr/bin/env python3
"""Check app9.nextaura.us API status."""
import requests
import json


def main():
    try:
        resp = requests.get('https://app9.nextaura.us/api/status', timeout=10)
        print(f'Status API: {resp.status_code}')
        if resp.status_code == 200:
            data = resp.json()
            print(f"Running: {data.get('running')}")
            print(f"Status: {data.get('status')}")
            print(f"Progress: {data.get('progress_pct')}%")
            print(f"Dataset: {data.get('dataset')}")
    except Exception as e:
        print(f'API check: {e}')


if __name__ == "__main__":
    main()
