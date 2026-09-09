#!/usr/bin/env python3
"""Check training status."""
import requests
import time


def main():
    for i in range(5):
        r = requests.get('https://app9.nextaura.us/api/status', timeout=5)
        d = r.json()
        print(f"{i}: status={d.get('status')} running={d.get('running')} progress={d.get('progress_pct')}%")
        time.sleep(2)


if __name__ == "__main__":
    main()
