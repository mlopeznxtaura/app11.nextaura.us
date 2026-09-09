#!/usr/bin/env python3
"""Reset session on app9.nextaura.us."""
import requests


def main():
    # Try to reset session
    try:
        resp = requests.post(
            'https://app9.nextaura.us/api/reset',
            timeout=30
        )
        print(f"Reset API: {resp.status_code}")
        print(f"Response: {resp.text[:500]}")
    except Exception as e:
        print(f"Reset error: {e}")


if __name__ == "__main__":
    main()
