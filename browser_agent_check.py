#!/usr/bin/env python3
"""Test unified browser agent."""
import asyncio
import os
from dataclasses import asdict


async def main():
    from browser_agent.config import BrowserConfig
    from browser_agent.integrator import UnifiedBrowserAgent

    # Load config from env
    os.environ["BROWSER_DEFAULT"] = "ondevice"
    os.environ["ONDEVICE_ENABLED"] = "true"

    config = BrowserConfig.from_env()
    print(f"Default backend: {config.default_backend}")

    agent = UnifiedBrowserAgent(asdict(config))
    await agent.initialize()

    print("\nBackend status:")
    for backend, status in (await agent.get_status()).items():
        print(f"  {backend}: {status}")

    # Test on-device backend
    result = await agent.execute_task("navigate to https://example.com", backend="ondevice")
    print(f"\nTask result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
