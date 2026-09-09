#!/usr/bin/env python3
"""Start unified browser agent server."""
import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()


async def main():
    """Initialize and start unified browser agent."""
    from browser_agent.config import BrowserConfig
    from browser_agent.integrator import UnifiedBrowserAgent, MCPUnifiedServer
    
    # Load configuration
    config = BrowserConfig.from_env()
    print(f"Default backend: {config.default_backend}")
    
    # Initialize unified agent
    agent = UnifiedBrowserAgent(config)
    await agent.initialize()
    
    # Get MCP server
    mcp_server = MCPUnifiedServer(agent)
    
    print("\nInitialized browser agents:")
    for backend, status in (await agent.get_status()).items():
        print(f"  - {backend}: {status}")
    
    print("\nMCP Tools Available:")
    for tool in mcp_server.get_tools_schema():
        print(f"  - {tool['name']}: {tool['description']}")
    
    print("\nServer ready. Waiting for MCP connections...")
    
    # Keep running
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
