#!/usr/bin/env python3
"""Start MCP browser agent service."""
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def main():
    """Start MCP browser server."""
    from browser_agent import BrowserAgent, MCPBrowserServer
    from playwright.async_api import async_playwright
    
    # Initialize browser
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=os.getenv("BROWSER_HEADLESS", "true") == "true"
        )
        context = await browser.new_context(
            viewport={"width": 1280, "height": 720}
        )
        page = await context.new_page()
        
        # Initialize model client (placeholder - replace with actual LLM client)
        model_client = type("ModelClient", (), {
            "generate": lambda self, prompt: asyncio.sleep(0.1) or "[]"
        })()
        
        # Create agent and MCP server
        agent = BrowserAgent(page, model_client)
        server = MCPBrowserServer(agent, port=int(os.getenv("BROWSER_PORT", "4000")))
        
        # Run browser service
        print("Browser agent ready")
        print(f"Tools: {[t['name'] for t in server.get_tools_schema()]}")
        
        # Keep running
        await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
