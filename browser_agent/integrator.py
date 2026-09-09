"""Unified browser agent integrator combining fouwser, ondevice, and lmnr."""
import asyncio
from typing import Any, Optional


class UnifiedBrowserAgent:
    """
    Unified agent layer that routes tasks to appropriate browser backend.
    
    - fouwser: MCP server access for external agents
    - ondevice: On-device private browsing (no API keys)
    - lmnr: Vision-based reasoning for complex tasks
    """
    
    def __init__(self, config: dict[str, Any]):
        """Initialize unified agent with backend configuration."""
        self.config = config
        self.active_backend = config.get("default_backend", "lmnr")
        self.backends = {
            "fouwser": None,
            "ondevice": None,
            "lmnr": None,
        }
    
    async def initialize(self):
        """Initialize all configured backends."""
        # Initialize fouwser MCP server
        if self.config.get("fouwser", {}).get("enabled", False):
            self.backends["fouwser"] = await self._init_fouwser()
        
        # Initialize on-device agent
        if self.config.get("ondevice", {}).get("enabled", False):
            self.backends["ondevice"] = await self._init_ondevice()
        
        # Initialize lmnr agent
        if self.config.get("lmnr", {}).get("enabled", True):
            self.backends["lmnr"] = await self._init_lmnr()
    
    async def _init_fouwser(self):
        """Initialize fouwser MCP server backend."""
        # Integrate with fouwser MCP server
        from browser_fouwser.mcp import FouwserMCPServer
        return FouwserMCPServer(self.config.get("fouwser", {}))
    
    async def _init_ondevice(self):
        """Initialize on-device browser agent."""
        # Use on-device agent with WebLLM/WebGPU
        from browser_ondevice.agent import OnDeviceAgent
        return OnDeviceAgent(self.config.get("ondevice", {}))
    
    async def _init_lmnr(self):
        """Initialize lmnr vision-based agent."""
        # Use lmnr index for complex vision-based tasks
        from browser_lmnr.agent import IndexAgent
        return IndexAgent(self.config.get("lmnr", {}))
    
    async def navigate(self, url: str, backend: Optional[str] = None) -> dict[str, Any]:
        """Navigate to URL using specified or default backend."""
        target = backend or self.active_backend
        if self.backends.get(target):
            return await self.backends[target].navigate(url)
        raise ValueError(f"Backend {target} not initialized")
    
    async def execute_task(self, task: str, backend: Optional[str] = None) -> dict[str, Any]:
        """Execute task - routes to best backend automatically."""
        target = backend or self._choose_backend(task)
        if self.backends.get(target):
            return await self.backends[target].execute_task(task)
        raise ValueError(f"Backend {target} not initialized")
    
    def _choose_backend(self, task: str) -> str:
        """Choose optimal backend based on task type."""
        # On-device tasks -> use ondevice
        if "private" in task.lower() or "local" in task.lower():
            return "ondevice"
        
        # Complex vision tasks -> use lmnr
        if any(kw in task.lower() for kw in ["visual", "image", "screenshot", "analyze"]):
            return "lmnr"
        
        # MCP/CLI access -> use fouwser
        if any(kw in task.lower() for kw in ["mcp", "cli", "automation"]):
            return "fouwser"
        
        # Default to lmnr
        return "lmnr"
    
    async def get_status(self) -> dict[str, Any]:
        """Get status of all backends."""
        return {
            backend: "active" if inst else "inactive"
            for backend, inst in self.backends.items()
        }


class MCPUnifiedServer:
    """MCP server exposing unified browser agent capabilities."""
    
    def __init__(self, agent: UnifiedBrowserAgent):
        self.agent = agent
    
    def get_tools_schema(self) -> list[dict[str, Any]]:
        """Define unified MCP tools schema."""
        return [
            {
                "name": "browser_navigate",
                "description": "Navigate to URL (uses optimal backend)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to navigate to"},
                        "backend": {"type": "string", "enum": ["fouwser", "ondevice", "lmnr"], 
                                    "description": "Specific backend to use"}
                    },
                    "required": ["url"]
                }
            },
            {
                "name": "browser_task",
                "description": "Execute task using vision-based reasoning",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string", "description": "Task description"},
                        "backend": {"type": "string", "enum": ["fouwser", "ondevice", "lmnr"]}
                    },
                    "required": ["task"]
                }
            },
            {
                "name": "browser_click",
                "description": "Click element by selector",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "selector": {"type": "string"}
                    },
                    "required": ["selector"]
                }
            },
            {
                "name": "browser_status",
                "description": "Get status of all browser backends",
                "inputSchema": {"type": "object", "properties": {}}
            }
        ]
    
    async def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Handle MCP tool call."""
        if name == "browser_navigate":
            return await self.agent.navigate(args.get("url"), args.get("backend"))
        elif name == "browser_task":
            return await self.agent.execute_task(args.get("task"), args.get("backend"))
        elif name == "browser_status":
            return await self.agent.get_status()
        return {"status": "error", "error": f"Unknown tool: {name}"}
