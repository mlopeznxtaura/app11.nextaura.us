"""Browser agent configuration."""
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class BrowserConfig:
    """Unified configuration for all browser agents."""
    
    # Default backend: "fouwser", "ondevice", or "lmnr"
    default_backend: str = "lmnr"
    
    # Fouwser configuration (MCP Chromium fork)
    fouwser: dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "mcp_port": 4000,
        "chrome_path": None,  # Path to custom Chromium if needed
    })
    
    # On-device configuration (WebLLM/WebGPU)
    ondevice: dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "model_path": "./models/ondevice-llm",
        "webgpu": True,
        "offline": True,
        "agents": ["planner", "navigator"],
    })
    
    # LMNR configuration (vision-based reasoning)
    lmnr: dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "api_key": None,  # LLM API key for vision-based reasoning
        "model": "gemini-2.5-pro",
        "observability": True,
    })
    
    # Common settings
    timeout: int = 30000
    headless: bool = True
    
    @classmethod
    def from_env(cls) -> "BrowserConfig":
        """Load configuration from environment variables."""
        import os
        
        return cls(
            default_backend=os.getenv("BROWSER_DEFAULT", "lmnr"),
            fouwser={
                "enabled": os.getenv("FOUWSER_ENABLED", "false").lower() == "true",
                "mcp_port": int(os.getenv("FOUWSER_PORT", "4000")),
            },
            ondevice={
                "enabled": os.getenv("ONDEVICE_ENABLED", "true").lower() == "true",
                "webgpu": os.getenv("ONDEVICE_WEBGPU", "true").lower() == "true",
                "offline": os.getenv("ONDEVICE_OFFLINE", "true").lower() == "true",
            },
            lmnr={
                "enabled": os.getenv("LMNR_ENABLED", "true").lower() == "true",
                "api_key": os.getenv("LMNR_API_KEY"),
                "model": os.getenv("LMNR_MODEL", "gemini-2.5-pro"),
            },
            headless=os.getenv("BROWSER_HEADLESS", "true").lower() == "true",
        )
