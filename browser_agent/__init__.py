"""Unified browser agent integration for app9.nextaura.us.

Integrates three browser agent backends:
- fouwser: MCP-enabled Chromium fork
- ondevice: Privacy-focused on-device agent (WebLLM/WebGPU)
- lmnr: Vision-based reasoning agent
"""

from .integrator import UnifiedBrowserAgent
from .config import BrowserConfig

__all__ = ["UnifiedBrowserAgent", "BrowserConfig"]
