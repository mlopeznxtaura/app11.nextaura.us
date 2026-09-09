"""Browser agent implementation for automated web interactions."""
import asyncio
import json
from typing import Any, Optional


class BrowserAgent:
    """Browser agent for executing navigation and interaction tasks."""
    
    def __init__(self, driver: Any, model_client: Any):
        """
        Initialize browser agent.
        
        Args:
            driver: Browser driver instance (e.g., Playwright, Selenium)
            model_client: LLM client for vision-based reasoning
        """
        self.driver = driver
        self.model_client = model_client
        self.history: list[dict[str, Any]] = []
        self.observation: Optional[dict[str, Any]] = None
    
    async def navigate(self, url: str) -> dict[str, Any]:
        """Navigate to a URL and capture state."""
        try:
            await self.driver.goto(url)
            await self.driver.wait_for_load_state("domcontentloaded")
            
            self.observation = {
                "url": self.driver.url,
                "title": await self.driver.title(),
                "screenshot": await self.driver.screenshot(
                    path="screenshot.png",
                    full_page=True
                ),
                "html": await self.driver.content(),
            }
            self.history.append({"action": "navigate", "url": url})
            
            return {"status": "success", "url": self.driver.url}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def execute_task(self, task: str) -> dict[str, Any]:
        """
        Execute a natural language task using vision-based reasoning.
        
        Args:
            task: Natural language description of the task
            
        Returns:
            Task execution result
        """
        steps = await self._plan_task(task)
        result = {"task": task, "steps": [], "status": "pending"}
        
        for step in steps:
            step_result = await self._execute_step(step)
            result["steps"].append(step_result)
            
            if step_result.get("status") == "error":
                result["status"] = "failed"
                break
        
        if result["steps"]:
            result["status"] = "completed"
        
        return result
    
    async def _plan_task(self, task: str) -> list[dict[str, Any]]:
        """Generate task plan using LLM."""
        prompt = f"""Plan browser automation steps for this task:
{task}

Provide steps as JSON array with action and parameters."""
        
        response = await self.model_client.generate(prompt)
        try:
            steps = json.loads(response)
            return steps if isinstance(steps, list) else []
        except json.JSONDecodeError:
            return []
    
    async def _execute_step(self, step: dict[str, Any]) -> dict[str, Any]:
        """Execute a single automation step."""
        action = step.get("action", "")
        params = step.get("params", {})
        
        try:
            if action == "click":
                await self.driver.click(params.get("selector"))
                return {"status": "success", "action": action}
            
            elif action == "fill":
                await self.driver.fill(params.get("selector"), params.get("value"))
                return {"status": "success", "action": action}
            
            elif action == "press":
                await self.driver.press(params.get("selector"), params.get("key"))
                return {"status": "success", "action": action}
            
            elif action == "get_text":
                text = await self.driver.text_content(params.get("selector"))
                return {"status": "success", "action": action, "value": text}
            
            elif action == "wait":
                await asyncio.sleep(params.get("duration", 1))
                return {"status": "success", "action": action}
            
            return {"status": "error", "error": f"Unknown action: {action}"}
        
        except Exception as e:
            return {"status": "error", "action": action, "error": str(e)}
