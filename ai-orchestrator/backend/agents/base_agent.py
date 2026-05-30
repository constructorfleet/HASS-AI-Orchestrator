"""
Base agent class providing common functionality for all specialist agents.
"""
import os
import json
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import ollama
from ha_client import HAWebSocketClient
from llm_providers import make_chat_provider
from mcp_server import MCPServer

# Optional bus import — avoids a hard dependency when running agents in
# isolation or unit tests that don't wire up the bus.
try:
    from agent_bus import AgentBus, AgentMessage
except ImportError:  # pragma: no cover
    AgentBus = None  # type: ignore[assignment,misc]
    AgentMessage = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------

_DAY_MAP: Dict[str, int] = {
    "mon": 0, "monday": 0,
    "tue": 1, "tuesday": 1,
    "wed": 2, "wednesday": 2,
    "thu": 3, "thursday": 3,
    "fri": 4, "friday": 4,
    "sat": 5, "saturday": 5,
    "sun": 6, "sunday": 6,
}


def _parse_active_hours(spec: str):
    """Parse ``"HH:MM-HH:MM"`` into ``(start_minutes, end_minutes)``."""
    parts = spec.strip().split("-")
    if len(parts) != 2:
        raise ValueError(f"active_hours must be 'HH:MM-HH:MM', got {spec!r}")
    def _to_min(s: str) -> int:
        h, m = s.strip().split(":")
        return int(h) * 60 + int(m)
    return _to_min(parts[0]), _to_min(parts[1])


def is_agent_active(schedule: Optional[Dict], dt: Optional[datetime] = None) -> bool:
    """
    Return ``True`` if the agent should be active at *dt* (default: now).

    Supported schedule keys:

    * ``active_hours`` – ``"HH:MM-HH:MM"``; spans midnight when
      start > end (e.g. ``"22:00-06:00"``).
    * ``days`` – list of day abbreviations, e.g. ``["mon", "tue"]``.
      ``"weekdays"`` and ``"weekends"`` are accepted as shortcuts.
    * ``cron`` – standard 5-field cron expression.  The agent is
      considered active during **any minute that matches** the expression
      (useful for "wake up at 06:00 every weekday" patterns when combined
      with a wide minute range like ``"* 6-9 * * 1-5"``).

    All specified keys must pass for the agent to be considered active.
    An empty or ``None`` schedule always returns ``True``.
    """
    if not schedule:
        return True

    dt = dt or datetime.now()

    # -- cron -----------------------------------------------------------------
    cron_expr = schedule.get("cron")
    if cron_expr:
        try:
            # Import lazily to avoid circular deps in minimal test environments.
            from triggers import CronExpr  # type: ignore[import]
            expr = CronExpr.parse(cron_expr)
            if not expr.matches(dt):
                return False
        except Exception as exc:
            # Bad cron → treat as always active so agents don't silently stop.
            import logging
            logging.getLogger(__name__).warning(
                "Invalid schedule.cron %r: %s — treating as always active", cron_expr, exc
            )

    # -- active_hours ---------------------------------------------------------
    active_hours = schedule.get("active_hours")
    if active_hours:
        start_min, end_min = _parse_active_hours(active_hours)
        current_min = dt.hour * 60 + dt.minute
        if start_min <= end_min:
            in_hours = start_min <= current_min <= end_min
        else:  # spans midnight
            in_hours = current_min >= start_min or current_min <= end_min
        if not in_hours:
            return False

    # -- days -----------------------------------------------------------------
    days_cfg = schedule.get("days")
    if days_cfg:
        # Accept shortcut strings alongside lists.
        if isinstance(days_cfg, str):
            days_cfg = [days_cfg]
        allowed: set = set()
        for d in days_cfg:
            d_lower = d.lower()
            if d_lower == "weekdays":
                allowed.update(range(0, 5))
            elif d_lower == "weekends":
                allowed.update([5, 6])
            else:
                val = _DAY_MAP.get(d_lower)
                if val is None:
                    raise ValueError(f"Unknown day abbreviation: {d!r}")
                allowed.add(val)
        if dt.weekday() not in allowed:
            return False

    return True


class BaseAgent(ABC):
    """
    Abstract base class for all specialist agents.
    Provides common functionality for LLM calls, decision logging, and tool execution.
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        mcp_server: MCPServer,
        ha_client, #: Union[HAWebSocketClient, Callable[[], HAWebSocketClient]]
        skills_path: str,
        rag_manager: Optional[Any] = None,
        model_name: str = "mistral:7b-instruct",
        decision_interval: int = 120,
        broadcast_func: Optional[Any] = None,
        schedule: Optional[Dict] = None,
        agent_bus: Optional[Any] = None,
        publishes: Optional[List[str]] = None,
        listens_to: Optional[List[str]] = None,
    ):
        """
        Initialize base agent.
        
        Args:
            agent_id: Unique agent identifier
            name: Human-readable agent name
            mcp_server: MCP server for tool execution
            ha_client: Home Assistant WebSocket client
            skills_path: Path to SKILLS.md file
            rag_manager: Optional RAG Manager for context retrieval
            model_name: Ollama model name
            decision_interval: Seconds between decisions
            broadcast_func: Optional async callback for dashboard updates
            schedule: Optional schedule dict with ``active_hours``, ``days``
                and/or ``cron`` keys that gate when the agent's decision loop
                runs.  ``None`` means always active.
            agent_bus: Optional :class:`AgentBus` instance for inter-agent
                pub/sub communication.
            publishes: List of topic names this agent will publish events on
                after each decision cycle that produces actions.
            listens_to: List of topic names from other agents; when a message
                arrives on any of these topics the decision loop is woken
                immediately instead of waiting for the next interval.
        """
        self.agent_id = agent_id
        self.name = name
        self.mcp_server = mcp_server
        self.mcp_server = mcp_server
        # Support lazy loading
        self._ha_provider = ha_client
        

        self.skills_path = Path(skills_path)
        self.rag_manager = rag_manager
        self.model_name = model_name
        self.decision_interval = decision_interval
        self.broadcast_func = broadcast_func
        self.status = "initializing"

        # Scheduling
        self.schedule = schedule

        # Inter-agent pub/sub
        self.agent_bus = agent_bus
        self.publishes: List[str] = list(publishes or [])
        self.listens_to: List[str] = list(listens_to or [])
        # Queue fed by on_bus_message(); the decision loop drains it.
        self._trigger_queue: asyncio.Queue = asyncio.Queue()
        # Register subscriptions if a bus was provided.
        if self.agent_bus is not None and self.listens_to:
            for topic in self.listens_to:
                self.agent_bus.subscribe(topic, self.on_bus_message)
        
        # Load skills from SKILLS.md
        self.skills = self.load_skills()
        
        # Ollama client (legacy attribute kept for back-compat) plus the
        # provider-aware chat façade introduced in Phase 9.
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_client = ollama.Client(host=ollama_host)
        self.llm_provider = make_chat_provider(ollama_host=ollama_host)
        
        # Decision storage
        self.decision_dir = Path("/data/decisions") / agent_id
        self.decision_dir.mkdir(parents=True, exist_ok=True)
    
    def load_skills(self) -> Dict:
        """
        Load and parse SKILLS.md file.
        
        Returns:
            Dict containing parsed skill information
        """
        if not self.skills_path.exists():
            print(f"⚠️ SKILLS.md not found at {self.skills_path}, using defaults")
            return {
                "identity": f"{self.name} agent",
                "controllable_entities": [],
                "observable_entities": [],
                "tools": [],
                "decision_criteria": {},
                "performance_targets": {}
            }
        
        with open(self.skills_path, "r") as f:
            content = f.read()
        
        # Basic parsing (can be enhanced with proper markdown parser)
        skills = {
            "identity": self._extract_section(content, "Identity"),
            "controllable_entities": self._extract_list(content, "Controllable Entities"),
            "observable_entities": self._extract_list(content, "Observable Entities"),
            "tools": self._extract_section(content, "Available Tools"),
            "decision_criteria": self._extract_section(content, "Decision Criteria"),
            "performance_targets": self._extract_section(content, "Performance Targets"),
            "full_content": content
        }
        
        return skills
    
    def _extract_section(self, content: str, heading: str) -> str:
        """Extract content from a markdown section"""
        lines = content.split("\n")
        in_section = False
        section_lines = []
        
        for line in lines:
            if heading.lower() in line.lower() and line.startswith("#"):
                in_section = True
                continue
            if in_section:
                if line.startswith("#"):
                    break
                section_lines.append(line)
        
        return "\n".join(section_lines).strip()
    
    def _extract_list(self, content: str, heading: str) -> List[str]:
        """Extract list items from a markdown section"""
        section = self._extract_section(content, heading)
        items = []
        for line in section.split("\n"):
            line = line.strip()
            if line.startswith("-") or line.startswith("*"):
                items.append(line.lstrip("-*").strip().strip("`"))
        return items
    
    async def _call_llm(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        Call the configured LLM provider (Ollama / OpenAI / GitHub
        Models / Foundry) and return the assistant text.

        Args:
            prompt: Prompt text
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        try:
            content = await asyncio.to_thread(
                self.llm_provider.chat,
                self.model_name,
                [{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Strip any remaining <think>...</think> blocks (Issue #12 failsafe)
            import re
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

            return content.strip()

        except Exception as e:
            print(f"❌ LLM call failed: {e}")
            return f"ERROR: {str(e)}"
    
    def _build_system_prompt(self) -> str:
        """Build system prompt from SKILLS.md"""
        prompt = f"""You are {self.skills['identity']}.

Your role is to make intelligent decisions about home automation based on current conditions.

Available Tools:
{self.skills['tools']}

Decision Criteria:
{self.skills['decision_criteria']}
"""
        # Inject RAG context if available in skills (populated during decide)
        if "relevant_knowledge" in self.skills:
             prompt += f"\nRELEVANT KNOWLEDGE (from memory/docs):\n{self.skills['relevant_knowledge']}\n"
             
        # Append output format instructions
        prompt += """
Respond with a JSON object containing your decision in this format:
{
  "reasoning": "Brief explanation of why you made this decision",
  "actions": [
    {
      "tool": "tool_name",
      "parameters": {
        "param1": "value1"
      }
    }
  ]
}

If no action is needed, return an empty actions array.
"""
        return prompt
    
    @abstractmethod
    async def decide(self, context: Dict) -> Dict:
        """
        Make a decision based on current context.
        Must be implemented by specialist agents.
        
        Args:
            context: Current state and context information
        
        Returns:
            Decision dict with reasoning and actions
        """
        pass
    
    async def execute(self, decision: Dict) -> List[Dict]:
        """
        Execute decision actions using MCP tools.
        
        Args:
            decision: Decision dict from decide()
        
        Returns:
            List of execution results
        """
        actions = decision.get("actions", [])
        results = []
        
        for action in actions:
            tool_name = action["tool"]
            parameters = action["parameters"]
            
            result = await self.mcp_server.execute_tool(
                tool_name=tool_name,
                parameters=parameters,
                agent_id=self.agent_id
            )
            
            results.append({
                "tool": tool_name,
                "parameters": parameters,
                "result": result
            })
        
        return results
    
    def log_decision(self, context: Dict, decision: Dict, results: List[Dict]):
        """Save decision to log file"""
        timestamp = datetime.now()
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "agent_id": self.agent_id,
            "context": context,
            "decision": decision,
            "execution_results": results,
            "dry_run": self.mcp_server.dry_run
        }
        
        log_file = self.decision_dir / f"{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, "w") as f:
            json.dump(log_entry, f, indent=2)

    async def retrieve_context(self, state_text: str) -> str:
        """
        Retrieve relevant context from RAG based on current state description.
        
        Args:
            state_text: Description of current situation to query against
            
        Returns:
            Formatted string of relevant knowledge
        """
        if not self.rag_manager:
            return ""
            
        results = self.rag_manager.query(
            query_text=state_text,
            collection_names=["knowledge_base", "entity_registry", "memory"],
            n_results=2
        )
        
        if not results:
            return ""
            
        knowledge_str = ""
        for res in results:
            source = res.get("source", "unknown")
            content = res.get("content", "").strip()
            knowledge_str += f"- [{source}] {content}\n"
            
        return knowledge_str
    
    def get_last_decision_file(self) -> Optional[Path]:
        """Get path to most recent decision log"""
        decision_files = sorted(self.decision_dir.glob("*.json"), reverse=True)
        return decision_files[0] if decision_files else None
    
    async def _broadcast_status(self, status: str):
        """Broadcast status update to dashboard"""
        if self.broadcast_func:
            await self.broadcast_func({
                "type": "agent_update",
                "data": {
                    "agent_id": self.agent_id,
                    "name": self.name,
                    "status": status,
                    "last_active": datetime.now().isoformat()
                }
            })

    async def run_decision_loop(self):
        """Main decision loop that runs continuously"""
        self.status = "idle"
        # Delay start slightly to allow system to settle
        await asyncio.sleep(5)
        sched_info = f", schedule={self.schedule}" if self.schedule else ""
        listen_info = f", listens_to={self.listens_to}" if self.listens_to else ""
        print(f"✓ {self.name} decision loop started (interval: {self.decision_interval}s{sched_info}{listen_info})")
        
        while True:
            try:
                # --- Schedule gate ---
                if not is_agent_active(self.schedule):
                    self.status = "scheduled_inactive"
                    await self._broadcast_status("scheduled_inactive")
                    # Re-check every 60 s to avoid tight busy-loop while still
                    # reacting promptly when the active window opens.
                    await asyncio.sleep(60)
                    continue

                self.status = "deciding"
                await self._broadcast_status("deciding")

                # Drain any pending bus trigger message to include in context.
                triggered_by: Optional[Dict] = None
                try:
                    msg = self._trigger_queue.get_nowait()
                    triggered_by = {
                        "topic": msg.topic,
                        "sender_id": msg.sender_id,
                        "payload": msg.payload,
                        "timestamp": msg.timestamp,
                    }
                    print(f"⚡ {self.name} triggered by bus event: {msg.topic} from {msg.sender_id}")
                except asyncio.QueueEmpty:
                    pass
                
                # Make decision
                context = await self.gather_context()
                if triggered_by:
                    context["triggered_by"] = triggered_by
                decision = await self.decide(context)
                
                # Execute decision
                results = await self.execute(decision)
                
                # Log decision
                self.log_decision(context, decision, results)
                
                # Publish decision events to the bus for listening agents.
                await self._publish_decision_events(decision)

                # Broadcast decision result
                if self.broadcast_func:
                    await self.broadcast_func({
                        "type": "decision",
                        "data": {
                            "timestamp": datetime.now().isoformat(),
                            "agent_id": self.agent_id,
                            "reasoning": decision.get("reasoning", ""),
                            "action": str(decision.get("actions", [])),
                            "dry_run": self.mcp_server.dry_run
                        }
                    })
                
                self.status = "idle"
                await self._broadcast_status("idle")
                print(f"✓ {self.name} decision completed (waiting {self.decision_interval}s)")
                
                # Sleep until next interval OR a bus message arrives on a
                # listened topic, whichever comes first.
                await self._wait_for_next_cycle()
            
            except Exception as e:
                self.status = "error"
                print(f"❌ {self.name} decision loop error: {e}")
                await self._broadcast_status("error")
                await asyncio.sleep(10)  # Back off on error

    async def _wait_for_next_cycle(self) -> None:
        """Sleep until ``decision_interval`` elapses **or** a bus message wakes us."""
        if not self.listens_to:
            # No subscriptions — plain sleep.
            await asyncio.sleep(self.decision_interval)
            return

        # Create tasks for both the timer and the next queue item.
        sleep_task = asyncio.create_task(asyncio.sleep(self.decision_interval))
        trigger_task = asyncio.create_task(self._trigger_queue.get())

        done, pending = await asyncio.wait(
            {sleep_task, trigger_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel whichever didn't fire.
        for task in pending:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        # If a bus message arrived, put it back so the next loop iteration
        # can include it in the decision context.
        if trigger_task in done:
            try:
                msg = trigger_task.result()
                await self._trigger_queue.put(msg)
            except Exception:
                pass

    async def on_bus_message(self, message: Any) -> None:
        """
        Receive a message from the :class:`AgentBus`.

        Called by the bus when a message arrives on any of the topics
        listed in :attr:`listens_to`.  The message is placed in the
        internal trigger queue so the decision loop picks it up on the
        next iteration (or wakes immediately if sleeping).
        """
        await self._trigger_queue.put(message)

    async def _publish_decision_events(self, decision: Dict) -> None:
        """
        Publish agent bus events after a decision that produced actions.

        Each topic in :attr:`publishes` receives an :class:`AgentMessage`
        with the decision reasoning and action list as payload.  Topics are
        only published when at least one action was returned.
        """
        if not self.agent_bus or not self.publishes:
            return
        actions = decision.get("actions", [])
        if not actions:
            return
        payload = {
            "reasoning": decision.get("reasoning", ""),
            "actions": actions,
        }
        for topic in self.publishes:
            try:
                msg = AgentMessage(  # type: ignore[call-arg]
                    topic=topic,
                    sender_id=self.agent_id,
                    payload=payload,
                )
                await self.agent_bus.publish(msg)
            except Exception as exc:
                print(f"⚠️ {self.name}: failed to publish to bus topic {topic!r}: {exc}")
    
    @abstractmethod
    async def gather_context(self) -> Dict:
        """
        Gather current context for decision making.
        Must be implemented by specialist agents.
        
        Returns:
            Context dict with current state
        """
        pass

    @property
    def ha_client(self):
        """Lazy retrieval of HA client"""
        if callable(self._ha_provider):
            return self._ha_provider()
        return self._ha_provider

    @ha_client.setter
    def ha_client(self, value):
        self._ha_provider = value
