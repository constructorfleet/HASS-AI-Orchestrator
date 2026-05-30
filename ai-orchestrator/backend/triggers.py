"""
Proactive triggers for the deep reasoning agent (Phase 8 / Milestone F).

Two trigger families:

* :class:`CronTrigger` — fires on a wall-clock schedule
  (5-field cron expressions plus a few common aliases).
* :class:`StateChangeTrigger` — fires when a Home Assistant entity
  matches a pattern, optionally for a sustained duration, with
  configurable cooldown to prevent thrash.

When a trigger fires it invokes a :class:`DeepReasoningAgent`
callback with a goal templated from the trigger config. Triggers
*always* run in ``mode="auto"``, so anything dangerous lands in the
PAE approval queue from Milestone E rather than firing unattended.

The registry persists triggers + recent fire history in SQLite at
``/data/triggers.db`` (or a workspace-local fallback).

This module deliberately has **no hard dependency** on
:mod:`apscheduler`, :mod:`croniter`, or any other optional package.
The cron evaluator below covers the common cases (`*`, ranges,
lists, steps, and `@hourly` / `@daily` / `@nightly` aliases) — which
is what every realistic home-automation schedule needs.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cron parsing
# ---------------------------------------------------------------------------
CRON_ALIASES: Dict[str, str] = {
    "@hourly":   "0 * * * *",
    "@daily":    "0 0 * * *",
    "@nightly":  "0 22 * * *",   # convenient default for "after dark"
    "@midnight": "0 0 * * *",
    "@weekly":   "0 0 * * 0",
    "@monthly":  "0 0 1 * *",
}

_CRON_RANGES = (
    (0, 59),   # minute
    (0, 23),   # hour
    (1, 31),   # day of month
    (1, 12),   # month
    (0, 6),    # day of week (0 = Sunday)
)


def _parse_cron_field(field: str, lo: int, hi: int) -> List[int]:
    """Expand one cron field to a sorted list of ints in ``[lo, hi]``."""
    out: set = set()
    for part in field.split(","):
        step = 1
        if "/" in part:
            base, step_s = part.split("/", 1)
            step = int(step_s)
            if step < 1:
                raise ValueError(f"step must be >=1 in cron field {field!r}")
        else:
            base = part

        if base == "*":
            start, end = lo, hi
        elif "-" in base:
            a, b = base.split("-", 1)
            start, end = int(a), int(b)
            if start < lo or end > hi or start > end:
                raise ValueError(f"range {base!r} outside [{lo},{hi}]")
        else:
            v = int(base)
            if v < lo or v > hi:
                raise ValueError(f"value {v} outside [{lo},{hi}]")
            start = end = v
        for v in range(start, end + 1, step):
            out.add(v)
    if not out:
        raise ValueError(f"empty cron field {field!r}")
    return sorted(out)


@dataclass
class CronExpr:
    minute: List[int]
    hour: List[int]
    dom: List[int]
    month: List[int]
    dow: List[int]
    raw: str

    @classmethod
    def parse(cls, expr: str) -> "CronExpr":
        s = expr.strip()
        if s.lower() in CRON_ALIASES:
            s = CRON_ALIASES[s.lower()]
        parts = s.split()
        if len(parts) != 5:
            raise ValueError(f"cron expression must have 5 fields: {expr!r}")
        fields = [_parse_cron_field(p, lo, hi) for p, (lo, hi) in zip(parts, _CRON_RANGES)]
        return cls(
            minute=fields[0], hour=fields[1], dom=fields[2],
            month=fields[3], dow=fields[4], raw=expr,
        )

    def matches(self, dt: datetime) -> bool:
        # Cron's day field: when both DOM and DOW are restricted (i.e.
        # not the full range), a match on *either* is enough. Standard
        # cron behaviour.
        dom_unrestricted = self.dom == list(range(1, 32))
        dow_unrestricted = self.dow == list(range(0, 7))
        dom_match = dt.day in self.dom
        # cron DoW: Sunday=0..6=Saturday; Python: Monday=0..6=Sunday.
        py_dow = dt.weekday()  # 0=Mon
        cron_dow = (py_dow + 1) % 7
        dow_match = cron_dow in self.dow

        if dom_unrestricted and dow_unrestricted:
            day_ok = True
        elif dom_unrestricted:
            day_ok = dow_match
        elif dow_unrestricted:
            day_ok = dom_match
        else:
            day_ok = dom_match or dow_match

        return (
            dt.minute in self.minute
            and dt.hour in self.hour
            and dt.month in self.month
            and day_ok
        )

    def next_fire_after(self, after: datetime, *, max_lookahead_minutes: int = 60 * 24 * 366) -> Optional[datetime]:
        """Return the next minute >= ``after`` that matches, or ``None``
        if nothing in the next ``max_lookahead_minutes``."""
        # Round up to the next whole minute.
        candidate = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
        for _ in range(max_lookahead_minutes):
            if self.matches(candidate):
                return candidate
            candidate += timedelta(minutes=1)
        return None


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class TriggerSpec:
    """Persisted trigger definition."""

    id: str
    name: str
    type: str             # "cron" | "state"
    goal_template: str
    enabled: bool = True
    # cron
    cron: Optional[str] = None
    # state
    entity_id: Optional[str] = None
    state_pattern: Optional[str] = None     # exact value or regex starting with "~"
    sustained_seconds: int = 0
    # for both
    cooldown_seconds: int = 600
    mode: str = "auto"                      # always auto-recommended
    extra_context: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_fired_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TriggerFireRecord:
    id: str
    trigger_id: str
    timestamp: str
    goal: str
    run_id: Optional[str] = None
    plan_id: Optional[str] = None
    status: str = "submitted"
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
class TriggerStore:
    """SQLite-backed store for trigger definitions and fire history."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS triggers (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        type TEXT NOT NULL,
        goal_template TEXT NOT NULL,
        enabled INTEGER NOT NULL,
        cron TEXT,
        entity_id TEXT,
        state_pattern TEXT,
        sustained_seconds INTEGER NOT NULL DEFAULT 0,
        cooldown_seconds INTEGER NOT NULL DEFAULT 600,
        mode TEXT NOT NULL DEFAULT 'auto',
        extra_context_json TEXT,
        created_at TEXT NOT NULL,
        last_fired_at TEXT
    );
    CREATE TABLE IF NOT EXISTS trigger_fires (
        id TEXT PRIMARY KEY,
        trigger_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        goal TEXT NOT NULL,
        run_id TEXT,
        plan_id TEXT,
        status TEXT NOT NULL,
        note TEXT,
        FOREIGN KEY (trigger_id) REFERENCES triggers(id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_trigger_fires ON trigger_fires(trigger_id, timestamp DESC);
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            import os
            data_base = Path(os.environ.get("DATA_DIR", "/data"))
            if not data_base.exists() or not os.access(str(data_base), os.W_OK):
                data_base = Path(__file__).parent.parent / "data"
            data_base.mkdir(parents=True, exist_ok=True)
            db_path = str(data_base / "triggers.db")
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(self.SCHEMA)
        logger.info("TriggerStore initialised at %s", self.db_path)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Trigger CRUD
    # ------------------------------------------------------------------
    def save(self, t: TriggerSpec) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO triggers (
                    id, name, type, goal_template, enabled,
                    cron, entity_id, state_pattern, sustained_seconds,
                    cooldown_seconds, mode, extra_context_json,
                    created_at, last_fired_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    t.id, t.name, t.type, t.goal_template, int(t.enabled),
                    t.cron, t.entity_id, t.state_pattern, t.sustained_seconds,
                    t.cooldown_seconds, t.mode,
                    json.dumps(t.extra_context) if t.extra_context else None,
                    t.created_at, t.last_fired_at,
                ),
            )

    def get(self, trigger_id: str) -> Optional[TriggerSpec]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM triggers WHERE id = ?", (trigger_id,)).fetchone()
        return _row_to_trigger(row) if row else None

    def list(self, *, enabled_only: bool = False) -> List[TriggerSpec]:
        q = "SELECT * FROM triggers"
        args: Tuple[Any, ...] = ()
        if enabled_only:
            q += " WHERE enabled = 1"
        q += " ORDER BY created_at DESC"
        with self._conn() as c:
            return [_row_to_trigger(r) for r in c.execute(q, args).fetchall()]

    def delete(self, trigger_id: str) -> bool:
        with self._conn() as c:
            cur = c.execute("DELETE FROM triggers WHERE id = ?", (trigger_id,))
        return cur.rowcount > 0

    def mark_fired(self, trigger_id: str, when: datetime) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE triggers SET last_fired_at = ? WHERE id = ?",
                (when.isoformat(), trigger_id),
            )

    # ------------------------------------------------------------------
    # Fire history
    # ------------------------------------------------------------------
    def record_fire(self, fire: TriggerFireRecord) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT INTO trigger_fires
                   (id, trigger_id, timestamp, goal, run_id, plan_id, status, note)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (fire.id, fire.trigger_id, fire.timestamp, fire.goal,
                 fire.run_id, fire.plan_id, fire.status, fire.note),
            )

    def list_fires(self, *, trigger_id: Optional[str] = None, limit: int = 50) -> List[TriggerFireRecord]:
        q = "SELECT * FROM trigger_fires"
        args: Tuple[Any, ...] = ()
        if trigger_id:
            q += " WHERE trigger_id = ?"
            args = (trigger_id,)
        q += " ORDER BY timestamp DESC LIMIT ?"
        args = args + (int(limit),)
        with self._conn() as c:
            return [_row_to_fire(r) for r in c.execute(q, args).fetchall()]


def _row_to_trigger(row: sqlite3.Row) -> TriggerSpec:
    extras_raw = row["extra_context_json"]
    return TriggerSpec(
        id=row["id"], name=row["name"], type=row["type"],
        goal_template=row["goal_template"], enabled=bool(row["enabled"]),
        cron=row["cron"], entity_id=row["entity_id"],
        state_pattern=row["state_pattern"],
        sustained_seconds=int(row["sustained_seconds"] or 0),
        cooldown_seconds=int(row["cooldown_seconds"] or 600),
        mode=row["mode"] or "auto",
        extra_context=json.loads(extras_raw) if extras_raw else {},
        created_at=row["created_at"],
        last_fired_at=row["last_fired_at"],
    )


def _row_to_fire(row: sqlite3.Row) -> TriggerFireRecord:
    return TriggerFireRecord(
        id=row["id"], trigger_id=row["trigger_id"],
        timestamp=row["timestamp"], goal=row["goal"],
        run_id=row["run_id"], plan_id=row["plan_id"],
        status=row["status"], note=row["note"],
    )


# ---------------------------------------------------------------------------
# Reasoner callback type
# ---------------------------------------------------------------------------
ReasonerCallback = Callable[[str, Dict[str, Any]], Awaitable[Any]]
"""``async (goal: str, context: dict) -> result_with_run_id_and_plan``.

The registry calls this with the templated goal whenever a trigger
fires. The callback should run the deep reasoner in
``mode="auto"`` and return an object with ``run_id`` and ``plan``
attributes (or a dict with the same keys).
"""


# ---------------------------------------------------------------------------
# State-change pattern matching
# ---------------------------------------------------------------------------
def _state_matches(value: Any, pattern: Optional[str]) -> bool:
    """Compare a Home Assistant state value to a trigger pattern.

    * ``None`` pattern matches anything.
    * Pattern starting with ``~`` is a regex (without the leading ``~``).
    * Otherwise exact string match (case-insensitive).
    """
    if pattern is None or pattern == "":
        return True
    s = "" if value is None else str(value)
    if pattern.startswith("~"):
        try:
            return re.search(pattern[1:], s) is not None
        except re.error:
            logger.warning("Invalid trigger regex %r — treating as no-match", pattern[1:])
            return False
    return s.lower() == pattern.lower()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class TriggerRegistry:
    """Owns the cron loop, the state-change subscription glue, and
    the cooldown / debounce bookkeeping for every active trigger.

    Lifecycle:

        registry = TriggerRegistry(store, reasoner_callback, ha_client=ws)
        await registry.start()
        # … runs until …
        await registry.stop()

    Triggers added via :meth:`add` are picked up immediately. Removing
    or disabling cancels any pending wait.
    """

    def __init__(
        self,
        store: TriggerStore,
        reasoner_callback: ReasonerCallback,
        *,
        ha_client: Optional[Any] = None,
        broadcast_func: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        cron_tick_seconds: float = 30.0,
    ) -> None:
        self.store = store
        self._reason = reasoner_callback
        self.ha_client = ha_client
        self.broadcast_func = broadcast_func
        self.cron_tick_seconds = cron_tick_seconds

        self._cron_task: Optional[asyncio.Task] = None
        self._stopping = asyncio.Event()
        # Per-trigger sustained-state pending-fire timers, keyed by id.
        self._sustain_tasks: Dict[str, asyncio.Task] = {}
        # Last fire timestamps (epoch seconds) for in-process cooldown.
        self._last_fired: Dict[str, float] = {}
        # State subscription handle (one shared subscription, internally fanned out).
        self._state_sub_id: Optional[int] = None

    # ------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._cron_task is not None and not self._cron_task.done()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        if self.running:
            return
        self._stopping.clear()
        self._cron_task = asyncio.create_task(self._cron_loop(), name="trigger_cron_loop")
        await self._refresh_state_subscription()
        logger.info("TriggerRegistry started")

    async def stop(self) -> None:
        self._stopping.set()
        if self._cron_task:
            self._cron_task.cancel()
            try:
                await self._cron_task
            except (asyncio.CancelledError, Exception):
                pass
            self._cron_task = None
        for t in list(self._sustain_tasks.values()):
            t.cancel()
        self._sustain_tasks.clear()
        logger.info("TriggerRegistry stopped")

    # ------------------------------------------------------------------
    # CRUD-ish
    # ------------------------------------------------------------------
    async def add(self, spec: TriggerSpec) -> TriggerSpec:
        _validate_spec(spec)
        if not spec.id:
            spec.id = uuid.uuid4().hex
        self.store.save(spec)
        if spec.type == "state" and self.running:
            await self._refresh_state_subscription()
        logger.info("Trigger added: %s (%s)", spec.id, spec.name)
        return spec

    async def update(self, spec: TriggerSpec) -> TriggerSpec:
        _validate_spec(spec)
        self.store.save(spec)
        # Drop any in-flight sustained-state debounce for this id.
        t = self._sustain_tasks.pop(spec.id, None)
        if t:
            t.cancel()
        if self.running:
            await self._refresh_state_subscription()
        return spec

    async def delete(self, trigger_id: str) -> bool:
        ok = self.store.delete(trigger_id)
        t = self._sustain_tasks.pop(trigger_id, None)
        if t:
            t.cancel()
        if ok and self.running:
            await self._refresh_state_subscription()
        return ok

    def list(self, **kw) -> List[TriggerSpec]:
        return self.store.list(**kw)

    def list_fires(self, **kw) -> List[TriggerFireRecord]:
        return self.store.list_fires(**kw)

    # ------------------------------------------------------------------
    # Cron loop
    # ------------------------------------------------------------------
    async def _cron_loop(self) -> None:
        # Track the last minute we evaluated to avoid double-firing
        # when the loop iteration happens to span a minute boundary.
        last_minute: Optional[datetime] = None
        try:
            while not self._stopping.is_set():
                now = datetime.now().replace(second=0, microsecond=0)
                if now != last_minute:
                    last_minute = now
                    await self._evaluate_cron(now)
                # Sleep until next tick or cancellation.
                try:
                    await asyncio.wait_for(self._stopping.wait(), timeout=self.cron_tick_seconds)
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("cron_loop crashed; trigger evaluation will halt until restart")

    async def _evaluate_cron(self, now: datetime) -> None:
        for spec in self.store.list(enabled_only=True):
            if spec.type != "cron" or not spec.cron:
                continue
            try:
                expr = CronExpr.parse(spec.cron)
            except ValueError as exc:
                logger.warning("Trigger %s has invalid cron %r: %s", spec.id, spec.cron, exc)
                continue
            if not expr.matches(now):
                continue
            if not self._cooldown_ok(spec):
                continue
            asyncio.create_task(self._fire(spec, reason="cron tick"))

    # ------------------------------------------------------------------
    # State-change handling
    # ------------------------------------------------------------------
    async def _refresh_state_subscription(self) -> None:
        if self.ha_client is None:
            return
        watched = sorted({s.entity_id for s in self.store.list(enabled_only=True)
                          if s.type == "state" and s.entity_id})
        if not watched:
            return
        if not getattr(self.ha_client, "connected", False):
            logger.debug("HA client not connected; deferring state-subscription refresh")
            return
        if self._state_sub_id is not None:
            # Already subscribed to state_changed (HA fires it for every
            # entity); we just update the in-memory watched-set via the
            # store. No need to resubscribe.
            return
        try:
            self._state_sub_id = await self.ha_client.subscribe_entities(
                entity_ids=watched,
                callback=self._handle_state_event,
            )
            logger.info("Subscribed to %d state-change trigger entities", len(watched))
        except Exception as exc:
            logger.warning("Failed to subscribe state-change triggers: %s", exc)

    async def _handle_state_event(self, event: Dict[str, Any]) -> None:
        data = event.get("data") or {}
        entity_id = data.get("entity_id")
        new_state = (data.get("new_state") or {}).get("state")
        if not entity_id:
            return
        for spec in self.store.list(enabled_only=True):
            if spec.type != "state" or spec.entity_id != entity_id:
                continue
            if not _state_matches(new_state, spec.state_pattern):
                # Cancel any pending sustain timer for this trigger.
                t = self._sustain_tasks.pop(spec.id, None)
                if t:
                    t.cancel()
                continue
            if spec.sustained_seconds > 0:
                # Already counting down? Leave it alone.
                if spec.id in self._sustain_tasks and not self._sustain_tasks[spec.id].done():
                    continue
                self._sustain_tasks[spec.id] = asyncio.create_task(
                    self._sustain_then_fire(spec, new_state)
                )
            else:
                if self._cooldown_ok(spec):
                    asyncio.create_task(self._fire(spec, reason=f"state {entity_id}={new_state}"))

    async def _sustain_then_fire(self, spec: TriggerSpec, observed_state: Any) -> None:
        try:
            await asyncio.sleep(spec.sustained_seconds)
        except asyncio.CancelledError:
            return
        # Re-check current state lazily — if the HA client gives us a
        # cheap path, prefer it; otherwise we trust our last observation.
        current = observed_state
        if self.ha_client is not None and hasattr(self.ha_client, "get_states"):
            try:
                fresh = await self.ha_client.get_states(spec.entity_id)
                current = (fresh or {}).get("state", current)
            except Exception:
                pass
        if not _state_matches(current, spec.state_pattern):
            return  # state moved on before sustain expired — skip
        if not self._cooldown_ok(spec):
            return
        await self._fire(spec, reason=f"state {spec.entity_id}={current} sustained {spec.sustained_seconds}s")

    # ------------------------------------------------------------------
    # Fire
    # ------------------------------------------------------------------
    def _cooldown_ok(self, spec: TriggerSpec) -> bool:
        last = self._last_fired.get(spec.id)
        if last is None:
            # Also honour persisted last_fired_at across restarts.
            if spec.last_fired_at:
                try:
                    ts = datetime.fromisoformat(spec.last_fired_at).timestamp()
                    last = ts
                except ValueError:
                    last = None
        if last is None:
            return True
        return (time.time() - last) >= spec.cooldown_seconds

    async def _fire(self, spec: TriggerSpec, *, reason: str) -> None:
        now = datetime.now(timezone.utc)
        self._last_fired[spec.id] = now.timestamp()
        self.store.mark_fired(spec.id, now)

        goal = _render_goal(spec.goal_template, {
            "entity_id": spec.entity_id,
            "trigger_name": spec.name,
            "trigger_id": spec.id,
            "now": now.isoformat(),
            "reason": reason,
        })
        context = {
            "trigger_id": spec.id,
            "trigger_name": spec.name,
            "trigger_type": spec.type,
            "trigger_reason": reason,
            **(spec.extra_context or {}),
        }
        fire = TriggerFireRecord(
            id=uuid.uuid4().hex,
            trigger_id=spec.id,
            timestamp=now.isoformat(),
            goal=goal,
            status="submitted",
        )
        try:
            result = await self._reason(goal, context)
            fire.run_id = _attr(result, "run_id")
            plan = _attr(result, "plan")
            if isinstance(plan, dict):
                fire.plan_id = plan.get("id")
                if plan.get("requires_approval"):
                    fire.status = "awaiting_approval"
                elif plan.get("status") == "executed":
                    fire.status = "executed"
                else:
                    fire.status = "completed"
            else:
                fire.status = "completed"
        except Exception as exc:
            logger.exception("Trigger %s reasoner call failed", spec.id)
            fire.status = "error"
            fire.note = f"{type(exc).__name__}: {exc}"

        self.store.record_fire(fire)
        if self.broadcast_func is not None:
            try:
                await self.broadcast_func({
                    "type": "trigger_fired",
                    "data": fire.to_dict(),
                })
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Validation + helpers
# ---------------------------------------------------------------------------
def _validate_spec(spec: TriggerSpec) -> None:
    if spec.type not in ("cron", "state"):
        raise ValueError("trigger.type must be 'cron' or 'state'")
    if not spec.goal_template.strip():
        raise ValueError("trigger.goal_template is required")
    if spec.type == "cron":
        if not spec.cron:
            raise ValueError("cron triggers require a cron expression")
        CronExpr.parse(spec.cron)  # raises ValueError if bad
    else:
        if not spec.entity_id:
            raise ValueError("state triggers require entity_id")
        if spec.sustained_seconds < 0:
            raise ValueError("sustained_seconds must be >= 0")
    if spec.cooldown_seconds < 0:
        raise ValueError("cooldown_seconds must be >= 0")
    if spec.mode not in ("auto", "plan", "execute"):
        raise ValueError("trigger.mode must be auto|plan|execute")


_TEMPLATE_RE = re.compile(r"\{(\w+)\}")


def _render_goal(template: str, vars: Dict[str, Any]) -> str:
    """Tiny ``{name}`` substitution. Missing names are left intact so
    the LLM still sees the placeholder rather than crashing."""
    def _sub(m: re.Match) -> str:
        key = m.group(1)
        v = vars.get(key)
        return str(v) if v is not None else m.group(0)
    return _TEMPLATE_RE.sub(_sub, template)


def _attr(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)
