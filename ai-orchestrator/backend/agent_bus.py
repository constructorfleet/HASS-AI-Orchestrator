"""
Lightweight async pub/sub message bus for inter-agent communication.

Agents can *publish* events to named topics and *subscribe* to receive
events published by other agents.  When a message arrives on a listened
topic the subscribing agent's decision loop is woken immediately instead
of waiting for its next ``decision_interval`` tick.

Usage (wired up by ``main.py``):

    bus = AgentBus()

    # security agent declares it publishes "person_detected"
    # porch-lighting agent subscribes to "person_detected"
    bus.subscribe("person_detected", porch_lighting_agent.on_bus_message)

    # security agent publishes after a detection
    await bus.publish(AgentMessage(
        topic="person_detected",
        sender_id="security",
        payload={"entity_id": "binary_sensor.front_door", "state": "on"},
    ))
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Type alias for subscriber callbacks.
Subscriber = Callable[["AgentMessage"], Awaitable[None]]


@dataclass
class AgentMessage:
    """A message sent over the agent bus."""

    topic: str
    sender_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class AgentBus:
    """
    Minimal async pub/sub event bus for agent-to-agent communication.

    * :meth:`subscribe` — register an async callback for a topic.
    * :meth:`unsubscribe` — remove a previously registered callback.
    * :meth:`publish` — deliver a message to all subscribers of a topic.

    All subscriber callbacks are awaited sequentially so that ordering is
    deterministic and backpressure is naturally applied.  If a subscriber
    raises an exception the error is logged and the remaining subscribers
    still receive the message.
    """

    def __init__(self) -> None:
        # topic → ordered list of async callbacks
        self._subscribers: Dict[str, List[Subscriber]] = {}

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    def subscribe(self, topic: str, callback: Subscriber) -> None:
        """Register *callback* to receive messages on *topic*."""
        self._subscribers.setdefault(topic, []).append(callback)
        logger.debug("AgentBus: subscribed %s to topic %r", callback, topic)

    def unsubscribe(self, topic: str, callback: Subscriber) -> None:
        """Remove *callback* from *topic*.  Silent no-op if not registered."""
        bucket = self._subscribers.get(topic)
        if bucket and callback in bucket:
            bucket.remove(callback)
            logger.debug("AgentBus: unsubscribed %s from topic %r", callback, topic)

    def subscribers(self, topic: str) -> List[Subscriber]:
        """Return a snapshot of current subscribers for *topic*."""
        return list(self._subscribers.get(topic, []))

    def all_topics(self) -> List[str]:
        """Return all topics that have at least one subscriber."""
        return [t for t, subs in self._subscribers.items() if subs]

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    async def publish(self, message: AgentMessage) -> int:
        """
        Deliver *message* to every subscriber of ``message.topic``.

        Returns the number of subscribers that were notified.
        """
        callbacks = list(self._subscribers.get(message.topic, []))
        notified = 0
        for cb in callbacks:
            try:
                await cb(message)
                notified += 1
            except Exception:
                logger.exception(
                    "AgentBus: subscriber error for topic %r", message.topic
                )
        if callbacks:
            logger.debug(
                "AgentBus: topic %r delivered to %d/%d subscribers",
                message.topic, notified, len(callbacks),
            )
        return notified
