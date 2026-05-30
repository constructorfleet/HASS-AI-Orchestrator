"""Smoke tests for the AgentBus pub/sub message bus."""
from __future__ import annotations

import asyncio
from typing import List

import pytest

from agent_bus import AgentBus, AgentMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg(topic: str = "test", sender: str = "agent_a", **payload) -> AgentMessage:
    return AgentMessage(topic=topic, sender_id=sender, payload=payload)


# ---------------------------------------------------------------------------
# Basic subscribe / publish
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_publish_delivers_to_subscriber():
    bus = AgentBus()
    received: List[AgentMessage] = []

    async def handler(msg: AgentMessage):
        received.append(msg)

    bus.subscribe("sensor_alert", handler)
    await bus.publish(_msg(topic="sensor_alert", value=42))

    assert len(received) == 1
    assert received[0].topic == "sensor_alert"
    assert received[0].payload["value"] == 42


@pytest.mark.asyncio
async def test_publish_returns_subscriber_count():
    bus = AgentBus()

    async def h1(msg): pass
    async def h2(msg): pass

    bus.subscribe("t", h1)
    bus.subscribe("t", h2)
    n = await bus.publish(_msg("t"))
    assert n == 2


@pytest.mark.asyncio
async def test_publish_to_topic_with_no_subscribers_returns_zero():
    bus = AgentBus()
    n = await bus.publish(_msg("unknown_topic"))
    assert n == 0


@pytest.mark.asyncio
async def test_multiple_topics_isolated():
    bus = AgentBus()
    topic_a: List[str] = []
    topic_b: List[str] = []

    async def handler_a(msg): topic_a.append(msg.sender_id)
    async def handler_b(msg): topic_b.append(msg.sender_id)

    bus.subscribe("topic_a", handler_a)
    bus.subscribe("topic_b", handler_b)

    await bus.publish(_msg("topic_a", sender="alice"))
    await bus.publish(_msg("topic_b", sender="bob"))

    assert topic_a == ["alice"]
    assert topic_b == ["bob"]


# ---------------------------------------------------------------------------
# Unsubscribe
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unsubscribe_stops_delivery():
    bus = AgentBus()
    received: List[AgentMessage] = []

    async def handler(msg): received.append(msg)

    bus.subscribe("t", handler)
    await bus.publish(_msg("t"))
    assert len(received) == 1

    bus.unsubscribe("t", handler)
    await bus.publish(_msg("t"))
    assert len(received) == 1  # no new delivery


@pytest.mark.asyncio
async def test_unsubscribe_unknown_handler_is_noop():
    bus = AgentBus()

    async def orphan(msg): pass

    # Should not raise even though orphan was never subscribed.
    bus.unsubscribe("never_registered", orphan)


# ---------------------------------------------------------------------------
# Subscriber error isolation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_subscriber_exception_does_not_prevent_other_subscribers():
    bus = AgentBus()
    reached_second = []

    async def bad_handler(msg):
        raise RuntimeError("intentional failure")

    async def good_handler(msg):
        reached_second.append("yes")

    bus.subscribe("crash", bad_handler)
    bus.subscribe("crash", good_handler)

    # Should not propagate the exception.
    count = await bus.publish(_msg("crash"))

    # good_handler was still called.
    assert reached_second == ["yes"]
    # Only the successful subscriber counts.
    assert count == 1


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------

def test_all_topics_returns_topics_with_subscribers():
    bus = AgentBus()

    async def h(msg): pass

    bus.subscribe("a", h)
    bus.subscribe("b", h)
    assert set(bus.all_topics()) == {"a", "b"}


def test_subscribers_returns_snapshot():
    bus = AgentBus()

    async def h(msg): pass

    bus.subscribe("x", h)
    snap = bus.subscribers("x")
    assert h in snap


def test_subscribers_empty_topic_returns_empty_list():
    bus = AgentBus()
    assert bus.subscribers("nope") == []


# ---------------------------------------------------------------------------
# AgentMessage dataclass
# ---------------------------------------------------------------------------

def test_message_timestamp_set_automatically():
    msg = AgentMessage(topic="t", sender_id="s")
    assert msg.timestamp  # non-empty ISO string
    assert "T" in msg.timestamp  # ISO 8601 format


def test_message_payload_defaults_to_empty_dict():
    msg = AgentMessage(topic="t", sender_id="s")
    assert msg.payload == {}
