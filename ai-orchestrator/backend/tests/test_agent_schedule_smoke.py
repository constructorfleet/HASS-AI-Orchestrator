"""Smoke tests for agent schedule checking (is_agent_active helper)."""
from __future__ import annotations

from datetime import datetime

import pytest

from agents.base_agent import is_agent_active, _parse_active_hours


# ---------------------------------------------------------------------------
# _parse_active_hours
# ---------------------------------------------------------------------------

class TestParseActiveHours:
    def test_simple(self):
        start, end = _parse_active_hours("06:00-09:00")
        assert start == 360
        assert end == 540

    def test_midnight_span(self):
        start, end = _parse_active_hours("22:00-06:00")
        assert start == 1320
        assert end == 360

    def test_invalid_format(self):
        with pytest.raises(ValueError):
            _parse_active_hours("0600-0900")


# ---------------------------------------------------------------------------
# is_agent_active — no schedule
# ---------------------------------------------------------------------------

class TestIsAgentActiveNoSchedule:
    def test_none_schedule_always_active(self):
        assert is_agent_active(None) is True

    def test_empty_dict_always_active(self):
        assert is_agent_active({}) is True


# ---------------------------------------------------------------------------
# is_agent_active — active_hours
# ---------------------------------------------------------------------------

class TestIsAgentActiveHours:
    # Tuesday 08:30
    _in_window = datetime(2026, 1, 6, 8, 30)
    # Tuesday 11:00
    _out_of_window = datetime(2026, 1, 6, 11, 0)

    def test_inside_window(self):
        assert is_agent_active({"active_hours": "06:00-09:00"}, self._in_window) is True

    def test_outside_window(self):
        assert is_agent_active({"active_hours": "06:00-09:00"}, self._out_of_window) is False

    def test_exact_start(self):
        dt = datetime(2026, 1, 6, 6, 0)
        assert is_agent_active({"active_hours": "06:00-09:00"}, dt) is True

    def test_exact_end(self):
        dt = datetime(2026, 1, 6, 9, 0)
        assert is_agent_active({"active_hours": "06:00-09:00"}, dt) is True

    def test_midnight_span_inside(self):
        # 23:30 → inside 22:00-06:00
        dt = datetime(2026, 1, 6, 23, 30)
        assert is_agent_active({"active_hours": "22:00-06:00"}, dt) is True

    def test_midnight_span_outside(self):
        # 12:00 → outside 22:00-06:00
        dt = datetime(2026, 1, 6, 12, 0)
        assert is_agent_active({"active_hours": "22:00-06:00"}, dt) is False


# ---------------------------------------------------------------------------
# is_agent_active — days
# ---------------------------------------------------------------------------

class TestIsAgentActiveDays:
    # Monday Jan 5 2026
    _monday = datetime(2026, 1, 5, 8, 0)
    # Saturday Jan 10 2026
    _saturday = datetime(2026, 1, 10, 8, 0)

    def test_active_on_listed_day(self):
        assert is_agent_active({"days": ["mon", "tue"]}, self._monday) is True

    def test_inactive_on_unlisted_day(self):
        assert is_agent_active({"days": ["mon", "tue"]}, self._saturday) is False

    def test_weekdays_shortcut_active(self):
        assert is_agent_active({"days": ["weekdays"]}, self._monday) is True

    def test_weekdays_shortcut_inactive_on_weekend(self):
        assert is_agent_active({"days": ["weekdays"]}, self._saturday) is False

    def test_weekends_shortcut_active(self):
        assert is_agent_active({"days": ["weekends"]}, self._saturday) is True

    def test_weekends_shortcut_inactive_on_weekday(self):
        assert is_agent_active({"days": ["weekends"]}, self._monday) is False

    def test_invalid_day_abbreviation(self):
        with pytest.raises(ValueError):
            is_agent_active({"days": ["xyz"]}, self._monday)


# ---------------------------------------------------------------------------
# is_agent_active — combined active_hours + days
# ---------------------------------------------------------------------------

class TestIsAgentActiveCombined:
    # Weekday (Monday) inside hours
    _mon_morning = datetime(2026, 1, 5, 7, 0)
    # Weekday (Monday) outside hours
    _mon_noon = datetime(2026, 1, 5, 12, 0)
    # Weekend (Saturday) inside hours
    _sat_morning = datetime(2026, 1, 10, 7, 0)

    _schedule = {"active_hours": "06:00-09:00", "days": ["mon", "tue", "wed", "thu", "fri"]}

    def test_active_weekday_morning(self):
        assert is_agent_active(self._schedule, self._mon_morning) is True

    def test_inactive_weekday_noon(self):
        assert is_agent_active(self._schedule, self._mon_noon) is False

    def test_inactive_weekend_morning(self):
        assert is_agent_active(self._schedule, self._sat_morning) is False


# ---------------------------------------------------------------------------
# is_agent_active — cron
# ---------------------------------------------------------------------------

class TestIsAgentActiveCron:
    def test_cron_matching(self):
        # "* 6-9 * * 1-5" → any minute, hours 6-9, Mon-Fri
        # Monday 7:30
        dt = datetime(2026, 1, 5, 7, 30)
        assert is_agent_active({"cron": "* 6-9 * * 1-5"}, dt) is True

    def test_cron_not_matching(self):
        # Monday at 11:00 — outside 6-9
        dt = datetime(2026, 1, 5, 11, 0)
        assert is_agent_active({"cron": "* 6-9 * * 1-5"}, dt) is False

    def test_invalid_cron_treated_as_active(self):
        # Bad cron must not crash; agent should default to active.
        dt = datetime(2026, 1, 5, 8, 0)
        assert is_agent_active({"cron": "not-valid-cron"}, dt) is True
