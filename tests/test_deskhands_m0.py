import os
import pytest
from deskhands import DesktopHands, ACTIONS


def test_twelve_actions_contract():
    assert ACTIONS == ("click", "double_click", "right_click", "move", "drag", "scroll", "type", "key", "wait", "focus_window", "launch", "screen")

@pytest.fixture(scope="module")
def hands():
    h = DesktopHands()
    yield h
    h.close()

@pytest.mark.parametrize("action,payload", [
    ("move", {"x": 10, "y": 10}),
    ("click", {"x": 10, "y": 10}),
    ("double_click", {"x": 10, "y": 10}),
    ("right_click", {"x": 10, "y": 10}),
    ("drag", {"x": 10, "y": 10, "x2": 30, "y2": 30, "ms": 20}),
    ("scroll", {"x": 10, "y": 10, "dy": 1}),
    ("type", {"text": "abc"}),
    ("key", {"keys": ["ctrl", "a"]}),
    ("wait", {"ms": 1}),
    ("focus_window", {"title_substr": "definitely-missing"}),
    ("launch", {"command": ["/bin/true"]}),
    ("screen", {}),
])
def test_action_result_shape(hands, action, payload):
    result = hands.perform({"type": action, **payload})
    assert set(result) == {"ok", "detail", "duration_ms"}
    assert isinstance(result["ok"], bool)
    assert isinstance(result["detail"], str)
    assert isinstance(result["duration_ms"], int) and result["duration_ms"] >= 0
    if action != "focus_window":
        assert result["ok"], result


def test_impossible_coordinates_refused(hands):
    for point in [(-1, 0), (0, -1), (10**9, 0), (0, 10**9)]:
        r = hands.perform({"type": "click", "x": point[0], "y": point[1]})
        assert not r["ok"]
