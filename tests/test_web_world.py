"""The mock desktop has to be honest about itself.

Two properties are load-bearing:

1. every episode the corpus writes can be replayed in the world it describes
   (otherwise the model is trained on a transcript the runtime would reject);
2. the oracle, which knows the labels, can finish the task (otherwise a model
   score of 0 means the world is broken, not that the model is bad).
"""

from __future__ import annotations

import random
import re

import pytest

import neurobrain.web_world as W


@pytest.fixture()
def recording(monkeypatch):
    made = []
    original = W.WebTask

    class Rec(original):  # type: ignore[misc, valid-type]
        def __init__(self, rng):
            super().__init__(rng)
            made.append(self)

    monkeypatch.setattr(W, "WebTask", Rec)
    return made


def test_episodes_replay_in_the_world(recording):
    for seed in range(60):
        recording.clear()
        rng = random.Random(seed)
        ep = W.web_episode(rng)
        task = recording[0]
        world = W.WebWorld(task)
        for x, y in re.findall(r"<\|act\|>click\((\d+),(\d+)\)", ep):
            world.act({"type": "click"}, (int(x), int(y)))
        assert world.saved, f"episode {seed} did not save"
        assert world.doc_text == task.answer, f"episode {seed} doc {world.doc_text!r}"


def test_oracle_finishes_every_task():
    for seed in range(30):
        task = W.WebTask(random.Random(seed))
        world = W.WebWorld(task)
        for label in ("chrome", task.topic, "copy", "docs", "paste", "save"):
            el = next((e for e in world.view().elements if e.label == label), None)
            assert el is not None, f"{label} missing for seed {seed}"
            world.act({"type": "click"}, (el.cx, el.cy))
        assert world.saved and world.doc_text == task.answer


def test_recovery_paths_exist():
    """A wrong result must be recoverable, or the policy can dead-end."""
    task = W.WebTask(random.Random(3))
    world = W.WebWorld(task)
    results = W.scan_labels(world)
    for label in ("chrome", [r for r in task.results if r != task.topic][0]):
        el = next(e for e in world.view().elements if e.label == label)
        world.act({"type": "click"}, (el.cx, el.cy))
    assert world.page == "article" and world.article != task.topic
    # the screen must differ from the results page, or the model cannot see it
    assert W.scan_labels(world) != results, "a wrong result left the screen unchanged"
    for label in ("back", task.topic, "copy", "docs", "paste", "save"):
        el = next((e for e in world.view().elements if e.label == label), None)
        assert el is not None, f"cannot get back to {label} after a wrong result"
        world.act({"type": "click"}, (el.cx, el.cy))
    assert world.saved and world.doc_text == task.answer


def test_scan_labels_are_unique_per_screen():
    from brainbox.lm_drive import scan_from_obs

    for seed in range(20):
        task = W.WebTask(random.Random(seed))
        world = W.WebWorld(task)
        for label in ("chrome", task.topic, "copy", "docs", "paste"):
            scan = scan_from_obs(world.observe())
            items = [x.rsplit(" ", 1)[0] for x in scan.split(": ", 1)[1].split("; ")]
            assert len(items) == len(set(items)), f"duplicate labels on screen: {items}"
            el = next((e for e in world.view().elements if e.label == label), None)
            if el is not None:
                world.act({"type": "click"}, (el.cx, el.cy))
