import numpy as np
import pytest
from app.core.config import Settings
from app.core.errors import KycError
from app.vision.liveness import Challenge


def metrics(ear=0.30, nose=0):
    return {
        "ear": ear,
        "ear_left": ear,
        "ear_right": ear,
        "nose_offset": nose,
        "brightness": 120,
        "sharpness": 100,
        "landmarks": [],
    }


def feed(ch, now, ear=0.30, nose=0):
    now += 0.5
    ch.advance(metrics(ear, nose), now)
    return now


def test_real_sequence_requires_close_open_and_both_turns():
    ch = Challenge(Settings(), now=0)
    ch.steps = [
        "NEUTRAL",
        "BLINK",
        "CENTER",
        "TURN_LEFT",
        "CENTER",
        "TURN_RIGHT",
        "CENTER",
    ]
    now = 0
    for _ in range(4):
        now = feed(ch, now)
    assert ch.baseline == 0.30 and ch.index == 1
    for _ in range(5):
        now = feed(ch, now)
    assert ch.index == 1  # An open eye does not count as a blink.
    for _ in range(2):
        now = feed(ch, now, ear=0.10)
    assert ch.closed and ch.index == 1
    for _ in range(2):
        now = feed(ch, now)
    assert ch.index == 2
    for _ in range(2):
        now = feed(ch, now)
    for _ in range(2):
        now = feed(ch, now, nose=0.23)
    for _ in range(2):
        now = feed(ch, now)
    for _ in range(2):
        now = feed(ch, now, nose=-0.23)
    for _ in range(2):
        now = feed(ch, now)
    assert ch.status == "PASSED" and len(ch.events) == 7


def test_noise_does_not_count_as_blink():
    ch = Challenge(Settings(), now=0)
    ch.steps = ["NEUTRAL", "BLINK", "CENTER"]
    now = 0
    for _ in range(4):
        now = feed(ch, now)
    for _ in range(5):
        now = feed(ch, now, 0.10)
        now = feed(ch, now, 0.30)
    assert not ch.closed and ch.index == 1


def test_closed_eyes_cannot_initialize():
    ch = Challenge(Settings(), now=0)
    for t in range(6):
        ch.advance(metrics(0.1), t)
    assert ch.index == 0 and ch.baseline is None


def test_replay_and_sequence():
    ch = Challenge(Settings(), now=0)
    img = np.full((10, 10, 3), 100, np.uint8)
    ch.accept_frame(img, 0, 0.2)
    with pytest.raises(KycError, match="sequence"):
        ch.accept_frame(img + 1, 4, 0.4)
    with pytest.raises(KycError, match="Repeated"):
        ch.accept_frame(img, 1, 0.5)
    assert ch.status == "FAILED"


def test_timeout():
    ch = Challenge(Settings(), now=0)
    with pytest.raises(KycError, match="timed out"):
        ch.accept_frame(np.zeros((10, 10, 3)), 0, 121)


def test_low_light_resets_partial():
    ch = Challenge(Settings(), now=0)
    ch.advance(metrics(), 0.5)
    m = metrics()
    m["brightness"] = 10
    ch.advance(m, 1)
    assert ch.streak == 0 and ch.index == 0
