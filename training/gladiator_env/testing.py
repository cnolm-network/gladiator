"""离线测试/自检用的假通道 B 连接。

它实现 `LinkClient` 的最小接口（`start/wait_welcome/latest/wait_observation/send_actions/stop`），
用于在没有 Minecraft 服务端时验证 Gymnasium 契约、观测/动作编码与回合结算。
"""

from __future__ import annotations

import time


def make_observation(
    username: str,
    *,
    x: float = 0.0,
    y: float = 65.0,
    z: float = 0.0,
    health: float = 20.0,
    state: str = "Alive",
    targets: list[dict] | None = None,
    tick: int = 0,
) -> dict:
    return {
        "tick": tick,
        "gameId": "",
        "state": state,
        "self": {
            "pos": {"x": x, "y": y, "z": z},
            "yaw": 0.0,
            "pitch": 0.0,
            "onGround": True,
            "health": health,
            "food": 20,
            "saturation": 5.0,
            "heldItem": {"name": "air", "count": 0, "slot": 0},
            "inventory": [],
        },
        "entities": targets or [],
        "blockWindow": {"origin": {"x": 0, "y": 0, "z": 0}, "size": [7, 4, 7], "palette": ["air"], "data": [0]},
        "restrictions": [],
        "chat": [],
        "pendingNotices": [],
    }


def make_target(entity_id: int, name: str, *, x: float = 3.0, y: float = 65.0, z: float = 0.0,
                distance: float | None = None) -> dict:
    if distance is None:
        distance = (x ** 2 + z ** 2) ** 0.5
    return {"id": entity_id, "kind": "player", "name": name, "pos": {"x": x, "y": y, "z": z},
            "distance": distance}


class FakeLink:
    control_hz = 20.0

    def __init__(self, username: str, observation: dict | None = None) -> None:
        self.username = username
        self.observation = observation or make_observation(username)
        self.actions: list[dict] = []
        self.started = False
        self.stopped = False

    def start(self) -> "FakeLink":
        self.started = True
        return self

    def wait_welcome(self, timeout: float | None = None) -> dict:
        return {
            "observationSpec": {"encoding": "json-v1", "fields": []},
            "actionSpec": {"encoding": "json-v1", "types": []},
            "controlHz": self.control_hz,
        }

    def control_hz_value(self) -> float:
        return self.control_hz

    def latest(self) -> dict:
        return self.observation

    def wait_observation(self, timeout: float | None = None, after: dict | None = None) -> dict:
        time.sleep(0.005)
        return self.observation

    def send_actions(self, actions: list[dict]) -> list[dict]:
        self.actions.extend(actions)
        return [{"ok": True} for _ in actions]

    def set_observation(self, observation: dict) -> None:
        self.observation = observation

    def stop(self) -> None:
        self.stopped = True
