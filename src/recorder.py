from __future__ import annotations

import gymnasium as gym


class VideoRecorder:
    """Wraps a gym env with RecordVideo using a consistent naming convention."""

    def __init__(self, folder: str, record_every: int, prefix: str):
        self._folder = folder
        self._every = record_every
        self._prefix = prefix

    def wrap(self, env: gym.Env) -> gym.Env:
        return gym.wrappers.RecordVideo(
            env,
            video_folder=self._folder,
            episode_trigger=lambda ep: (ep + 1) % self._every == 0,
            name_prefix=self._prefix,
        )
