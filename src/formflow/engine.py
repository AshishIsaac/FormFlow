"""Build the models once per process from the config: tracker (YOLOX + RTMPose) and the 3D lifter."""

from __future__ import annotations

from dataclasses import dataclass

from .data.download import download_model, have_model, model_path


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def pick_models(cfg: dict, mode: str = "offline") -> dict[str, str]:
    """Model names for this machine. ``live`` may name its own lifter (``live.lifter``)."""
    dev = resolve_device(cfg.get("device", "auto"))
    det = cfg.get("detector", "auto")
    lift = cfg.get("lifter", "auto")
    if mode == "live" and cfg.get("live", {}).get("lifter"):
        lift = cfg["live"]["lifter"]
    return {
        "device": dev,
        "detector": det if det != "auto" else ("yolox-m" if dev == "cuda" else "yolox-tiny"),
        "pose2d": cfg.get("pose2d", "rtmpose-m"),
        "lifter": lift if lift != "auto" else "motionbert-lite",
    }


@dataclass
class Engine:
    device: str
    names: dict[str, str]
    detector: object
    pose: object
    lifter: object

    def tracker(self, det_every: int = 10, async_detect: bool = False):
        from .pose2d.tracker import PoseTracker

        return PoseTracker(self.detector, self.pose, det_every=det_every, async_detect=async_detect)

    def describe(self) -> dict:
        return {"device": self.device, **self.names, "pose2d_device": getattr(self.pose, "device", "?")}


def build_engine(cfg: dict, mode: str = "live", download: bool = True) -> Engine:
    """``mode`` picks the lifter settings block (live | offline)."""
    from .lift.lifter import Lifter
    from .pose2d.models import YOLOX, RTMPose

    names = pick_models(cfg, mode)
    dev = names.pop("device")
    for n in names.values():
        if not have_model(n):
            if not download:
                raise FileNotFoundError(f"model {n} is missing: run `formflow models download`")
            download_model(n)
    sub = cfg.get(mode, {})
    lifter = Lifter(str(model_path(names["lifter"])), "full" if names["lifter"].endswith("full") else "lite",
                    device=dev, flip=bool(sub.get("flip", False)), half=bool(sub.get("half", False)), norm_mode="crop")
    ort_dev = "cuda" if dev == "cuda" else "cpu"
    det = YOLOX(str(model_path(names["detector"])), ort_dev)
    pose = RTMPose(str(model_path(names["pose2d"])), ort_dev)
    return Engine(dev, names, det, pose, lifter)
