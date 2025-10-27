"""Register custom SELite modules with Ultralytics YOLO."""

from __future__ import annotations

import ultralytics.nn.tasks as yolo_tasks

from modules.se_layers import C2f_SELite

__all__ = ["register_selite_modules"]


def register_selite_modules() -> None:
    """Expose the SELite layers so Ultralytics YAML parsing can find them."""

    if getattr(register_selite_modules, "_done", False):
        return

    modules_dict = yolo_tasks.__dict__
    modules_dict.setdefault("modules.C2f_SELite", C2f_SELite)
    modules_dict.setdefault("C2f_SELite", C2f_SELite)
    setattr(yolo_tasks, "C2f_SELite", C2f_SELite)

    register_selite_modules._done = True
