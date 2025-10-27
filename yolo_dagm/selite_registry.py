"""Helpers for registering custom SELite modules with Ultralytics YOLO."""

from __future__ import annotations

import contextlib
from typing import Any

import torch

import ultralytics.nn.tasks as yolo_tasks

from modules.se_layers import C2f_SELite

__all__ = ["register_selite_modules"]


def _lookup_module(name: str) -> type[torch.nn.Module]:
    """Resolve a module name from the Ultralytics registry or our custom layers."""

    if name in {"modules.C2f_SELite", "C2f_SELite"}:
        return C2f_SELite
    if "nn." in name:
        return getattr(torch.nn, name[3:])
    if "torchvision.ops." in name:
        return getattr(__import__("torchvision").ops, name[16:])
    return yolo_tasks.__dict__[name]


def _parse_model_with_selite(d: dict[str, Any], ch: int, verbose: bool = True):
    """Patched version of :func:`ultralytics.nn.tasks.parse_model` with SELite support."""

    import ast

    modules_dict = yolo_tasks.__dict__
    modules_dict.setdefault("modules.C2f_SELite", C2f_SELite)
    modules_dict.setdefault("C2f_SELite", C2f_SELite)

    legacy = True  # backward compatibility for v3/v5/v8/v9 models
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    scale = d.get("scale")
    if scales:
        if not scale:
            scale = next(iter(scales.keys()))
            yolo_tasks.LOGGER.warning("no model scale passed. Assuming scale='%s'.", scale)
        depth, width, max_channels = scales[scale]

    if act:
        yolo_tasks.Conv.default_act = eval(act)  # noqa: S307 - evaluated from trusted YAML
        if verbose:
            yolo_tasks.LOGGER.info("%s %s", yolo_tasks.colorstr("activation:"), act)

    if verbose:
        yolo_tasks.LOGGER.info(
            "\n%3s%20s%3s%10s  %-45s%-30s",
            "",
            "from",
            "n",
            "params",
            "module",
            "arguments",
        )

    ch_list = [ch]
    layers, save = [], []
    base_modules = frozenset(
        {
            yolo_tasks.Classify,
            yolo_tasks.Conv,
            yolo_tasks.ConvTranspose,
            yolo_tasks.GhostConv,
            yolo_tasks.Bottleneck,
            yolo_tasks.GhostBottleneck,
            yolo_tasks.SPP,
            yolo_tasks.SPPF,
            yolo_tasks.C2fPSA,
            yolo_tasks.C2PSA,
            yolo_tasks.DWConv,
            yolo_tasks.Focus,
            yolo_tasks.BottleneckCSP,
            yolo_tasks.C1,
            yolo_tasks.C2,
            yolo_tasks.C2f,
            yolo_tasks.C3k2,
            yolo_tasks.C2fAttn,
            yolo_tasks.C3,
            yolo_tasks.C3TR,
            yolo_tasks.C3Ghost,
            torch.nn.ConvTranspose2d,
            yolo_tasks.DWConvTranspose2d,
            yolo_tasks.C3x,
            yolo_tasks.RepC3,
            yolo_tasks.PSA,
            yolo_tasks.SCDown,
            yolo_tasks.C2fCIB,
            yolo_tasks.A2C2f,
            C2f_SELite,
        }
    )
    repeat_modules = frozenset(
        {
            yolo_tasks.BottleneckCSP,
            yolo_tasks.C1,
            yolo_tasks.C2,
            yolo_tasks.C2f,
            yolo_tasks.C3k2,
            yolo_tasks.C2fAttn,
            yolo_tasks.C3,
            yolo_tasks.C3TR,
            yolo_tasks.C3Ghost,
            yolo_tasks.C3x,
            yolo_tasks.RepC3,
            yolo_tasks.C2fPSA,
            yolo_tasks.C2fCIB,
            yolo_tasks.C2PSA,
            yolo_tasks.A2C2f,
            C2f_SELite,
        }
    )

    def _get_channels(index: int | list[int] | tuple[int, ...]) -> int:
        """Return the channel count for a layer reference handling list inputs."""

        if isinstance(index, (list, tuple)):
            if not index:
                raise ValueError("Layer reference list cannot be empty")
            index = index[-1]
        return ch_list[index]

    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        module_cls = _lookup_module(m)
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n_ = n
        if n > 1:
            n_ = max(round(n * depth), 1)

        if module_cls in base_modules:
            c1, c2 = _get_channels(f), args[0]
            if c2 != nc:
                c2 = yolo_tasks.make_divisible(min(c2, max_channels) * width, 8)
            if module_cls is yolo_tasks.C2fAttn:
                args[1] = yolo_tasks.make_divisible(min(args[1], max_channels // 2) * width, 8)
                args[2] = int(
                    max(
                        round(min(args[2], max_channels // 2 // 32)) * width,
                        1,
                    )
                    if args[2] > 1
                    else args[2]
                )
            args = [c1, c2, *args[1:]]
            if module_cls in repeat_modules:
                args.insert(2, n_)
                n_ = 1
            if module_cls is yolo_tasks.C3k2:
                legacy = False
                if scale in "mlx":
                    args[3] = True
            if module_cls is yolo_tasks.A2C2f:
                legacy = False
                if scale in "lx":
                    args.extend((True, 1.2))
        elif module_cls is yolo_tasks.AIFI:
            args = [_get_channels(f), *args]
        elif module_cls in {yolo_tasks.HGStem, yolo_tasks.HGBlock}:
            c1, cm, c2 = _get_channels(f), args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if module_cls is yolo_tasks.HGBlock:
                args.insert(4, n_)
                n_ = 1
        elif module_cls is yolo_tasks.ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif module_cls is torch.nn.BatchNorm2d:
            args = [_get_channels(f)]
            c2 = _get_channels(f)
        elif module_cls is yolo_tasks.Concat:
            c2 = sum(ch_list[x] for x in f)
        elif module_cls in {
            yolo_tasks.Detect,
            yolo_tasks.WorldDetect,
            yolo_tasks.YOLOEDetect,
            yolo_tasks.Segment,
            yolo_tasks.YOLOESegment,
            yolo_tasks.Pose,
            yolo_tasks.OBB,
            yolo_tasks.ImagePoolingAttn,
            yolo_tasks.v10Detect,
        }:
            args.append([ch_list[x] for x in f])
            if module_cls in {yolo_tasks.Segment, yolo_tasks.YOLOESegment}:
                args[2] = yolo_tasks.make_divisible(min(args[2], max_channels) * width, 8)
            if module_cls in {
                yolo_tasks.Detect,
                yolo_tasks.YOLOEDetect,
                yolo_tasks.Segment,
                yolo_tasks.YOLOESegment,
                yolo_tasks.Pose,
                yolo_tasks.OBB,
            }:
                module_cls.legacy = legacy
            c2 = _get_channels(f)
        elif module_cls is yolo_tasks.RTDETRDecoder:
            args.insert(1, [ch_list[x] for x in f])
            c2 = _get_channels(f)
        elif module_cls is yolo_tasks.CBLinear:
            c2 = args[0]
            c1 = _get_channels(f)
            args = [c1, c2, *args[1:]]
        elif module_cls is yolo_tasks.CBFuse:
            c2 = ch_list[f[-1]]
        elif module_cls in {yolo_tasks.TorchVision, yolo_tasks.Index}:
            c2 = args[0]
            args = [*args[1:]]
        else:
            c2 = _get_channels(f)

        module = (
            torch.nn.Sequential(*(module_cls(*args) for _ in range(n_)))
            if n_ > 1
            else module_cls(*args)
        )
        module.np = sum(x.numel() for x in module.parameters())
        module.i, module.f, module.type = i, f, str(module)[8:-2].replace("__main__.", "")
        if verbose:
            yolo_tasks.LOGGER.info(
                "%3d%20s%3d%10.0f  %-45s%-30s",
                i,
                f,
                n_,
                module.np,
                module.type,
                args,
            )
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(module)
        if i == 0:
            ch_list = []
        ch_list.append(c2)

    return torch.nn.Sequential(*layers), sorted(save)


def register_selite_modules() -> None:
    """Patch Ultralytics so that YAMLs can reference ``modules.C2f_SELite``."""

    if getattr(register_selite_modules, "_done", False):
        return

    yolo_tasks.parse_model = _parse_model_with_selite  # type: ignore[assignment]
    register_selite_modules._done = True

