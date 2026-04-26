"""Флаги оверлея для рендера UI и телеметрии."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OverlayFlags:
    show_grid: bool = False
    show_trails: bool = True
    show_threats: bool = True
    show_attention: bool = False
