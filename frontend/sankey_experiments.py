"""Plotly Sankey diagram generator for concept -> class flows."""

from __future__ import annotations

import re
from typing import Sequence

import plotly.graph_objects as go

DEFAULT_CONCEPT_COLORS = [
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#e45756",
    "#72b7b2",
    "#b279a2",
    "#ff9da6",
    "#9d755d",
    "#bab0ab",
    "#5f6b6d",
]

DEFAULT_CLASS_COLORS = [
    "#1f2937",
    "#2563eb",
    "#16a34a",
    "#d97706",
    "#9333ea",
    "#dc2626",
]


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    if len(color) == 3:
        color = "".join(ch * 2 for ch in color)
    if len(color) < 6:
        raise ValueError(f"Unsupported hex color: #{color}")
    return (int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16))


def _color_with_opacity(color: str, alpha: float) -> str:
    """Return an rgba(...) color using the given alpha when possible."""

    color = color.strip()
    if color.startswith("#"):
        r, g, b = _hex_to_rgb(color)
        return f"rgba({r},{g},{b},{alpha})"
    if color.startswith("rgb("):
        nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", color)
        if len(nums) >= 3:
            r, g, b = (int(float(nums[0])), int(float(nums[1])), int(float(nums[2])))
            return f"rgba({r},{g},{b},{alpha})"
    if color.startswith("rgba("):
        nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", color)
        if len(nums) >= 3:
            r, g, b = (int(float(nums[0])), int(float(nums[1])), int(float(nums[2])))
            return f"rgba({r},{g},{b},{alpha})"
    return color


def generate_sankey(
    concepts: Sequence[str],
    classes: Sequence[str],
    flows: Sequence[Sequence[float]],
    concept_colors: Sequence[str] | None = None,
    class_colors: Sequence[str] | None = None,
    link_opacity: float = 0.35,
) -> go.Figure:
    """Create a left-to-right Sankey from concepts to classes.

    flows is a matrix with shape (len(concepts), len(classes)).
    """

    # Validate the flow matrix shape.
    if len(flows) != len(concepts):
        raise ValueError("flows rows must match number of concepts")

    labels = list(concepts) + list(classes)
    source: list[int] = []
    target: list[int] = []
    value: list[float] = []

    # Build the link list by flattening the concept->class matrix.
    for i, row in enumerate(flows):
        if len(row) != len(classes):
            raise ValueError("each flow row must match number of classes")
        for j, v in enumerate(row):
            if v == 0:
                continue
            source.append(i)
            target.append(len(concepts) + j)
            value.append(float(v))

    # Resolve palettes and map one color per node.
    if concept_colors is None:
        concept_colors = go.Figure().layout.template.layout.colorway or []
    if not concept_colors:
        concept_colors = DEFAULT_CONCEPT_COLORS

    if class_colors is None:
        class_colors = DEFAULT_CLASS_COLORS

    concept_colors = list(concept_colors)
    class_colors = list(class_colors)

    concept_color_list = [
        concept_colors[i % len(concept_colors)] for i in range(len(concepts))
    ]
    class_color_list = [
        class_colors[i % len(class_colors)] for i in range(len(classes))
    ]

    node_colors = concept_color_list + class_color_list
    link_colors = [
        _color_with_opacity(concept_color_list[src], link_opacity) for src in source
    ]

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=18,
                    thickness=18,
                    line=dict(color="black", width=0.5),
                    label=labels,
                    color=node_colors,
                ),
                link=dict(source=source, target=target, value=value, color=link_colors),
            )
        ]
    )
    # White background for easy export into docs and slides.
    fig.update_layout(
        title_text="Concept -> Classes",
        font_size=11,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="black"),
    )
    return fig
