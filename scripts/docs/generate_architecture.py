"""
Generate architecture diagrams for Threat-aware Swarm (architecture-as-code).
"""

from __future__ import annotations

import base64
import os
import re
import sys
from pathlib import Path


def _ensure_out_dir() -> str:
    out_dir = os.path.join("docs", "images")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DIAGRAM_FORMAT = os.getenv("DIAGRAM_FORMAT", "png")
INLINE_IMAGES = os.getenv("DIAGRAM_INLINE_IMAGES", "1") == "1"

GRAPH_ATTR = {
    "dpi": "300",
    "pad": "0.5",
    "splines": "spline",
    "nodesep": "0.8",
    "ranksep": "1.0",
    "fontname": "Arial",
    "fontsize": "16",
    "bgcolor": "transparent",
}

NODE_ATTR = {
    "fontname": "Arial",
    "fontsize": "13",
}

EDGE_ATTR = {
    "fontname": "Arial",
    "fontsize": "11",
    "color": "#555555",
}


_IMAGE_HREF_RE = re.compile(r'xlink:href="([^"]+)"')


def _inline_svg_images(svg_path: str) -> None:
    svg_file = Path(svg_path)
    if not svg_file.exists():
        return
    text = svg_file.read_text(encoding="utf-8")
    replaced = False

    def _replace(match: re.Match[str]) -> str:
        nonlocal replaced
        href = match.group(1)
        if href.startswith("data:"):
            return match.group(0)
        img_path = Path(href)
        if not img_path.is_file():
            return match.group(0)
        suffix = img_path.suffix.lower().lstrip(".")
        mime = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "svg": "image/svg+xml",
        }.get(suffix)
        if mime is None:
            return match.group(0)
        data = img_path.read_bytes()
        encoded = base64.b64encode(data).decode("ascii")
        replaced = True
        return f'xlink:href="data:{mime};base64,{encoded}"'

    updated = _IMAGE_HREF_RE.sub(_replace, text)
    if replaced:
        svg_file.write_text(updated, encoding="utf-8")


def _merge_graph_attr(overrides: dict[str, str]) -> dict[str, str]:
    merged = GRAPH_ATTR.copy()
    merged.update(overrides)
    return merged


def generate_mlops_pipeline() -> None:
    from diagrams import Cluster, Diagram, Edge, Node
    from diagrams.generic.compute import Rack
    from diagrams.generic.network import Switch
    from diagrams.onprem.compute import Server
    from diagrams.onprem.container import Docker
    from diagrams.onprem.database import PostgreSQL
    from diagrams.programming.language import Python

    out_path = "docs/images/mlops_pipeline"
    graph_attr = _merge_graph_attr(
        {
            "splines": "spline",
            "nodesep": "0.8",
            "ranksep": "1.1",
            "concentrate": "false",
            "newrank": "true",
        }
    )
    with Diagram(
        "MLOps Pipeline",
        show=False,
        filename=out_path,
        outformat=DIAGRAM_FORMAT,
        direction="LR",
        graph_attr=graph_attr,
        node_attr=NODE_ATTR,
        edge_attr=EDGE_ATTR,
    ):
        with Cluster("Configuration & Launch"):
            hydra = Switch("Hydra Configs")
            launcher = Python("Train CLI")
            hydra >> launcher

        with Cluster("Execution Environment (Docker)"):
            container = Docker("Trainer Container")
            with Cluster("Vectorized Environments"):
                vec_workers = [Server("Worker 1"), Server("Worker 2"), Server("Worker N")]
            launcher >> Edge(label="Submits task", color="#2b6cb0", penwidth="2.0", weight="2") >> container
            container >> Edge(color="#2b6cb0", weight="2") >> vec_workers[1]
            vec_workers[0] >> Edge(style="invis") >> vec_workers[1] >> Edge(style="invis") >> vec_workers[2]

        with Cluster("Tracking & Artifacts"):
            tb = Switch("TensorBoard")
            mlflow = PostgreSQL("MLflow Registry")
            clearml = Rack("ClearML")
            hub = Node("", shape="point", width="0.01", height="0.01")
            container >> Edge(color="#2b6cb0", penwidth="1.5", weight="2") >> hub
            hub >> Edge(label="Logs Final Model", color="#2f855a", penwidth="2.0", constraint="false") >> mlflow
            hub >> Edge(label="Live Metrics", color="#2b6cb0", penwidth="1.5", constraint="false") >> clearml
            hub >> Edge(style="dashed", color="#a0aec0", constraint="false") >> tb
    if DIAGRAM_FORMAT == "svg" and INLINE_IMAGES:
        _inline_svg_images(f"{out_path}.svg")


def generate_system_architecture() -> None:
    from diagrams import Cluster, Diagram, Edge
    from diagrams.generic.compute import Rack
    from diagrams.generic.network import Switch
    from diagrams.onprem.compute import Server
    from diagrams.programming.language import Python

    out_path = "docs/images/system_architecture"
    graph_attr = _merge_graph_attr(
        {
            "splines": "spline",
            "nodesep": "0.8",
            "ranksep": "1.3",
            "newrank": "true",
        }
    )
    with Diagram(
        "System Logic Architecture",
        show=False,
        filename=out_path,
        outformat=DIAGRAM_FORMAT,
        direction="TB",
        graph_attr=graph_attr,
        node_attr=NODE_ATTR,
        edge_attr=EDGE_ATTR,
    ):
        with Cluster("RL Agent (Recurrent PPO)"):
            cnn = Rack("AdvancedSwarmCNN")
            lstm = Rack("LSTM Memory")
            policy = Python("Action Head")
            cnn >> lstm >> policy

        with Cluster("Threat-Aware Swarm Environment"):
            obs_builder = Switch("Observation Builder")
            physics = Server("Core Physics")
            threats = Server("Threat Engine")
            oracle = Switch("Oracle Manager")
            threats >> Edge(style="dotted", color="#e53e3e", constraint="false") >> physics
            physics >> obs_builder
            oracle >> Edge(label="Path Ratio", style="dashed", constraint="false") >> obs_builder

        policy >> Edge(label="Action (Velocity)", color="#e53e3e", penwidth="2.5", weight="2") >> physics
        obs_builder >> Edge(label="Dict Obs (41x41 + Vec)", color="#2b6cb0", penwidth="2.5", weight="2") >> cnn
    if DIAGRAM_FORMAT == "svg" and INLINE_IMAGES:
        _inline_svg_images(f"{out_path}.svg")


def generate_nn_topology() -> None:
    try:
        import torch
        from gymnasium import spaces
        from torchview import draw_graph

        from models.feature_extractors import AdvancedSwarmCNN
    except Exception as exc:
        print(f"[WARN] NN topology skipped: {exc}")
        return

    obs_space = spaces.Dict(
        {
            "vector": spaces.Box(-1.0, 1.0, shape=(8,)),
            "grid": spaces.Box(0.0, 1.0, shape=(1, 41, 41)),
        }
    )
    model = AdvancedSwarmCNN(obs_space, features_dim=512)
    dummy_grid = torch.zeros(1, 1, 41, 41)
    dummy_vec = torch.zeros(1, 8)
    draw_graph(
        model,
        input_data=({"grid": dummy_grid, "vector": dummy_vec},),
        save_graph=True,
        filename="docs/images/neural_network_topology",
        expand_nested=True,
        graph_dir="TB",
        hide_inner_tensors=False,
        hide_module_functions=False,
    )


def main() -> None:
    _ensure_out_dir()
    print("Generating MLOps Pipeline...")
    generate_mlops_pipeline()
    print("Generating System Architecture...")
    generate_system_architecture()
    print("Generating NN Topology (torchview)...")
    generate_nn_topology()
    print("Done. Images saved to docs/images/")


if __name__ == "__main__":
    main()
