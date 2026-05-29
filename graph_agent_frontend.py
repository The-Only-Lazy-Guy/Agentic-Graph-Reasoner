from __future__ import annotations

import html
import json
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from answerer_v2 import (
    DEFAULT_LLAMA_SERVER_URL,
    LlamaServerConfig,
    LlamaServerController,
    MockController,
    answer_query_v2_with_session,
)
from graph_core import MemoryGraph, canonical_relation


APP_TITLE = "Graph-Agent"
DEFAULT_SERVER_URL = DEFAULT_LLAMA_SERVER_URL
GRAPH_DIR = Path("graphs")

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="graph",
    layout="wide",
    initial_sidebar_state="expanded",
)


NODE_COLORS = {
    "question": "#111827",
    "evidence": "#2563EB",
    "plan_step": "#F59E0B",
    "note": "#0F766E",
    "conclusion": "#111827",
    "hypothesis": "#B45309",
    "unknown": "#6B7280",
}

EDGE_COLORS = {
    "support": "#10B981",
    "contradict": "#EF4444",
    "refine": "#14B8A6",
    "depend": "#F97316",
    "cause": "#F97316",
    "part_of": "#6366F1",
    "example_of": "#8B5CF6",
    "related": "#94A3B8",
}


def set_page_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #171717;
            --muted: #64615c;
            --hair: rgba(23, 23, 23, 0.10);
            --paper: #fbfaf7;
            --panel: rgba(255, 255, 255, 0.78);
            --panel-solid: #ffffff;
            --sand: #ede8dd;
            --clay: #c7784f;
            --sage: #7f9278;
            --blue: #356c9f;
            --green: #277466;
            --red: #b94b4b;
        }
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
                "SF Pro Text", "Segoe UI", sans-serif;
        }
        .stApp {
            background:
                radial-gradient(circle at 10% 5%, rgba(199, 120, 79, 0.18), transparent 32rem),
                radial-gradient(circle at 82% 10%, rgba(127, 146, 120, 0.16), transparent 30rem),
                linear-gradient(180deg, #fbfaf7 0%, #f4f0e8 100%);
            color: var(--ink);
        }
        [data-testid="stSidebar"] {
            background: rgba(250, 248, 243, 0.82);
            border-right: 1px solid var(--hair);
            backdrop-filter: blur(18px);
        }
        [data-testid="stHeader"] {
            background: rgba(251, 250, 247, 0.55);
            backdrop-filter: blur(12px);
        }
        h1, h2, h3 {
            letter-spacing: -0.045em;
        }
        .hero {
            position: relative;
            overflow: hidden;
            border: 1px solid var(--hair);
            border-radius: 34px;
            padding: 34px;
            min-height: 310px;
            background:
                linear-gradient(135deg, rgba(255,255,255,0.96), rgba(255,255,255,0.70)),
                radial-gradient(circle at 90% 18%, rgba(53, 108, 159, 0.16), transparent 22rem);
            box-shadow: 0 24px 80px rgba(58, 48, 33, 0.11);
        }
        .hero:after {
            content: "";
            position: absolute;
            inset: 0;
            background-image:
                linear-gradient(rgba(23,23,23,0.045) 1px, transparent 1px),
                linear-gradient(90deg, rgba(23,23,23,0.045) 1px, transparent 1px);
            background-size: 34px 34px;
            mask-image: linear-gradient(90deg, transparent 0%, black 45%, transparent 100%);
            pointer-events: none;
        }
        .eyebrow {
            color: var(--green);
            font-size: 0.80rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }
        .hero-title {
            max-width: 780px;
            margin: 10px 0 12px;
            font-size: clamp(2.4rem, 6vw, 5.6rem);
            line-height: 0.92;
            font-weight: 760;
            letter-spacing: -0.075em;
        }
        .hero-copy {
            max-width: 720px;
            color: var(--muted);
            font-size: 1.08rem;
            line-height: 1.65;
        }
        .glass-card {
            border: 1px solid var(--hair);
            border-radius: 24px;
            padding: 20px;
            background: var(--panel);
            box-shadow: 0 18px 48px rgba(58, 48, 33, 0.08);
            backdrop-filter: blur(16px);
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 14px;
            margin-top: 18px;
        }
        .metric-card {
            border: 1px solid var(--hair);
            border-radius: 22px;
            padding: 16px;
            background: rgba(255,255,255,0.78);
        }
        .metric-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 650;
        }
        .metric-value {
            margin-top: 6px;
            color: var(--ink);
            font-size: 1.65rem;
            font-weight: 760;
            letter-spacing: -0.04em;
        }
        .pipeline {
            display: grid;
            grid-template-columns: repeat(5, minmax(120px, 1fr));
            gap: 12px;
            margin: 18px 0;
        }
        .pipe-node {
            min-height: 150px;
            border: 1px solid var(--hair);
            border-radius: 24px;
            padding: 18px;
            background: rgba(255,255,255,0.74);
        }
        .pipe-index {
            width: 28px;
            height: 28px;
            border-radius: 999px;
            display: grid;
            place-items: center;
            background: #171717;
            color: #fff;
            font-weight: 700;
            font-size: 0.82rem;
        }
        .pipe-title {
            margin-top: 18px;
            font-weight: 760;
            font-size: 1.02rem;
            letter-spacing: -0.03em;
        }
        .pipe-copy {
            margin-top: 8px;
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.45;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border: 1px solid var(--hair);
            border-radius: 999px;
            background: rgba(255,255,255,0.78);
            color: var(--muted);
            font-size: 0.82rem;
            font-weight: 650;
            margin-right: 8px;
            margin-bottom: 8px;
        }
        .dot {
            width: 8px;
            height: 8px;
            border-radius: 99px;
            background: var(--green);
        }
        .trace-card {
            border: 1px solid var(--hair);
            border-radius: 18px;
            padding: 14px 16px;
            background: rgba(255,255,255,0.74);
            margin-bottom: 10px;
        }
        .trace-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
            margin-bottom: 8px;
        }
        .trace-action {
            font-weight: 760;
            letter-spacing: -0.02em;
        }
        .trace-phase {
            color: var(--muted);
            font-size: 0.82rem;
        }
        .trace-ok {
            color: var(--green);
            font-weight: 720;
        }
        .trace-fail {
            color: var(--red);
            font-weight: 720;
        }
        .small-muted {
            color: var(--muted);
            font-size: 0.86rem;
            line-height: 1.45;
        }
        .answer-card {
            border: 1px solid var(--hair);
            border-radius: 26px;
            padding: 22px;
            background: rgba(255,255,255,0.82);
            box-shadow: 0 18px 55px rgba(58, 48, 33, 0.09);
        }
        .soft-warning {
            border: 1px solid rgba(199, 120, 79, 0.28);
            border-radius: 18px;
            background: rgba(199, 120, 79, 0.09);
            padding: 12px 14px;
            color: #6f412b;
            font-size: 0.88rem;
            line-height: 1.5;
        }
        div[data-testid="stMetric"] {
            border: 1px solid var(--hair);
            border-radius: 18px;
            padding: 14px;
            background: rgba(255,255,255,0.72);
        }
        .stChatMessage {
            border-radius: 24px;
            border: 1px solid var(--hair);
            background: rgba(255,255,255,0.74);
        }
        @media (max-width: 960px) {
            .metric-grid, .pipeline {
                grid-template-columns: 1fr;
            }
            .hero {
                padding: 24px;
                border-radius: 26px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def safe_text(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def truncate(value: Any, limit: int = 160) -> str:
    text = str(value if value is not None else "")
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "..."


def read_doc_line(path: Path, prefix: str) -> str:
    if not path.exists():
        return "Unavailable"
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(prefix):
            return line.replace(prefix, "").strip().strip("`").strip()
    return "Unavailable"


def graph_paths() -> List[Path]:
    if not GRAPH_DIR.exists():
        return []
    return sorted(GRAPH_DIR.glob("*.json"))


@st.cache_data(show_spinner=False)
def load_graph_summary(path: str) -> Dict[str, Any]:
    graph = MemoryGraph.load_json(path)
    node_types = Counter(n.node_type for n in graph.nodes.values())
    edge_rels = Counter(canonical_relation(e.relation) for e in graph.edges)
    return {
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "node_types": dict(node_types),
        "edge_relations": dict(edge_rels),
    }


@st.cache_data(show_spinner=False)
def load_latest_novelty_summary() -> Dict[str, Any]:
    candidates = [
        Path("artifacts/novelty_local_sample_stage1.json"),
        Path("artifacts/novelty_local_sample_stage0.json"),
        Path("artifacts/novelty_local_sample.json"),
        Path("artifacts/novelty_mock.json"),
    ]
    for path in candidates:
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                summary = payload.get("summary", {}) or {}
                return {
                    "source": str(path),
                    "n": payload.get("n_rows", summary.get("n", 0)),
                    "pass_rate": summary.get("frac_novelty_pass"),
                    "graph_dep": summary.get("avg_graph_dependency"),
                    "anchor_quality": summary.get("avg_anchor_quality"),
                }
            except Exception:
                continue
    return {}


@st.cache_resource(show_spinner="Connecting to llama-server...")
def get_server_controller(
    server_url: str,
    temperature: float,
    max_tokens: int,
) -> LlamaServerController:
    return LlamaServerController(
        LlamaServerConfig(
            base_url=server_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    )


def reset_server_controller(controller: LlamaServerController) -> None:
    # The controller object is cached across reruns, but each user query
    # should start with a clean reasoning session and short history.
    controller._conversation_history = []
    controller._expanded_nodes = set()
    controller._last_action_result = ""


def make_controller(settings: Mapping[str, Any]):
    if settings["controller"] == "Llama Server":
        controller = get_server_controller(
            settings.get("server_url", DEFAULT_SERVER_URL),
            settings["temperature"],
            settings["max_tokens"],
        )
        reset_server_controller(controller)
        return controller
    return MockController()


def run_answer(question: str, graph_path: Path, settings: Mapping[str, Any]) -> Dict[str, Any]:
    graph = MemoryGraph.load_json(graph_path)
    controller = make_controller(settings)
    t0 = time.time()
    session, packet, trace = answer_query_v2_with_session(
        question=question,
        graph=graph,
        controller=controller,
        max_steps=int(settings["max_steps"]),
        k_anchors=int(settings["k_anchors"]),
        anchor_strategy=str(settings["anchor_strategy"]),
        graph_basename=graph_path.stem,
    )
    elapsed = time.time() - t0
    return {
        "question": question,
        "graph_path": str(graph_path),
        "graph_name": graph_path.stem,
        "session": session,
        "session_dict": session.to_dict(),
        "packet": packet,
        "trace": trace,
        "elapsed": elapsed,
        "settings": dict(settings),
    }


def relation_color(relation: str) -> str:
    return EDGE_COLORS.get(canonical_relation(relation), "#94A3B8")


def node_color(node_type: str) -> str:
    return NODE_COLORS.get(str(node_type or "unknown"), NODE_COLORS["unknown"])


def render_session_graph(session: Any, *, height: int = 690) -> None:
    try:
        from pyvis.network import Network
    except Exception as exc:
        st.warning(f"PyVis is not available, showing tables instead: {exc}")
        st.dataframe(pd.DataFrame([n.to_dict() for n in session.nodes.values()]), use_container_width=True)
        st.dataframe(pd.DataFrame([e.to_dict() for e in session.edges]), use_container_width=True)
        return

    try:
        net = Network(
            height=f"{height}px",
            width="100%",
            directed=True,
            bgcolor="#fbfaf7",
            font_color="#171717",
            cdn_resources="in_line",
        )
    except TypeError:
        net = Network(height=f"{height}px", width="100%", directed=True, bgcolor="#fbfaf7", font_color="#171717")

    net.set_options(
        """
        {
          "layout": { "improvedLayout": true },
          "physics": {
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
              "gravitationalConstant": -76,
              "centralGravity": 0.015,
              "springLength": 140,
              "springConstant": 0.055,
              "damping": 0.44
            },
            "stabilization": { "iterations": 180 }
          },
          "nodes": {
            "borderWidth": 1,
            "borderWidthSelected": 3,
            "font": { "size": 14, "face": "Segoe UI" },
            "shadow": { "enabled": true, "color": "rgba(0,0,0,0.10)", "size": 12, "x": 0, "y": 5 }
          },
          "edges": {
            "smooth": { "type": "dynamic" },
            "font": { "size": 11, "align": "middle", "face": "Segoe UI" },
            "arrows": { "to": { "enabled": true, "scaleFactor": 0.7 } }
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 80,
            "navigationButtons": true,
            "keyboard": true
          }
        }
        """
    )

    for node in session.nodes.values():
        ntype = str(getattr(node, "node_type", "unknown"))
        text = str(getattr(node, "text", ""))
        source = getattr(node, "source_memory_id", None)
        confidence = float(getattr(node, "confidence", 0.5))
        size = 18 + int(24 * max(0.0, min(confidence, 1.0)))
        shape = {
            "question": "box",
            "plan_step": "diamond",
            "conclusion": "box",
            "hypothesis": "triangle",
        }.get(ntype, "dot")
        label = f"{node.id}\n{truncate(text, 46)}"
        title = (
            f"<b>{safe_text(node.id)}</b><br>"
            f"{safe_text(text)}<br><br>"
            f"<i>type:</i> {safe_text(ntype)}<br>"
            f"<i>source:</i> {safe_text(source or '-') }<br>"
            f"<i>confidence:</i> {confidence:.2f}"
        )
        font_color = "#ffffff" if ntype in {"question", "evidence", "conclusion"} else "#171717"
        net.add_node(
            str(node.id),
            label=label,
            title=title,
            color={
                "background": node_color(ntype),
                "border": "rgba(23,23,23,0.28)",
                "highlight": {"background": "#c7784f", "border": "#171717"},
            },
            size=size,
            shape=shape,
            font={"color": font_color},
        )

    for edge in session.edges:
        src = str(getattr(edge, "src", ""))
        dst = str(getattr(edge, "dst", ""))
        if src not in session.nodes or dst not in session.nodes:
            continue
        rel = canonical_relation(str(getattr(edge, "relation", "related")))
        status = str(getattr(edge, "status", "draft"))
        confidence = float(getattr(edge, "confidence", 0.5))
        net.add_edge(
            src,
            dst,
            label=rel,
            title=(
                f"<b>{safe_text(rel)}</b><br>"
                f"{safe_text(src)} -> {safe_text(dst)}<br>"
                f"status: {safe_text(status)}<br>"
                f"confidence: {confidence:.2f}"
            ),
            color=relation_color(rel),
            width=max(1.2, 1.0 + confidence * 3.0),
            dashes=status not in {"active", "verified", "committed"},
        )

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as tmp:
        tmp_path = Path(tmp.name)
    try:
        net.save_graph(str(tmp_path))
        rendered = tmp_path.read_text(encoding="utf-8")
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
    components.html(rendered, height=height + 18, scrolling=False)


def render_action_flow(trace: Sequence[Mapping[str, Any]]) -> None:
    counts = Counter(str(e.get("action", "?")) for e in trace)
    if not counts:
        st.info("No controller actions recorded yet.")
        return
    pills = "".join(
        f'<span class="status-pill"><span class="dot"></span>{safe_text(action)}: {count}</span>'
        for action, count in counts.most_common()
    )
    st.markdown(pills, unsafe_allow_html=True)


def render_trace(trace: Sequence[Mapping[str, Any]]) -> None:
    st.markdown(
        '<div class="soft-warning">This panel exposes the model-provided public '
        'thought_summary, action arguments, executor results, and tool/action flow. '
        'It does not expose hidden private chain-of-thought.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    render_action_flow(trace)
    for entry in trace:
        result = entry.get("result", {}) or {}
        success = bool(result.get("success", entry.get("action") in {"FINALIZE_ANSWER", "STOP"}))
        status_class = "trace-ok" if success else "trace-fail"
        status_text = "OK" if success else "FAIL"
        step = safe_text(entry.get("step", "?"))
        phase = safe_text(entry.get("phase", "?"))
        action = safe_text(entry.get("action", "?"))
        summary = safe_text(result.get("summary", ""))
        thought = safe_text(entry.get("thought", ""))
        reason = safe_text(entry.get("reason", ""))
        args = json.dumps(entry.get("args", {}) or {}, ensure_ascii=False, indent=2)
        with st.expander(f"Step {step}: {phase} / {action} / {status_text}", expanded=False):
            st.markdown(
                f"""
                <div class="trace-card">
                    <div class="trace-top">
                        <div>
                            <div class="trace-action">{action}</div>
                            <div class="trace-phase">phase {phase} | step {step}</div>
                        </div>
                        <div class="{status_class}">{status_text}</div>
                    </div>
                    <div class="small-muted"><b>Public summary:</b> {thought or '-'}</div>
                    <div class="small-muted"><b>Reason:</b> {reason or '-'}</div>
                    <div class="small-muted"><b>Executor:</b> {summary or '-'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.code(args, language="json")
            if result:
                st.code(json.dumps(result, ensure_ascii=False, indent=2), language="json")


def session_metrics(run: Mapping[str, Any]) -> Dict[str, Any]:
    session = run["session"]
    packet = run["packet"]
    trace = run["trace"]
    rejected = sum(
        1
        for entry in trace
        if entry.get("action") in {"NOTE", "CONCLUDE"} and not (entry.get("result", {}) or {}).get("success", True)
    )
    return {
        "Steps": getattr(packet, "steps_taken", len(trace)),
        "Confidence": round(float(getattr(packet, "confidence", 0.0)), 3),
        "Nodes": len(session.nodes),
        "Edges": len(session.edges),
        "Paths": len(session.paths),
        "Rejected writes": rejected,
        "Elapsed": f"{run['elapsed']:.1f}s",
    }


def render_overview() -> None:
    current_stage = read_doc_line(Path("PROGRESS.md"), "**Current stage:**")
    current_focus = read_doc_line(Path("PROGRESS.md"), "**Current focus:**")
    active_controller = read_doc_line(Path("TASK_LIST.md"), "**Official active script:**")
    novelty = load_latest_novelty_summary()

    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">Deployment Interface</div>
            <div class="hero-title">A graph-native reasoning system you can inspect.</div>
            <div class="hero-copy">
                The current stack answers by building a temporary session graph:
                plan steps, retrieved evidence, notes, grounded conclusions, verified
                edges, and a deterministic graph readout. The UI is designed to make
                that state visible instead of hiding failures behind prose.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    col_a, col_b, col_c = st.columns([1.2, 1, 1])
    with col_a:
        st.markdown("### Current Status")
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="small-muted"><b>Stage</b><br>{safe_text(current_stage)}</div>
                <br>
                <div class="small-muted"><b>Focus</b><br>{safe_text(current_focus)}</div>
                <br>
                <div class="small-muted"><b>Reference executor</b><br>{safe_text(active_controller)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown("### Latest Novelty Sample")
        if novelty:
            st.metric("Rows", novelty.get("n", 0))
            if novelty.get("pass_rate") is not None:
                st.metric("Novelty pass", f"{float(novelty['pass_rate']) * 100:.0f}%")
            if novelty.get("graph_dep") is not None:
                st.metric("Graph dependency", f"{float(novelty['graph_dep']):.3f}")
            st.caption(novelty.get("source", ""))
        else:
            st.info("No novelty artifact found.")
    with col_c:
        st.markdown("### Visibility Contract")
        st.markdown(
            """
            <div class="glass-card">
                <span class="status-pill"><span class="dot"></span>Session graph</span>
                <span class="status-pill"><span class="dot"></span>Tool/action trace</span>
                <span class="status-pill"><span class="dot"></span>Evidence supports</span>
                <span class="status-pill"><span class="dot"></span>Executor failures</span>
                <p class="small-muted">
                The UI shows public action summaries and tool calls. It avoids exposing
                hidden private chain-of-thought while still making the protocol inspectable.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Model Flow")
    st.markdown(
        """
        <div class="pipeline">
            <div class="pipe-node">
                <div class="pipe-index">1</div>
                <div class="pipe-title">Question</div>
                <div class="pipe-copy">The user query enters as Q0 and defines the session objective.</div>
            </div>
            <div class="pipe-node">
                <div class="pipe-index">2</div>
                <div class="pipe-title">Anchor Retrieval</div>
                <div class="pipe-copy">Answerer-v2 pulls top-k relevant memory nodes using lexical plus MiniLM scores.</div>
            </div>
            <div class="pipe-node">
                <div class="pipe-index">3</div>
                <div class="pipe-title">Controller Actions</div>
                <div class="pipe-copy">The model emits JSON actions: PLAN, EXPAND_NODE, NOTE, CONCLUDE, VERIFY_EDGE.</div>
            </div>
            <div class="pipe-node">
                <div class="pipe-index">4</div>
                <div class="pipe-title">Validated Session Graph</div>
                <div class="pipe-copy">The executor rejects invalid supports, phase errors, unsupported claims, and unsafe contradictions.</div>
            </div>
            <div class="pipe-node">
                <div class="pipe-index">5</div>
                <div class="pipe-title">Graph Readout</div>
                <div class="pipe-copy">The final answer is composed from grounded conclusions, not from a hidden free-form prose pass.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Deployment Notes")
    st.markdown(
        """
        <div class="glass-card">
            <p class="small-muted">
            Use the Chat Lab page for live model runs. The default controller is Mock
            so the UI is responsive without an LLM. Switch to <em>Llama Server</em> after
            starting <code>llama-server</code> on <code>127.0.0.1:6767</code> with the GGUF model.
            Session graph visualization is generated per answer and does not dump the
            full memory graph unless you explicitly extend the app to do so.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat_page(settings: Mapping[str, Any], selected_graph: Optional[Path]) -> None:
    st.markdown("## Chat Lab")
    st.caption("Ask the graph-agent a question. After each response, inspect the session graph, public trace, and tool/action usage.")

    if selected_graph is None:
        st.error("No graph JSON files found under ./graphs.")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_run" not in st.session_state:
        st.session_state.last_run = None

    top_left, top_right = st.columns([1.3, 0.7])
    with top_left:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="small-muted"><b>Active graph</b></div>
                <h3 style="margin: 4px 0 0; letter-spacing: -0.04em;">{safe_text(selected_graph.name)}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_right:
        if st.button("Clear chat and graph", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_run = None
            st.rerun()

    if not st.session_state.messages:
        st.markdown(
            """
            <div class="soft-warning">
            Suggested prompts: "Why can light travel through space but sound cannot?",
            "Can Dijkstra be trusted with one negative edge?", or "Are heat and temperature the same quantity?"
            </div>
            """,
            unsafe_allow_html=True,
        )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask the graph-agent...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Running graph-reasoning controller..."):
                try:
                    run = run_answer(prompt, selected_graph, settings)
                    st.session_state.last_run = run
                    answer = str(run["packet"].answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.markdown(answer)
                except Exception as exc:
                    error = f"Run failed: {exc}"
                    st.session_state.messages.append({"role": "assistant", "content": error})
                    st.error(error)

    run = st.session_state.last_run
    if not run:
        return

    st.write("")
    st.markdown("## Run Inspector")
    metrics = session_metrics(run)
    metric_cols = st.columns(len(metrics))
    for col, (label, value) in zip(metric_cols, metrics.items()):
        col.metric(label, value)

    tab_answer, tab_graph, tab_trace, tab_tables, tab_raw = st.tabs(
        ["Answer", "Session Graph", "CoT / Tool Trace", "Graph Tables", "Raw JSON"]
    )
    with tab_answer:
        st.markdown(
            f"""
            <div class="answer-card">
                <div class="small-muted"><b>Question</b><br>{safe_text(run['question'])}</div>
                <hr style="border: 0; border-top: 1px solid rgba(23,23,23,0.10); margin: 16px 0;">
                <div>{safe_text(run['packet'].answer).replace(chr(10), '<br>')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with tab_graph:
        render_session_graph(run["session"])
    with tab_trace:
        render_trace(run["trace"])
    with tab_tables:
        session = run["session"]
        nodes = [n.to_dict() for n in session.nodes.values()]
        edges = [e.to_dict() for e in session.edges]
        st.markdown("#### Session Nodes")
        st.dataframe(pd.DataFrame(nodes), use_container_width=True, height=320)
        st.markdown("#### Session Edges")
        st.dataframe(pd.DataFrame(edges), use_container_width=True, height=260)
    with tab_raw:
        payload = {
            "question": run["question"],
            "graph": run["graph_path"],
            "answer": run["packet"].answer,
            "confidence": run["packet"].confidence,
            "steps_taken": run["packet"].steps_taken,
            "trace": run["trace"],
            "session": run["session_dict"],
        }
        st.code(json.dumps(payload, ensure_ascii=False, indent=2), language="json")


def sidebar_settings() -> Tuple[Dict[str, Any], Optional[Path], str]:
    st.sidebar.markdown("## Graph-Agent")
    page = st.sidebar.radio("Page", ["Presentation", "Chat Lab"], label_visibility="collapsed")

    paths = graph_paths()
    graph_label_to_path = {p.name: p for p in paths}
    selected_graph: Optional[Path] = None
    if paths:
        graph_name = st.sidebar.selectbox("Graph", list(graph_label_to_path.keys()), index=0)
        selected_graph = graph_label_to_path[graph_name]
        try:
            summary = load_graph_summary(str(selected_graph))
            st.sidebar.caption(f"{summary['nodes']} nodes | {summary['edges']} edges")
        except Exception as exc:
            st.sidebar.warning(f"Could not load graph summary: {exc}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Controller")
    controller = st.sidebar.selectbox("Mode", ["Mock", "Llama Server"], index=0)
    anchor_strategy = st.sidebar.selectbox("Anchor strategy", ["topk", "mmr", "legacy"], index=0)
    max_steps = st.sidebar.slider("Max steps", min_value=4, max_value=24, value=12, step=1)
    k_anchors = st.sidebar.slider("Anchors", min_value=3, max_value=16, value=8, step=1)

    with st.sidebar.expander("Llama Server settings", expanded=False):
        st.caption(
            "Start the server first (separate terminal):\n"
            "```\nllama-server -m cache/models/GLM-4.7-Flash-Q2_K.gguf "
            "-ngl 20 -c 4096 --host 127.0.0.1 --port 6767\n```"
        )
        server_url = st.text_input("Server URL (localhost only)", DEFAULT_SERVER_URL)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        max_tokens = st.slider("Max action tokens", min_value=128, max_value=1024, value=512, step=64)

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Trace visibility shows public action summaries and executor/tool calls, not hidden private chain-of-thought."
    )

    settings = {
        "controller": controller,
        "anchor_strategy": anchor_strategy,
        "max_steps": max_steps,
        "k_anchors": k_anchors,
        "server_url": server_url,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    return settings, selected_graph, page


def main() -> None:
    set_page_style()
    settings, selected_graph, page = sidebar_settings()
    if page == "Presentation":
        render_overview()
    else:
        render_chat_page(settings, selected_graph)


if __name__ == "__main__":
    main()
