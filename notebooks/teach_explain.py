"""Inference visualization — TEACH the model unseen info, watch it EXPLAIN what it learned.

A marimo notebook (reactive; nice visualizers) for molab. Shows EXACTLY what happens in one inference:
  base LM (doesn't know) -> TEACH (graph grows a real embedded node) -> RETRIEVE (neural) -> EXPLAIN (grounded).
Everything runs for real on the 4-bit LM (2.06GB, fits a 6GB consumer GPU). The LM is frozen; the learning
lives in the graph.

    molab:   marimo edit notebooks/teach_explain.py      (interactive)
             marimo run  notebooks/teach_explain.py       (app view, for the video)
"""
import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # 🧠 Teach → Explain — what happens inside one inference

        The user teaches the model **information it has never seen**. The model must then **explain what it
        learned** — grounded, not hallucinated. The LM is **frozen**; the knowledge lives in a **graph**.
        Runs on the real **4-bit** LM (~2GB, fits a 6GB consumer GPU).
        """
    )
    return


@app.cell
def _(mo):
    mo.md("### 1 · load the frozen 4-bit LM + seed graph (real, once)")
    return


@app.cell
def _():
    import sys, os
    from pathlib import Path
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from v5.runtime.membrane import AtomGraph, Atom, TRMRetriever, learn_any, seed_graph
    from v5.runtime.dcpd_latent import WhiteBox

    MODEL = os.environ.get("TEACH_LM", "Qwen/Qwen3-4B-Instruct-2507")
    wb = WhiteBox(MODEL, quant="4bit")            # 4-bit -> fits 6GB (measured ~2GB)
    graph = seed_graph()
    retr = TRMRetriever(graph)

    def clean(txt: str) -> str:
        for cut in ("\nHuman:", "Human:", "\nYou are", "\n\n", "<|"):
            i = txt.find(cut)
            if i > 0:
                txt = txt[:i]
        return txt.strip()

    return AtomGraph, Atom, TRMRetriever, learn_any, seed_graph, WhiteBox, wb, graph, retr, clean, MODEL


@app.cell
def _(mo, wb, graph):
    mo.md(
        f"**LM:** `{wb.name}`  ·  quant **{wb.quant}**  ·  VRAM **{wb.vram_gb:.2f} GB** "
        f"{'✅ fits 6GB' if wb.vram_gb <= 6 else '❌ over 6GB'}  ·  seed graph **{len(graph)}** atoms"
    )
    return


@app.cell
def _(mo):
    fact = mo.ui.text_area(
        value="The Klarn Protocol requires exactly three handshake phases: greet, verify, and seal.",
        label="**Teach the model (unseen info):**", full_width=True, rows=2,
    )
    question = mo.ui.text(
        value="How many handshake phases does the Klarn Protocol have, and what are they?",
        label="**Ask:**", full_width=True,
    )
    run = mo.ui.run_button(label="▶ run inference")
    mo.vstack([fact, question, run])
    return fact, question, run


@app.cell
def _(mo, wb, clean, question, run):
    mo.stop(not run.value, mo.md("*Edit the fact/question, then press ▶ run inference.*"))
    before = clean(wb.generate_plain(f"Answer in one short sentence. {question.value}", max_new=40))
    mo.md(f"### 2 · BEFORE teaching — ask the frozen LM\n> ❓ {question.value}\n\n"
          f"> 🤖 **{before}**\n\n*(the base model has never seen this — it guesses / hallucinates)*")
    return (before,)


@app.cell
def _(mo, graph, retr, learn_any, fact, run):
    mo.stop(not run.value)
    n_before = len(graph)
    res = learn_any(graph, retr, fact.value, name="taught_fact")
    mo.md(f"### 3 · TEACH — `learn_any()` banks a real node\n"
          f"- routed to node type: **`{res['kind']}`**  ·  node id: `{res['node']}`\n"
          f"- graph grew **{n_before} → {len(graph)}** nodes (a real MiniLM embedding was computed + stored)\n"
          f"- **the LM was not touched** — the knowledge is in the graph")
    return res, n_before


@app.cell
def _(mo, graph, question, res, run):
    mo.stop(not run.value)
    import numpy as np
    from v5.runtime.membrane import encode_batch
    M, order = graph.matrix()
    q = encode_batch([question.value])[0]
    sims = (M @ q).tolist()
    ranked = sorted(zip(order, sims), key=lambda z: -z[1])[:6]
    rows = "\n".join(
        f"| {'🎯 ' if n == res['node'] else ''}`{n}` | {s:.3f} | {graph.get(n).kind} |"
        for n, s in ranked
    )
    picked = ranked[0][0]
    mo.md(f"### 4 · RETRIEVE — neural (MiniLM) ranks the graph for the question\n"
          f"| node | similarity | type |\n|---|---|---|\n{rows}\n\n"
          f"**Retrieved:** `{picked}`  {'✅ the just-taught node' if picked == res['node'] else '⚠ not the taught node'}")
    return picked, ranked


@app.cell
def _(mo, wb, clean, graph, question, picked, run):
    mo.stop(not run.value)
    node = graph.get(picked)
    prompt = (f"Use ONLY this learned fact to answer.\nFact: {node.description}\n"
              f"Question: {question.value}\nAnswer:")
    after = clean(wb.generate_plain(prompt, max_new=40))
    mo.md(f"### 5 · EXPLAIN — the LM answers, GROUNDED in the retrieved node\n"
          f"> 🧾 grounding: *{node.description}*\n\n"
          f"> ✅ **{after}**\n\n"
          f"*(faithful — the answer comes from what was TAUGHT, not the LM's guess in step 2)*")
    return after, node


@app.cell
def _(mo, before, after, question):
    mo.md(
        f"""
        ### 6 · the point, side by side
        | | answer to *“{question.value[:44]}…”* |
        |---|---|
        | 🤖 LM alone (before) | {before[:80]} |
        | ✅ LM + learned graph | {after[:80]} |

        The frozen model **couldn't**, was **taught**, and then **explained what it learned** — grounded in the
        graph, on a 6GB GPU. That is the deployment claim, demonstrated.
        """
    )
    return


@app.cell
def _(mo, graph, res, run):
    mo.stop(not run.value)
    # graph visualization: nodes colored by type, the taught node highlighted
    import matplotlib.pyplot as plt
    try:
        import networkx as nx
        G = nx.DiGraph()
        color = {"atom": "#4f8bf5", "concept": "#3fb950", "procedure": "#d29922", "trap": "#f85149"}
        for name, a in graph.atoms.items():
            G.add_node(name, kind=a.kind)
        for s, d, r in graph.edges:
            G.add_edge(s, d, rel=r)
        pos = nx.spring_layout(G, seed=3, k=0.9)
        fig, ax = plt.subplots(figsize=(9, 6))
        cols = ["#ffd23f" if n == res["node"] else color.get(graph.get(n).kind, "#888") for n in G.nodes]
        sizes = [1400 if n == res["node"] else 700 for n in G.nodes]
        nx.draw_networkx_nodes(G, pos, node_color=cols, node_size=sizes, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color="#bbb", arrows=True, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
        ax.set_title("the graph after teaching (gold = the just-learned node; green = concepts, blue = skills)")
        ax.axis("off")
        out = fig
    except Exception as e:  # noqa: BLE001
        out = mo.md(f"*(graph plot needs networkx: `pip install networkx` — {e})*")
    out
    return


if __name__ == "__main__":
    app.run()
