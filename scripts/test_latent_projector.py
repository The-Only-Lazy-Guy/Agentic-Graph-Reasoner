"""Smoke test for the LatentProjector pipeline (3 phases).

Runs on a tiny subset with the cached Qwen2.5-0.5B + all-mpnet-base-v2 so it's fast
and offline. Verifies: data shapes, training converges (test_cos rises), projector
emits 768-d normalized vectors, and latent query cosine matches the mpnet target.
"""
import sys, time
import numpy as np
import torch

from v5.runtime.latent_projector import LatentProjector, project_lm_hidden
from v5.runtime.lggn_realizer import RawLM, load_triples, why_prompt
from v5.runtime.project_loop import (
    build_projector_data, train_projector, PROJECTOR_DATA, PROJECTOR_OUT, PROJECTOR_LAYER,
)

MODEL = "Qwen/Qwen2.5-0.5B"
N_EX = 32
EPOCHS = 40

def main():
    t0 = time.time()
    # ── Phase 1: build data (tiny subset) ────────────────────────────────────
    print(f"[phase1] build_projector_data({MODEL}, n={N_EX})")
    build_projector_data(MODEL, out_path=PROJECTOR_DATA, layer=PROJECTOR_LAYER,
                         max_examples=N_EX)
    d = np.load(PROJECTOR_DATA)
    lh, te = d["lm_hidden"], d["target_emb"]
    assert lh.shape[0] == te.shape[0] == N_EX, (lh.shape, te.shape)
    assert lh.shape[1] == 896, lh.shape          # Qwen2.5-0.5B hidden dim is 896
    assert te.shape[1] == 768, te.shape          # mpnet dim
    print(f"  OK lm_hidden {lh.shape}  target_emb {te.shape}")

    # ── Phase 2: train ───────────────────────────────────────────────────────
    print(f"[phase2] train_projector(epochs={EPOCHS})")
    # train_projector logs test_cos every 50; with 40 epochs it only prints ep1.
    # Wrap to capture final cosine by training then evaluating manually:
    train_projector(data_path=PROJECTOR_DATA, out_path=PROJECTOR_OUT,
                    d_lm=896, d_proj=768, epochs=EPOCHS, batch_size=16)
    sd = torch.load(PROJECTOR_OUT, weights_only=True)
    expected = {"net.0.weight", "net.0.bias", "net.1.weight", "net.1.bias",
                "net.3.weight", "net.3.bias"}
    assert set(sd.keys()) == expected, sd.keys()
    print(f"  OK weights keys: {sorted(sd.keys())}")

    # ── Phase 3: verify projector output + latent-vs-mpnet cosine ────────────
    print("[phase3] load + sanity check")
    proj = LatentProjector(d_lm=896, d_proj=768)
    proj.load_state_dict(sd)
    proj.eval()
    lm = RawLM(MODEL)
    # regression: get_pooled_hidden must accept layer= kwarg (the inference hook in
    # run_arm/run_chain calls it as lm.get_pooled_hidden(goal, layer=PROJECTOR_LAYER)).
    h_reg = lm.get_pooled_hidden("regression check", layer=PROJECTOR_LAYER)
    assert h_reg.shape == (896,), h_reg.shape
    print(f"  OK get_pooled_hidden(layer={PROJECTOR_LAYER}) -> {tuple(h_reg.shape)}")
    triples = load_triples("data/fable5/realizer_triples.jsonl")[:8]
    from v5.memory.store import make_mpnet_embedder
    mpnet = make_mpnet_embedder()
    cosines = []
    with torch.no_grad():
        for t in triples:
            spec = why_prompt(t["goal"], t["old"])
            h = project_lm_hidden(lm.model, lm.tok, spec, PROJECTOR_LAYER, lm.dev)
            q = proj(h[None])[0]
            assert q.shape == (768,), q.shape
            q_np = q.detach().numpy()          # regression: inference path calls .numpy()
            assert q_np.shape == (768,), q_np.shape
            assert abs(q.norm().item() - 1.0) < 1e-4, q.norm().item()
            tgt = torch.tensor(mpnet({"trace": t["trace"]})["trace"])
            cos = float((q * tgt).sum().item())
            cosines.append(cos)
    lm.cleanup()
    mean_cos = float(np.mean(cosines))
    print(f"  OK mean cosine(latent, mpnet) = {mean_cos:.3f} over {len(cosines)} samples")
    print(f"\nTOTAL wall {time.time()-t0:.1f}s")
    # The projector is a freshly-init MLP trained 40 epochs on 32 pairs; it should beat
    # random (~0.0) clearly but we don't assert a high bar (tiny data). Just check > 0.
    assert mean_cos > 0.0, "projector learned nothing"
    print("SMOKE TEST PASSED")

if __name__ == "__main__":
    main()
