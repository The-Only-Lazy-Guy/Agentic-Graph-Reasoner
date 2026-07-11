"""LGGN v3 total-memory package — three disk-backed layers:

  L0 syntax   (v5.memory.syntax)   — symbols/signatures/identifiers
  L1 episodic (v5.memory.episodic) — verified implementation records
  L2 semantic (v5.memory.semantic) — concept nodes (GraphEditEngine lifecycle)

`v5.memory.memory.TotalMemory` is the facade the agent loop talks to:
read(ctx) -> MemoryHit (concept-mediated impl retrieval), write(outcome) -> lifecycle updates.
"""
