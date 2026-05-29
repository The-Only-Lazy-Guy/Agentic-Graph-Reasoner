import json

try:
    with open('artifacts/v4_difficulty_sweep/20260529_005312/03_trivial_vacuum_sound_paraphrase_2/packet.json', encoding='utf-8') as f:
        d = json.load(f)
    print("Fallback used:", d.get("controller_fallback_used"))
    print("Micro outcome:", d.get("micro_controller_outcome"))
except Exception as e:
    print("Error:", e)
