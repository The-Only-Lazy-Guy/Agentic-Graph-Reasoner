"""
prepare_glm_model.py

Downloads GLM-4.7-Flash GGUF model for local inference with Answerer-v2.

Usage:
  python prepare_glm_model.py                # default: Q2_K
  python prepare_glm_model.py --quant Q2_K   # explicit
  python prepare_glm_model.py --quant Q4_K_M # higher quality, larger
  python prepare_glm_model.py --list         # list available quants

Quantization size guide for RTX 4050 6GB:
  Q2_K        11.34 GB   best for partial GPU offload (default)
  UD-IQ2_M    10.99 GB   slightly smaller, more aggressive
  Q3_K_M     ~13 GB      needs more CPU offload
  Q4_K_M     ~15 GB      CPU-only
"""

import argparse
from pathlib import Path

REPO_ID = "unsloth/GLM-4.7-Flash-GGUF"

AVAILABLE_QUANTS = [
    "Q2_K", "Q2_K_L", "Q3_K_S", "Q3_K_M", "Q4_0", "Q4_1",
    "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0",
    "IQ4_NL", "IQ4_XS", "MXFP4_MOE",
    "UD-IQ1_M", "UD-IQ1_S", "UD-IQ2_M", "UD-IQ2_XXS",
    "UD-IQ3_XXS", "UD-Q2_K_XL", "UD-Q3_K_XL",
]


def filename_for_quant(quant: str) -> str:
    return f"GLM-4.7-Flash-{quant}.gguf"


def list_quants() -> None:
    print(f"Available quantizations for {REPO_ID}:")
    print()
    print(f"  {'Quant':<20} {'File':<40}")
    print(f"  {'-'*20} {'-'*40}")
    for q in AVAILABLE_QUANTS:
        print(f"  {q:<20} {filename_for_quant(q):<40}")
    print()
    print("Recommended for RTX 4050 6GB:")
    print("  Q2_K      — best balance (11.3 GB, partial GPU offload)")
    print("  UD-IQ2_M  — more aggressive (11.0 GB)")
    print("  Q3_K_M    — medium quality, more CPU offload needed")
    print("  Q4_K_M    — good quality, CPU only (15+ GB)")


def download_model(quant: str, cache_dir: str) -> Path:
    fname = filename_for_quant(quant)
    local_path = Path(cache_dir) / fname

    if local_path.exists():
        size_gb = local_path.stat().st_size / 1e9
        print(f"Already exists: {local_path} ({size_gb:.2f} GB)")
        return local_path

    print(f"Downloading {REPO_ID}/{fname} ...")
    print(f"  -> {local_path}")
    print(f"  Size: ~{_estimate_size(quant):.1f} GB")
    print(f"  This may take a while depending on your connection.")
    print()

    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=REPO_ID,
        filename=fname,
        local_dir=cache_dir,
    )
    resolved = Path(downloaded)
    size_gb = resolved.stat().st_size / 1e9
    print(f"Downloaded: {resolved} ({size_gb:.2f} GB)")
    return resolved


def _estimate_size(quant: str) -> float:
    sizes = {
        "Q2_K": 11.34, "Q2_K_L": 11.42, "Q3_K_S": 10.5, "Q3_K_M": 13.0,
        "Q4_0": 14.0, "Q4_1": 15.5, "Q4_K_S": 14.0, "Q4_K_M": 15.0,
        "Q5_K_S": 17.0, "Q5_K_M": 17.5, "Q6_K": 20.0, "Q8_0": 26.0,
        "IQ4_NL": 13.0, "IQ4_XS": 12.0, "MXFP4_MOE": 12.0,
        "UD-IQ1_M": 7.0, "UD-IQ1_S": 6.0, "UD-IQ2_M": 10.99,
        "UD-IQ2_XXS": 9.0, "UD-IQ3_XXS": 10.0,
        "UD-Q2_K_XL": 11.89, "UD-Q3_K_XL": 13.5,
    }
    return sizes.get(quant, 12.0)


def verify_model(path: Path, quick: bool = True) -> bool:
    try:
        from llama_cpp import Llama
    except ImportError:
        print("llama-cpp-python not installed, skipping verification.")
        return True

    print(f"Verifying model with llama.cpp (n_gpu_layers=0, quick load)...")
    llm = Llama(
        model_path=str(path),
        n_gpu_layers=0,
        n_ctx=256,
        verbose=False,
    )
    result = llm.create_chat_completion(
        messages=[{"role": "user", "content": "Say OK"}],
        max_tokens=8,
        temperature=0,
    )
    text = result["choices"][0]["message"]["content"]
    if "ok" in text.lower():
        print(f"Model verified: {text.strip()}")
        return True
    print(f"Warning: unexpected output: {text[:100]}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GLM-4.7-Flash GGUF")
    parser.add_argument("--quant", type=str, default="Q2_K",
                        help=f"Quantization (default: Q2_K)")
    parser.add_argument("--cache-dir", type=str, default="models",
                        help="Download directory (default: models/)")
    parser.add_argument("--list", action="store_true",
                        help="List available quantizations")
    parser.add_argument("--verify", action="store_true",
                        help="Run quick inference test after download")
    args = parser.parse_args()

    if args.list:
        list_quants()
        return

    if args.quant not in AVAILABLE_QUANTS:
        print(f"Unknown quant: {args.quant}")
        print(f"Available: {', '.join(AVAILABLE_QUANTS)}")
        raise SystemExit(1)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    path = download_model(args.quant, str(cache_dir))

    if args.verify:
        verify_model(path)

    print()
    print(f"Model ready: {path}")
    print()
    print(f"To use with Answerer-v2:")
    print(f"  python answerer_v2.py \\")
    print(f"    --graph graphs/cs4.json \\")
    print(f"    --question 'Explain...' \\")
    print(f"    --controller local \\")
    print(f"    --model-path {path} \\")
    print(f"    --n-gpu-layers 20")
    print()

    gguf_filename = f"GLM-4.7-Flash-{args.quant}.gguf"
    if path.name != gguf_filename:
        symlink = path.parent / gguf_filename
        if not symlink.exists():
            try:
                symlink.symlink_to(path.name)
                print(f"Created symlink: {symlink} -> {path.name}")
            except OSError:
                pass


if __name__ == "__main__":
    main()
