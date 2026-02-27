from __future__ import annotations

import os


def _register_torch_safe_globals() -> None:
    # PyTorch 2.6+ defaults to weights_only=True for some loads. Pyannote
    # checkpoints may reference OmegaConf container classes, so allow-list them.
    try:
        import torch
        from omegaconf import DictConfig, ListConfig

        torch.serialization.add_safe_globals([ListConfig, DictConfig])
    except Exception:
        # Best-effort registration; if unavailable we continue and let the
        # downstream load raise a concrete error.
        return


def main() -> None:
    os.environ.setdefault("HF_HOME", "/app/models/huggingface")
    os.environ.setdefault("TORCH_HOME", "/app/models/torch")
    os.environ.setdefault("XDG_CACHE_HOME", "/app/models")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/app/models/huggingface")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/app/models/huggingface/hub")

    import torch
    import whisperx
    _register_torch_safe_globals()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = os.environ.get("WHISPERX_MODEL", "large-v3-turbo")
    compute_type = os.environ.get("WHISPERX_COMPUTE_TYPE", "float16")
    whisperx.load_model(
        model_name,
        device,
        compute_type=compute_type,
        language=None,
        task="transcribe",
    )

    align_langs = os.environ.get("WHISPERX_ALIGN_LANGS", "en").split(",")
    for lang in [lang_code.strip() for lang_code in align_langs if lang_code.strip()]:
        whisperx.load_align_model(language_code=lang, device=device, model_name=None)


if __name__ == "__main__":
    main()
