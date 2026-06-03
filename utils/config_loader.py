"""
Shared YAML config loader for ADynamics training/analysis scripts.

Usage in any training script:
    from utils.config_loader import apply_yaml_defaults

    def parse_args():
        pre = argparse.ArgumentParser(add_help=False)
        pre.add_argument("--config", type=str, default=None)
        pre_args, _ = pre.parse_known_args()

        defaults = apply_yaml_defaults(pre_args.config, MAPPING)
        parser = argparse.ArgumentParser(parents=[pre])
        parser.set_defaults(**defaults)
        parser.add_argument(...)
        ...
"""

import os
from typing import Any, Dict, List, Optional, Tuple


def load_yaml_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load and return the full YAML config as a nested dict."""
    if not config_path or not os.path.exists(config_path):
        return {}
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for YAML config support. "
            "Install with: pip install pyyaml"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_yaml_defaults(
    config_path: Optional[str],
    mapping: List[Tuple[Tuple[str, ...], str]],
) -> Dict[str, Any]:
    """
    Apply YAML config values as argparse defaults.

    Args:
        config_path: Path to YAML config file (or None to skip).
        mapping: List of (yaml_key_path, argparse_arg_name) tuples.
                 E.g. (("model", "latent_channels"), "latent_channels") means
                 yaml cfg["model"]["latent_channels"] -> args.latent_channels.

    Returns:
        Dict of {argparse_arg_name: value} ready for parser.set_defaults(**d).
    """
    cfg = load_yaml_config(config_path)
    if not cfg:
        return {}

    defaults: Dict[str, Any] = {}
    for key_path, arg_name in mapping:
        val: Any = cfg
        try:
            for k in key_path:
                val = val[k]
        except (KeyError, TypeError):
            continue
        defaults[arg_name] = val
    return defaults


def merge_config(
    config_path: Optional[str],
    args: Any,
    mapping: List[Tuple[Tuple[str, ...], str]],
) -> Any:
    """
    Apply YAML config as defaults to an argparse Namespace.

    Returns a new Namespace with config values filled in for any arg
    that wasn't explicitly set on the CLI (i.e. was left at its argparse default).
    Useful when you want a single source of truth for both CLI overrides
    and programmatic config loading.
    """
    import copy
    defaults = apply_yaml_defaults(config_path, mapping)
    merged = copy.deepcopy(args)
    for k, v in defaults.items():
        if not hasattr(merged, k) or getattr(merged, k) is None:
            setattr(merged, k, v)
    return merged


def remap_labels_3class(data_list: List[Dict[str, Any]]) -> None:
    """
    Remap 4-class labels (NC=0, SCD=1, MCI=2, AD=3) to 3-class
    (NC=0, SCD+MCI=1, AD=2) in-place.

    Only remaps when labels are in the 4-class format. Idempotent
    (safe to call on already-remapped data).
    """
    for item in data_list:
        label = item.get("label", 0)
        if label == 1 or label == 2:
            item["label"] = 1
        elif label == 3:
            item["label"] = 2
