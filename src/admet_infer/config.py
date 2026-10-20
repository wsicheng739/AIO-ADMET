import json
import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = PACKAGE_ROOT / "configs" / "properties"


def _expand(value, context):
    if isinstance(value, dict):
        return {key: _expand(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item, context) for item in value]
    if not isinstance(value, str):
        return value

    text = os.path.expandvars(os.path.expanduser(value))

    # Resolve nested placeholders such as:
    # {project_root}/models -> {package_root}/../models -> absolute path.
    for _ in range(8):
        previous = text
        try:
            text = text.format(**context)
        except KeyError:
            break
        if text == previous:
            break

    return text


def load_property_config(property_name, config_path=None):
    path = Path(config_path) if config_path else DEFAULT_CONFIG_DIR / f"{property_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Property config not found: {path}")

    with open(path) as handle:
        config = json.load(handle)

    context = {
        "package_root": str(PACKAGE_ROOT),
        "property": property_name,
    }
    context.update(config)
    config = _expand(config, context)
    config["config_path"] = str(path.resolve())
    config.setdefault("name", property_name)
    return config


def apply_overrides(config, args):
    config = dict(config)

    if getattr(args, "model_root", None):
        config["model_root"] = str(Path(args.model_root).resolve())
    if getattr(args, "checkpoint_name", None):
        config["checkpoint_name"] = args.checkpoint_name
    if getattr(args, "pretrain_model_dir", None):
        config["pretrain_model_dir"] = str(Path(args.pretrain_model_dir).resolve())

    columns = dict(config.get("columns", {}))
    for argument, key in [
        ("id_col", "id"),
        ("smiles_col", "smiles"),
        ("task_col", "task"),
        ("value_col", "value"),
    ]:
        value = getattr(args, argument, None)
        if value:
            columns[key] = value
    config["columns"] = columns

    return config
