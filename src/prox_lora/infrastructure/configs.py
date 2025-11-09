import functools
import importlib
from argparse import Namespace
from collections.abc import Mapping, MutableMapping
from copy import copy, replace
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, cast

import ruamel.yaml

from prox_lora.utils.other import raise_if_not_contained


def _initialize_yaml() -> ruamel.yaml.YAML:
    """YAML setup (style, tuple representation)."""
    yaml = ruamel.yaml.YAML(typ="safe")
    yaml.allow_unicode = True
    yaml.indent = 4
    yaml.default_flow_style = False
    yaml.sort_base_mapping_type_on_output = False  # type: ignore
    # Represent and parse tuples as `!tuple [a, b, c]`
    yaml.representer.add_representer(tuple, lambda r, o: r.represent_sequence("!tuple", o, flow_style=True))
    yaml.constructor.add_constructor("!tuple", lambda c, node: tuple(c.construct_sequence(node)))
    return yaml


yaml = _initialize_yaml()


def save_config(config: Any, path: Path) -> None:
    yaml.dump(config, path)


def load_config(path: Path) -> Any:
    return yaml.load(Path(path))


def flatten_dict(d: Any, /, delimiter: str = ".", *, flatten_lists: bool = False, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict/dataclass into a flat dict with delimiter-separated keys.

    For example, `flatten_dict({"a": {"b": "c"}})` returns `{"a.b": "c"}`.

    Dataclasses are seen as dicts with `asdict()`, Namespaces with `vars()`.

    If `flatten_lists` is True, lists are flattened as well, like:
        `{"a": [{"b": x}, {"b": y}]}` becomes
        `{"a[0].b": x, "a[1].b": y}`.
    """
    result = dict[str, Any]()
    for k, v in d.items():
        new_key = prefix + str(k)

        if isinstance(v, Mapping) or (is_dataclass(v) and not isinstance(v, type)):
            if is_dataclass(v) and not isinstance(v, type):
                v = asdict(v)
            elif isinstance(v, Namespace):
                v = vars(v)
            result |= flatten_dict(v, prefix=new_key + delimiter, delimiter=delimiter)
        # Also handle the case where v is a list of dictionaries
        elif flatten_lists and isinstance(v, list):
            for i, item in enumerate(v):
                result |= flatten_dict(item, prefix=f"{new_key}[{i}]{delimiter}", delimiter=delimiter)
        else:
            result[new_key] = v
    return result


def unflatten_dict(d: Mapping[str, Any], /, delimiter: str = ".") -> dict[str | int, Any]:
    """
    Unflatten a dict with delimiter-separated keys into a nested dict.

    For example, `unflatten_dict({"a.b": "c"})` returns `{"a": {"b": "c"}}`.

    Supports integer indices in keys, like:
        `{"a[0].b": x, "a[1].b": y}` becomes
        `{"a": {0: {"b": x}, 1: {"b": y}}}`.
    """
    result = dict[str | int, Any]()
    for flat_key, v in d.items():
        keys: list[str | int] = [
            k if i == 0 else int(k.removesuffix("]"))
            for part in flat_key.split(delimiter)
            for i, k in enumerate(part.split("["))
        ]
        subresult: dict[str | int, Any] = result
        for key in keys[:-1]:
            if key not in subresult:
                subresult[key] = {}
            subresult = subresult[key]
        subresult[keys[-1]] = v
    return result


def deep_asdict(d: Any) -> dict[str, Any]:
    """Convert a nested dataclass/dict/list structure into a nested dict."""
    if is_dataclass(d) and not isinstance(d, type):
        d = asdict(d)
    elif isinstance(d, Namespace):
        d = vars(d)

    result = dict[str, Any]()
    for k, v in d.items():
        if isinstance(v, Mapping) or (is_dataclass(v) and not isinstance(v, type)):
            result[k] = deep_asdict(v)
        elif isinstance(v, list):
            result[k] = [deep_asdict(item) for item in v]
        else:
            result[k] = v
    return result


def deep_replace[T](obj: T, replacements: dict[str, Any]) -> T:
    """
    Recursively replace specified values in a nested dataclass/dict structure.

    Example: `deep_replace(obj, {"a": {0: x}, "c.d[1].e": y})`
    replaces `obj.a[0]` with `x` and `obj.c.d[1].e` with `y`.
    """
    for k, v in unflatten_dict(flatten_dict(replacements)).items():
        if isinstance(obj, MutableMapping):
            assert isinstance(k, str), f"Mapping keys into {type(obj)} must be str, got: {type(k)}"
            if k in obj and isinstance(v, dict):
                v = deep_replace(obj[k], v)
            obj = copy(obj)
            obj[k] = v
        elif isinstance(obj, list):
            assert isinstance(k, int), f"Indices into {type(obj)} must be int, got: {type(k)}"
            if k in obj and isinstance(v, dict):
                v = deep_replace(obj[k], v)
            obj = copy(obj)
            if k == len(obj):
                obj.append(v)
            else:
                obj[k] = v
        else:
            assert is_dataclass(obj) and not isinstance(v, type), f"Expected mapping or dataclass, got: {type(obj)}"
            assert isinstance(k, str), f"Attr names into {type(obj)} must be str, got: {type(k)}"
            if hasattr(obj, k) and isinstance(v, dict):
                v = deep_replace(getattr(obj, k), v)
            obj = replace(obj, **{k: v})  # type: ignore

    return obj


_REGISTERED_CONFIGS = dict[Any, dict[str, Any]]()


def register_configs[T](*configs: T) -> None:
    """Register one or more config dataclass instances, so they can be found with get_config()."""
    for config in configs:
        name = config.name  # type: ignore[attr-defined]
        if type(config) not in _REGISTERED_CONFIGS:
            _REGISTERED_CONFIGS[type(config)] = {}
        _REGISTERED_CONFIGS[type(config)][name] = config


@functools.cache
def _import_all_configs() -> dict[Any, dict[str, Any]]:
    """Import all .py files in configs/ or subdirectories of it (recursively)."""
    config_dir = Path(__file__).parent.parent / "configs"
    for p in config_dir.rglob("*.py"):
        module_path = "..configs." + ".".join(p.relative_to(config_dir).parts).removesuffix(".py")
        importlib.import_module(module_path, __package__)
    return _REGISTERED_CONFIGS


def get_config[T](config_type: type[T], name: str) -> T:
    """Find a previously registered config instance by type and name."""
    register_configs = _import_all_configs()

    raise_if_not_contained(str(config_type), [str(k) for k in register_configs.keys()], "Config type")
    raise_if_not_contained(name, register_configs[config_type], "Config")

    return cast(T, register_configs[config_type][name])
