import contextlib
import copy
import re
from os import PathLike
from pathlib import Path
from typing import Literal, overload

from yaml import Loader, load

_VALID_CONFIG_FILE_NAMES = ["digworm_config.yaml", "digworm_config.yml"]


def get_config_file_list(path: Path) -> list[Path]:
    """Returns a list of configuration files that would affect a given path.
    The list is ordered from highest to lowest priority, where a higher priority
    entry will overwrite a lower priority entry.

    Args:
        path (Path): The working directory for a configuration

    Returns:
        list[Path]: The list of config files.
    """
    path = path.absolute()
    config_files = []
    # do while (path has parent)
    while True:
        # does path have a valid configuration file?
        for cfg_fname in _VALID_CONFIG_FILE_NAMES:
            with contextlib.suppress(OSError):
                check = path / cfg_fname
                if check.is_file():
                    # yes.
                    config_files.append(check)

                # expect only one config file per directory
                break

        if len(path.parts) <= 1:
            break

        path = path.parent
    return config_files


def _read_default_config_files(
    path: Path | None = None, *, config_file_list: list[Path] | None = None
):
    from ._defaults import _config_defaults  # noqa: PLC0415

    config = copy.deepcopy(_config_defaults)

    config_search = Path.cwd() if path is None else path
    # search directories for configuration file (trace up to root)
    # files in subdirectories have priority (overwrites config)

    def recursive_update(entry, new_entry):
        # for now, assume entry is a dictionary, since we don't
        # support advanced list manipulation
        assert isinstance(entry, dict)
        assert isinstance(new_entry, dict)

        if isinstance(new_entry, dict):
            for k, v in new_entry.items():
                if k not in entry:
                    entry[k] = v
                    continue
                if isinstance(entry[k], dict):
                    # dictionary update
                    recursive_update(entry[k], v)
                    continue

                if isinstance(entry[k], list):
                    assert isinstance(v, list)
                    # just append for now
                    entry[k].extend(v)
                    continue

                # perhaps add more advanced type-checking in the future?
                entry[k] = v

    files_to_load = (
        get_config_file_list(config_search)
        if config_file_list is None
        else config_file_list
    )
    for config_file in reversed(files_to_load):
        with config_file.open() as f:
            recursive_update(config, load(f, Loader))

    return config


class ConfigurationNode:
    """A branch in the configuration tree, corresponding to a certain namespace.
    A ConfigurationNode can represent either a list or dictionary in the config.
    """

    _parent: "ConfigurationNode | None"
    _namespace: tuple[str, ...]
    _root: "ConfigurationNode"
    _contents_original: dict[str, list | dict | str] | list[list | dict | str]
    _contents: (
        dict[str, "ConfigurationNode | str"] | list["ConfigurationNode | str"]
    )

    def __init__(
        self,
        contents: dict[str, list | dict | str] | list[list | dict | str],
        parent: "ConfigurationNode | None" = None,
        current_name: str | None = None,
    ):
        self._parent = parent
        self._contents_original = contents
        if parent is None:
            self._root = self
            self._namespace = () if current_name is None else (current_name,)
        else:
            self._root = parent._root
            if current_name is None:
                e = ValueError(
                    "When passing in a `parent`, `current_name` "
                    "must be provided."
                )
                raise e
            self._namespace = (*parent._namespace, current_name)

        if isinstance(contents, dict):
            self._contents = {
                key: (
                    ConfigurationNode(
                        contents=value, parent=self, current_name=key
                    )
                    if isinstance(value, (dict, list))
                    else str(value)
                )
                for key, value in contents.items()
            }
        elif isinstance(contents, list):
            self._contents = [
                (
                    ConfigurationNode(
                        contents=value, parent=self, current_name=str(ivalue)
                    )
                    if isinstance(value, (dict, list))
                    else str(value)
                )
                for ivalue, value in enumerate(contents)
            ]
        else:
            e = ValueError("`contents` of invalid type.")
            raise e

    def _separate_namespace(self, namespace: str) -> tuple[str, ...]:
        return tuple(namespace.split("."))

    def _resolve_here(self, name: str) -> "str | ConfigurationNode":
        if isinstance(self._contents, dict):
            return self._contents[name]
        if isinstance(self._contents, list):
            return self._contents[int(name)]
        e = ValueError(
            "_contents is of invalid type! The configuration is broken!"
        )
        raise e

    def _content_keys(self) -> list[str]:
        if isinstance(self._contents, dict):
            return list(self._contents.keys())
        return [f"{i}" for i in range(len(self._contents))]

    @overload
    def resolve_name(
        self,
        name: str | tuple[str, ...],
        return_full_namespace: Literal[False] = False,
        substitute_references: bool = True,
    ) -> "ConfigurationNode | str": ...
    @overload
    def resolve_name(
        self,
        name: str | tuple[str, ...],
        return_full_namespace: Literal[True],
        substitute_references: bool = True,
    ) -> "tuple[ConfigurationNode | str, tuple[str,...]]": ...

    def resolve_name(
        self,
        name: str | tuple[str, ...],
        return_full_namespace: bool = False,
        substitute_references: bool = True,
    ) -> "ConfigurationNode | str | tuple[ConfigurationNode | str, tuple[str,...]]":
        if isinstance(name, str):
            name = self._separate_namespace(name)

        # we may need to trace back until the first name matches
        try:
            inst = self
            inst_parent = self
            full_namespace = self._namespace + name
            height = 0
            while height < len(name):
                curname = name[height]
                if isinstance(inst, str):
                    # we should have been at the leaf
                    e = ValueError(
                        f"Cannot resolve configuration namespace {name}: at "
                        f"{name[:height]}, {curname} is a string and cannot be "
                        "resolved further."
                    )
                    raise e
                if isinstance(inst, ConfigurationNode):
                    try:
                        inst_parent = inst
                        inst = inst._resolve_here(curname)
                        height += 1
                    except Exception as e:
                        e2 = ValueError(
                            f"Cannot resolve configuration namespace {name}: "
                            f"failed to expand '{curname}' at height = {height}"
                        )
                        e2.add_note(
                            f"Valid keys in namespace {inst._namespace}:\n"
                            + "\n".join(inst._content_keys())
                        )
                        raise e2 from e

            if substitute_references and isinstance(inst, str):
                inst = inst_parent._expand_references(inst)
            if return_full_namespace:
                return inst, full_namespace
            return inst
        except ValueError as e:
            if self._parent is None:
                raise e

        # try one level up. This will only be called if ValueError was raised,
        # but we put it here to clear out the traceback
        return self._parent.resolve_name(
            name=name,
            return_full_namespace=return_full_namespace,
            substitute_references=substitute_references,
        )

    def __getitem__(self, indices: tuple[str, ...] | str):
        if isinstance(indices, tuple) and len(indices) == 1:
            indices = indices[0]
        return self.resolve_name(indices)

    def _expand_references(
        self,
        strcontent: str,
        already_expanded_refs: list[tuple[str, ...]] | None = None,
    ):
        if already_expanded_refs is None:
            already_expanded_refs = []

        while match := re.search(r"\$(?:\{([\.\w-]+)\}|(-\w+))", strcontent):
            # first or second group (first is if {} was used)
            replcode = match.group(1)
            if replcode is None:
                replcode = match.group(2)

            repl, repl_full_ns = self.resolve_name(
                replcode,
                substitute_references=False,
                return_full_namespace=True,
            )

            namespace = repl_full_ns
            if namespace in already_expanded_refs:
                e = ValueError(
                    "When expanding reference: circular reference found: "
                    f"{namespace}"
                )

            if isinstance(repl, ConfigurationNode):
                e = ValueError(
                    f"when replacing {replcode} in {strcontent}: "
                    "did not recover a string!"
                )
                raise e

            # this may also have references. append to stack to prevent circular
            already_expanded_refs.append(namespace)
            parent_of_subs = self.resolve_name(repl_full_ns[:-1])
            assert isinstance(parent_of_subs, ConfigurationNode), (
                repl_full_ns,
                parent_of_subs,
            )
            repl = parent_of_subs._expand_references(
                repl, already_expanded_refs
            )
            already_expanded_refs.pop()

            strcontent = (
                strcontent[: match.start()] + repl + strcontent[match.end() :]
            )
        return strcontent


_configs: dict[str, ConfigurationNode] = {}


def get_config(
    working_directory: str | PathLike | None = None,
) -> ConfigurationNode:
    if working_directory is None:
        config_file_list = None
        key = "<cwd>"
    else:
        config_file_list = get_config_file_list(Path(working_directory))
        if len(config_file_list) > 0:
            key = str(config_file_list[0].parent)
        else:
            key = "_default"
    if key not in _configs:
        _configs[key] = ConfigurationNode(
            _read_default_config_files(config_file_list=config_file_list)
        )
    return _configs[key]
