from dataclasses import dataclass
from pathlib import Path
from typing import Literal, overload


@dataclass
class ExternalMesherFileConfig:
    """The files used by Exporter when outputting."""

    base_folder: Path
    """Base directory of the external meshing files."""

    mesh_file: str = "mesh"
    """Name of the file (path relative to `destination_folder`).

    Defaults to "mesh".
    """
    node_coords_file: str = "node_coords"
    """Name of the file (path relative to `destination_folder`).

    Defaults to "node_coords".
    """
    materials_file: str = "materials"
    """Name of the file (path.

                relative to `destination_folder`). Defaults to
                "materials".
    """
    free_surface_file: str = "free_surface"
    """Name of the file.

                (path relative to `destination_folder`). Defaults
                to "free_surface".
    """

    axial_elements_file: str | None = None
    """Name of the file.

                (path relative to `destination_folder`), or None
                for no export. Defaults to None.
    """
    absorbing_surface_file: str | None = "absorbing_surface"
    """Name of the file.

                (path relative to `destination_folder`), or None
                for no export. Defaults to "absorbing_surface".
    """
    acoustic_forcing_surface_file: str | None = None
    """Name of.

                the file (path relative to `destination_folder`),
                or None for no export. Defaults to None.
    """
    absorbing_cpml_file: str | None = None
    """Name of the file.

                (path relative to `destination_folder`), or None
                for no export. Defaults to None.
    """
    tangential_detection_curve_file: str | None = None
    """Name.

                of the file (path relative to `destination_folder`),
                or None for no export. Defaults to None.
    """
    nonconforming_adjacencies_file: str | None = None
    """Name of.

                the file (path relative to `destination_folder`),
                or None for no export. Defaults to None.
    """

    @overload
    def _resolve_file(
        self, file: str, strict: Literal[False]
    ) -> Path | None: ...
    @overload
    def _resolve_file(
        self, file: str, strict: Literal[True] = True
    ) -> Path: ...

    def _resolve_file(self, file: str, strict: bool = True) -> Path | None:
        filename = getattr(self, file)
        if filename is None:
            if strict:
                e = ValueError(
                    f"ExporterFileConfig: `{file}` is not set. Cannot resolve."
                )
                raise e
            return None
        return self.base_folder / filename

    @overload
    def resolve_mesh_file(self, strict: Literal[False]) -> Path | None: ...
    @overload
    def resolve_mesh_file(
        self, strict: Literal[True] = True
    ) -> Path | None: ...
    def resolve_mesh_file(self, strict=True) -> Path | None:
        return self._resolve_file("mesh_file", strict=strict)

    @overload
    def resolve_node_coords_file(
        self, strict: Literal[False]
    ) -> Path | None: ...
    @overload
    def resolve_node_coords_file(
        self, strict: Literal[True] = True
    ) -> Path | None: ...
    def resolve_node_coords_file(self, strict=True) -> Path | None:
        return self._resolve_file("node_coords_file", strict=strict)

    @overload
    def resolve_materials_file(self, strict: Literal[False]) -> Path | None: ...
    @overload
    def resolve_materials_file(
        self, strict: Literal[True] = True
    ) -> Path | None: ...
    def resolve_materials_file(self, strict=True) -> Path | None:
        return self._resolve_file("materials_file", strict=strict)

    @overload
    def resolve_free_surface_file(
        self, strict: Literal[False]
    ) -> Path | None: ...
    @overload
    def resolve_free_surface_file(
        self, strict: Literal[True] = True
    ) -> Path | None: ...
    def resolve_free_surface_file(self, strict=True) -> Path | None:
        return self._resolve_file("free_surface_file", strict=strict)

    @overload
    def resolve_axial_elements_file(
        self, strict: Literal[False]
    ) -> Path | None: ...
    @overload
    def resolve_axial_elements_file(
        self, strict: Literal[True] = True
    ) -> Path | None: ...
    def resolve_axial_elements_file(self, strict=True) -> Path | None:
        return self._resolve_file("axial_elements_file", strict=strict)

    @overload
    def resolve_absorbing_surface_file(
        self, strict: Literal[False]
    ) -> Path | None: ...
    @overload
    def resolve_absorbing_surface_file(
        self, strict: Literal[True] = True
    ) -> Path | None: ...
    def resolve_absorbing_surface_file(self, strict=True) -> Path | None:
        return self._resolve_file("absorbing_surface_file", strict=strict)

    @overload
    def resolve_acoustic_forcing_surface_file(
        self, strict: Literal[False]
    ) -> Path | None: ...
    @overload
    def resolve_acoustic_forcing_surface_file(
        self, strict: Literal[True] = True
    ) -> Path | None: ...
    def resolve_acoustic_forcing_surface_file(self, strict=True) -> Path | None:
        return self._resolve_file(
            "acoustic_forcing_surface_file", strict=strict
        )

    @overload
    def resolve_absorbing_cpml_file(
        self, strict: Literal[False]
    ) -> Path | None: ...
    @overload
    def resolve_absorbing_cpml_file(
        self, strict: Literal[True] = True
    ) -> Path | None: ...
    def resolve_absorbing_cpml_file(self, strict=True) -> Path | None:
        return self._resolve_file("absorbing_cpml_file", strict=strict)

    @overload
    def resolve_tangential_detection_curve_file(
        self, strict: Literal[False]
    ) -> Path | None: ...
    @overload
    def resolve_tangential_detection_curve_file(
        self, strict: Literal[True] = True
    ) -> Path | None: ...
    def resolve_tangential_detection_curve_file(
        self, strict=True
    ) -> Path | None:
        return self._resolve_file(
            "tangential_detection_curve_file", strict=strict
        )

    @overload
    def resolve_nonconforming_adjacencies_file(
        self, strict: Literal[False]
    ) -> Path | None: ...
    @overload
    def resolve_nonconforming_adjacencies_file(
        self, strict: Literal[True] = True
    ) -> Path | None: ...
    def resolve_nonconforming_adjacencies_file(
        self, strict=True
    ) -> Path | None:
        return self._resolve_file(
            "nonconforming_adjacencies_file", strict=strict
        )

    def param_string(self):
        def dummy_if_None(value):
            return "dummy" if value is None else value

        return f"""
mesh_file                       = {dummy_if_None(self.resolve_mesh_file(strict=False))}
nodes_coords_file               = {dummy_if_None(self.resolve_node_coords_file(strict=False))}
materials_file                  = {dummy_if_None(self.resolve_materials_file(strict=False))}
free_surface_file               = {dummy_if_None(self.resolve_free_surface_file(strict=False))}
axial_elements_file             = {dummy_if_None(self.resolve_axial_elements_file(strict=False))}
absorbing_surface_file          = {dummy_if_None(self.resolve_absorbing_surface_file(strict=False))}
acoustic_forcing_surface_file   = {dummy_if_None(self.resolve_acoustic_forcing_surface_file(strict=False))}
absorbing_cpml_file             = {dummy_if_None(self.resolve_absorbing_cpml_file(strict=False))}
tangential_detection_curve_file = {dummy_if_None(self.resolve_tangential_detection_curve_file(strict=False))}
nonconforming_adjacencies_file  = {dummy_if_None(self.resolve_nonconforming_adjacencies_file(strict=False))}
        """
