import os
from dataclasses import dataclass
from pathlib import Path

from .model import Model
from .model.edges import EdgeType
from .model.physical_group import (
    NullPhysicalGroup,
    PhysicalGroupBase,
)

NONCONFORMING_CONNECTION_TYPE = 3


def model_edge_to_meshfem_edge(value: EdgeType):
    if value == EdgeType.BOTTOM:
        return 1
    if value == EdgeType.RIGHT:
        return 2
    if value == EdgeType.TOP:
        return 3
    if value == EdgeType.LEFT:
        return 4
    err = ValueError(
        f"model_edge_to_meshfem_edge(): Cannot process value: {value}"
    )
    err.add_note(
        f"`value` must be one of EdgeType.TOP ({EdgeType.TOP}),"
        f" EdgeType.BOTTOM ({EdgeType.BOTTOM}), EdgeType.LEFT"
        f" ({EdgeType.LEFT}), or EdgeType.RIGHT ({EdgeType.RIGHT})."
    )
    raise err


@dataclass
class ExporterFileConfig:
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

    def _resolve_file(self, file: str) -> Path:
        filename = getattr(self, file)
        if filename is None:
            e = ValueError(
                f"ExporterFileConfig: `{file}` is not set. Cannot resolve."
            )
            raise e
        return self.base_folder / filename

    def resolve_mesh_file(self):
        return self._resolve_file("mesh_file")

    def resolve_node_coords_file(self):
        return self._resolve_file("node_coords_file")

    def resolve_materials_file(self):
        return self._resolve_file("materials_file")

    def resolve_free_surface_file(self):
        return self._resolve_file("free_surface_file")

    def resolve_axial_elements_file(self):
        return self._resolve_file("axial_elements_file")

    def resolve_absorbing_surface_file(self):
        return self._resolve_file("absorbing_surface_file")

    def resolve_acoustic_forcing_surface_file(self):
        return self._resolve_file("acoustic_forcing_surface_file")

    def resolve_absorbing_cpml_file(self):
        return self._resolve_file("absorbing_cpml_file")

    def resolve_tangential_detection_curve_file(self):
        return self._resolve_file("tangential_detection_curve_file")

    def resolve_nonconforming_adjacencies_file(self):
        return self._resolve_file("nonconforming_adjacencies_file")


class Exporter:
    destination_folder: Path
    file_outputs: ExporterFileConfig

    model: Model

    acoustic_free_surface_physical_group: PhysicalGroupBase
    absorbing_surface_physical_group: PhysicalGroupBase

    def __init__(
        self,
        model: Model,
        file_outputs: ExporterFileConfig | str | os.PathLike,
        acoustic_free_surface_physical_group: str
        | None = "acoustic_free_surface",
        absorbing_surface_physical_group: str | None = "absorbing",
    ):
        """Initialize an Exporter2D object to write `model` to files for.

        Args:
            model (Model): the model to export
            file_outputs (ExporterFileConfig | str | os.PathLike): An
                ExporterFileConfig object containing the configuration of all
                file exports, or a path of the folder to export to.
                If a path is given (not ExporterFileConfig), then the default
                file configuration is set.
            acoustic_free_surface_physical_group (str | None, optional):
                the name of the physical group used for referencing the
                acoustic free surface. Defaults to "acoustic_free_surface".
            absorbing_surface_physical_group (str | None, optional):
                the name of the physical group used for referencing the
                absorbing boundary. Defaults to "absorbing".
        """
        self.model = model
        self.file_outputs = (
            file_outputs
            if isinstance(file_outputs, ExporterFileConfig)
            else ExporterFileConfig(Path(file_outputs))
        )
        self.acoustic_free_surface_physical_group = (
            NullPhysicalGroup("_NULL_AFS_")
            if acoustic_free_surface_physical_group is None
            or acoustic_free_surface_physical_group
            not in self.model.physical_groups
            else self.model.physical_groups[
                acoustic_free_surface_physical_group
            ]
        )
        self.absorbing_surface_physical_group = (
            NullPhysicalGroup("_NULL_ABS_")
            if absorbing_surface_physical_group is None
            or absorbing_surface_physical_group
            not in self.model.physical_groups
            else self.model.physical_groups[absorbing_surface_physical_group]
        )

    def export_mesh(self):
        if not self.destination_folder.exists():
            self.destination_folder.mkdir()

        # =========================
        # node coords
        # =========================
        with self.file_outputs.resolve_node_coords_file().open("w") as f:
            # gmsh is in 3d. this Exporter is in a 2d namespace.
            # by default, we will resolve by assuming y=0, but in the future,
            # some more general projection rule may be desired.
            nodes_arr = self.model.nodes[..., (0, 2)]

            # header is number of lines (1 line per node)
            nnodes = nodes_arr.shape[0]
            f.write(str(nnodes) + "\n")

            for inod in range(nnodes):
                f.write(
                    f"{nodes_arr[inod, 0]:.10f} {nodes_arr[inod, 1]:.10f}\n"
                )

        nelem = self.model.elements.shape[0]

        # =========================
        # elements
        # =========================
        with self.file_outputs.resolve_mesh_file().open("w") as f:
            elem_arr = self.model.elements

            f.write(str(nelem) + "\n")
            for ielem in range(nelem):
                f.write(
                    " ".join(f"{k + 1:d}" for k in elem_arr[ielem, :]) + "\n"
                )

        # =========================
        # materials
        # =========================
        with self.file_outputs.resolve_materials_file().open("w") as f:
            # no header entry

            for mat in self.model.materials:
                f.write(f"{mat}\n")

        # =========================
        # free surface
        # =========================
        with self.file_outputs.resolve_free_surface_file().open("w") as f:
            elements, edgetypes = (
                self.acoustic_free_surface_physical_group.get_all_edges()
            )

            f.write(str(elements.shape[0]) + "\n")

            # we're not handling corner cases. Make sure this is fine.
            # (or just let it go until something breaks and you find this
            # comment)

            for elem, edgetype in zip(elements, edgetypes, strict=False):
                node_indices = self.model.elements[
                    elem, EdgeType.QUA_9_node_indices_on_type(edgetype)[::2]
                ]
                f.write(
                    f"{elem + 1} 2 {node_indices[0] + 1} "
                    f"{node_indices[1] + 1}\n"
                )

        # =========================
        # absorbing bdries (if needed)
        # =========================
        if self.file_outputs.absorbing_surface_file is not None:
            with self.file_outputs.resolve_absorbing_surface_file().open(
                "w"
            ) as f:
                elements, edgetypes = (
                    self.absorbing_surface_physical_group.get_all_edges()
                )
                f.write(str(elements.shape[0]) + "\n")

                for elem, edgetype in zip(elements, edgetypes, strict=False):
                    node_indices = self.model.elements[
                        elem, EdgeType.QUA_9_node_indices_on_type(edgetype)[::2]
                    ]
                    f.write(
                        f"{elem + 1} 2 {node_indices[0] + 1} "
                        f"{node_indices[1] + 1} {edgetype + 1}\n"
                    )

        # =========================
        # acoustic forcing (if needed)
        # =========================
        if self.file_outputs.acoustic_forcing_surface_file is not None:
            with self.file_outputs.resolve_acoustic_forcing_surface_file().open(
                "w"
            ) as f:
                # NotImplemented
                f.write(str(0))

        # =========================
        # absorbing cpml (if needed)
        # =========================
        if self.file_outputs.absorbing_cpml_file is not None:
            with self.file_outputs.resolve_absorbing_cpml_file().open("w") as f:
                # NotImplemented
                f.write(str(0))

        # =========================
        # tangential curve (if needed)
        # =========================
        if self.file_outputs.tangential_detection_curve_file is not None:
            with (
                self.file_outputs.resolve_tangential_detection_curve_file().open(
                    "w"
                ) as f
            ):
                # NotImplemented
                f.write(str(0))

        # =========================
        # nonconforming adjacencies (if needed)
        # =========================
        if self.file_outputs.nonconforming_adjacencies_file is not None:
            with (
                self.file_outputs.resolve_nonconforming_adjacencies_file().open(
                    "w"
                ) as f
            ):
                num_pairs = self.model.nonconforming_interfaces.edges_a.shape[0]
                f.write(str(num_pairs * 2) + "\n")

                for ispec_a, ispec_b, edge_a, edge_b in zip(
                    self.model.nonconforming_interfaces.elements_a,
                    self.model.nonconforming_interfaces.elements_b,
                    self.model.nonconforming_interfaces.edges_a,
                    self.model.nonconforming_interfaces.edges_b,
                    strict=True,
                ):
                    f.write(
                        f"{ispec_a + 1:d} {ispec_b + 1:d} "
                        f"{NONCONFORMING_CONNECTION_TYPE:d} "
                        f"{model_edge_to_meshfem_edge(edge_a):d}\n"
                    )
                    f.write(
                        f"{ispec_b + 1:d} {ispec_a + 1:d} "
                        f"{NONCONFORMING_CONNECTION_TYPE:d} "
                        f"{model_edge_to_meshfem_edge(edge_b):d}\n"
                    )
