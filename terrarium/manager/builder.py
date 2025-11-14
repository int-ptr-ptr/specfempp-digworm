import shutil
import subprocess
from pathlib import Path

_SPECFEM_REPO = Path(__file__).parent.parent / "SPECFEMPP"


def initialize_build_dir(
    build_dir=_SPECFEM_REPO / "build", capture_cmd_output=False
):
    subprocess.run(
        ["git", "checkout", "d54f53173eabafcdeab90313f5466bd4cf2ff9b3"],
        check=False,
        cwd=_SPECFEM_REPO,
        capture_output=capture_cmd_output,
    )

    # if build_dir.exists():
    #     shutil.rmtree(build_dir)
    cmd = [
        "cmake",
        "-S",
        str(_SPECFEM_REPO.resolve()),
        "-B",
        str(build_dir.resolve()),
    ]
    cmd.append("-D SPECFEM_ENABLE_VTK=OFF")
    cmd.append("-D Kokkos_ENABLE_SERIAL=ON")

    cmd.append("-D CMAKE_BUILD_TYPE=Release")
    # cmd.append("-D CMAKE_BUILD_TYPE=Debug")

    cmd.append("-D Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON")
    cmd.append("-D SPECFEM_ENABLE_SIMD=ON")
    cmd.append("-D Kokkos_ENABLE_ATOMICS_BYPASS=OFF")

    # TODO user flags to set
    cmd.append("-D CMAKE_C_COMPILER=gcc-13")
    cmd.append("-D CMAKE_CXX_COMPILER=g++-13")
    cmd.append("-D CMAKE_Fortran_COMPILER=gfortran-13")
    try:
        out = subprocess.run(
            cmd,
            check=True,
            cwd=_SPECFEM_REPO,
            capture_output=capture_cmd_output,
        )
    except subprocess.SubprocessError as e:
        raise e


def build(
    build_dir=_SPECFEM_REPO / "build", capture_cmd_output=False, nthreads=4
):
    cmd = ["cmake", "--build", str(build_dir.resolve()), "--parallel", str(nthreads)]
    try:
        out = subprocess.run(
            cmd,
            check=True,
            cwd=_SPECFEM_REPO,
            capture_output=capture_cmd_output,
        )
    except subprocess.SubprocessError as e:
        raise e


initialize_build_dir()
build()