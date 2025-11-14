import os
from pathlib import Path

import nox

nox.options.default_venv_backend = "uv"

nox.options.sessions = []


_PROJECT_DIR = Path(__file__).parent


@nox.session(python=False)
def initialize_specfem_repo(session):
    SPECFEM_REPO_URL = "https://github.com/PrincetonUniversity/SPECFEMPP"

    repo_wrapper = _PROJECT_DIR / "terrarium"
    if not repo_wrapper.exists():
        repo_wrapper.mkdir()
    session.cd(repo_wrapper)
    os.system(f"git clone {SPECFEM_REPO_URL}")

    # build_dir = config.get("specfem.live.build")
    # if os.path.exists(build_dir):
    #     shutil.rmtree(build_dir)
    # os.system(
    #     f"cmake -S . -B {build_dir} {config.get('specfem.live.cmake_build_options')}"
    # )
