import re
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import numpy as np
import setup_sims
from setup_sims import (
    get_gridsims,
    test_config,
    test_config_fol,
    truesol_gridsim,
)

labdir = Path(__file__).parent

sys.path.append(str((labdir / "../simple_err_experiment").resolve()))

from read_external_mesh import display_mesh  # type:ignore  # noqa: E402, I001, PGH003


sims_dir = labdir / "sims"
out_dir = labdir / "outputs"

simulations_per_config = {}

for simfol in sims_dir.iterdir():
    if not simfol.is_dir():
        continue
    if m := re.match(r"(\d+)_(\d+)_([\dx]+)_(True|False)", simfol.stem):
        iconf = 0
        nx_below = int(m.group(1))
        nx_above = int(m.group(2))
        is_nonconforming = bool(m.group(4))

        if iconf not in simulations_per_config:
            simulations_per_config[iconf] = []

        simulations_per_config[iconf].append(
            SimpleNamespace(
                iconf=iconf,
                nx_below=nx_below,
                nx_above=nx_above,
                is_nonconforming=is_nonconforming,
                path=simfol,
            )
        )


cutoff_freq = 100
do_cutoff_freq = True


def get_truesol_seismos(iconf):
    seisfol = test_config_fol(**truesol_gridsim()) / "OUTPUT_FILES/seismograms"
    seismos = {}

    for file in seisfol.iterdir():
        data = np.loadtxt(file)

        if do_cutoff_freq:
            data[:, 1] = np.real(
                np.fft.ifft(
                    np.fft.fft(data[:, 1])
                    * (
                        np.abs(
                            np.fft.fftfreq(
                                data.shape[0], data[1, 0] - data[0, 0]
                            )
                        )
                        < cutoff_freq
                    )
                )
            )

        seismos[file.stem] = data
        if np.any(np.isnan(data[:, 1])):
            print(f"{iconf}: seismo {file.stem} has NaN values ({file})")
            (inds,) = np.where(np.isnan(data[:, 1]))
            print(f"first instance at index {inds[0]} (t = {data[inds[0], 0]})")

    if not seismos:
        e = RuntimeError(
            f"True solution simulation not run (config# = {iconf})."
        )
        raise e

    return seismos


def plot_errs(iconf):
    simulations = simulations_per_config[iconf]

    truesol_seismos = get_truesol_seismos(iconf)

    truesol_params = truesol_gridsim()

    total_truesol_integ = 0
    for _seis, data in truesol_seismos.items():
        total_truesol_integ += np.sum(data[:, 1] ** 2)
    # fig, ax = plt.subplots()

    DX1 = []
    DX2 = []
    DT = []
    ERR = []

    DX1_conforming = []
    DX2_conforming = []
    DT_conforming = []
    ERR_conforming = []

    DX1_blowup = []
    DX2_blowup = []
    DT_blowup = []

    for sim_params in get_gridsims():
        if truesol_params == sim_params:
            continue
        dx1 = 2 * np.pi / sim_params["nx_below"]
        dx2 = 2 * np.pi / sim_params["nx_above"]
        dt = sim_params["dt"]
        sim_fol = test_config_fol(**sim_params)
        seismofol = sim_fol / "OUTPUT_FILES/seismograms"

        if not seismofol.exists():
            continue

        err = 0
        seismos_compared = 0

        for file in seismofol.iterdir():
            data = np.loadtxt(file)
            trueseis = truesol_seismos[file.stem]

            # # skip other seismos
            # if file.stem not in [
            #     "AA.S0001.S2.PRE",
            #     "AA.S0002.S2.PRE",
            #     "AA.S0003.S2.PRE",
            #     "AA.S0006.S2.PRE",
            #     "AA.S0007.S2.PRE",
            #     "AA.S0008.S2.PRE",
            #     "AA.S0011.S2.PRE",
            # ]:
            #     continue

            if do_cutoff_freq:
                data[:, 1] = np.real(
                    np.fft.ifft(
                        np.fft.fft(data[:, 1])
                        * (
                            np.abs(
                                np.fft.fftfreq(
                                    data.shape[0], data[1, 0] - data[0, 0]
                                )
                            )
                            < cutoff_freq
                        )
                    )
                )

            interp_data = np.interp(trueseis[:, 0], data[:, 0], data[:, 1])

            err += np.sum((interp_data - trueseis[:, 1]) ** 2)
            seismos_compared += 1

            # plt.plot(data[:, 0], data[:, 1])

        if seismos_compared < 1:
            continue

        err_rms = np.sqrt(err / total_truesol_integ)

        if sim_params["do_nonconforming"]:
            DX1.append(dx1)
            DX2.append(dx2)
            DT.append(dt)
            ERR.append(err_rms)
        else:
            DX1_conforming.append(dx1)
            DX2_conforming.append(dx2)
            DT_conforming.append(dt)
            ERR_conforming.append(err_rms)

        ERR_BLOWUP_THRESHOLD = 1e3
        if err_rms > ERR_BLOWUP_THRESHOLD or np.isnan(err_rms):
            DX1_blowup.append(dx1)
            DX2_blowup.append(dx2)
            DT_blowup.append(dt)

    outfol = out_dir / "seismoplots"
    outfol.mkdir(parents=True, exist_ok=True)

    DX1 = np.array(DX1)
    DX2 = np.array(DX2)
    DT = np.array(DT)

    DX1_conforming = np.array(DX1_conforming)
    DX2_conforming = np.array(DX2_conforming)
    DT_conforming = np.array(DT_conforming)

    has_blowups = len(DX1_blowup) > 0

    DX1_blowup = np.array(DX1_blowup)
    DX2_blowup = np.array(DX2_blowup)
    DT_blowup = np.array(DT_blowup)

    # =================================
    # plots
    # =================================

    # 3d
    # ---------------------------------

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    errscatter = ax.scatter(
        DX1,
        DX2,
        DT,
        c=np.log10(ERR),
        marker="v",
        label="nonconforming",
        vmin=-3,
        vmax=1,
    )

    fig.colorbar(errscatter, label="$\\log_{10}$ error", pad=0.1)
    ax.set_title("seismogram RMS error from reference simulation")
    ax.scatter(
        DX1_conforming,
        DX2_conforming,
        DT_conforming,
        c=np.log10(ERR_conforming),
        marker="^",
        label="conforming",
        vmin=-3,
        vmax=1,
    )
    if has_blowups:
        ax.scatter(
            DX1_blowup,
            DX2_blowup,
            DT_blowup,
            color="r",
            marker="+",
            label="blowups",
        )

    ax.set_xlabel(r"$(\Delta x)_S$")
    ax.set_ylabel(r"$(\Delta x)_F$")
    ax.set_zlabel(r"$(\Delta t)$")
    ax.legend()
    fig.tight_layout()

    fig.savefig(outfol / f"errs3D_{iconf}.png")
    fig.clf()

    # by CFL per side
    # ---------------------------------

    fig = plt.figure()
    ax = fig.add_subplot()

    CFL1 = 1 * DT / DX1
    CFL2 = 1963 / 3400 * DT / DX2

    CFL1_conforming = 1 * DT_conforming / DX1_conforming
    CFL2_conforming = 1963 / 3400 * DT_conforming / DX2_conforming

    CFL1_blowup = 1 * DT_blowup / DX1_blowup
    CFL2_blowup = 1963 / 3400 * DT_blowup / DX2_blowup

    errscatter = ax.scatter(
        CFL1,
        CFL2,
        c=np.log10(ERR),
        marker="v",
        label="nonconforming",
        vmin=-3,
        vmax=1,
    )

    fig.colorbar(errscatter, label="$\\log_{10}$ error", pad=0.1)
    ax.set_title("seismogram RMS error from reference simulation")
    ax.scatter(
        CFL1_conforming,
        CFL2_conforming,
        c=np.log10(ERR_conforming),
        marker="^",
        label="conforming",
        vmin=-3,
        vmax=1,
    )
    if has_blowups:
        ax.scatter(
            CFL1_blowup,
            CFL2_blowup,
            color="r",
            marker="+",
            label="blowups",
        )

    ax.set_xlabel(r"solid Courant Number")
    ax.set_ylabel(r"fluid Courant Number")
    ax.legend()
    fig.tight_layout()

    fig.savefig(outfol / f"errs_cfl_{iconf}.png")
    fig.clf()

    # by u/dx per side
    # ---------------------------------

    fig, ax = plt.subplots(ncols=4, figsize=(12, 5))

    U_BY_DX1 = 1 / DX1
    U_BY_DX2 = 1963 / 3400 / DX2

    U_BY_DX1_conforming = 1 / DX1_conforming
    U_BY_DX2_conforming = 1963 / 3400 / DX2_conforming

    U_BY_DX1_blowup = 1 / DX1_blowup
    U_BY_DX2_blowup = 1963 / 3400 / DX2_blowup

    for m1param, m2param, err, kwargs in [
        (U_BY_DX1, U_BY_DX2, ERR, {"marker": "v", "label": "nonconforming"}),
        (
            U_BY_DX1_conforming,
            U_BY_DX2_conforming,
            ERR_conforming,
            {"marker": "^", "label": "conforming"},
        ),
    ]:
        ax[0].scatter(
            m1param, np.log10(err), c=np.log10(err), vmin=-3, vmax=1, **kwargs
        )
        ax[1].scatter(
            m2param, np.log10(err), c=np.log10(err), vmin=-3, vmax=1, **kwargs
        )
        m1scale = 1
        m2scale = 1
        ax[2].scatter(
            np.where(
                m1param * m1scale < m2param * m2scale,
                m1param * m1scale,
                m2param * m2scale,
            ),
            np.log10(err),
            c=np.log10(err),
            vmin=-3,
            vmax=1,
            **kwargs,
        )

        m1scale = 1500 / 3400
        m2scale = 1
        ax[3].scatter(
            np.where(
                m1param * m1scale < m2param * m2scale,
                m1param * m1scale,
                m2param * m2scale,
            ),
            np.log10(err),
            c=np.log10(err),
            vmin=-3,
            vmax=1,
            **kwargs,
        )

    fig.suptitle("seismogram RMS error from reference simulation")
    ax[0].set_title(r"vs. solid $v_p / \Delta x$")
    ax[1].set_title(r"vs. fluid $v_p / \Delta x$")
    ax[2].set_title(r"vs. min $v_p / \Delta x$")
    ax[3].set_title(r"vs. min $v / \Delta x$ (incl. $v_s$)")

    for i in range(4):
        ax[i].set_ylim(None, 1)
    ax[0].set_xlabel(r"solid $v_p / \Delta x$")
    ax[0].set_ylabel(r"$\log_{10} error$")
    ax[1].set_xlabel(r"fluid $v_p / \Delta x$")
    ax[1].set_ylabel(r"$\log_{10} error$")
    ax[2].set_xlabel(r"$min(v_p / \Delta x)$")
    ax[2].set_ylabel(r"$\log_{10} error$")
    ax[3].set_xlabel(r"$min(v / \Delta x)$")
    ax[3].set_ylabel(r"$\log_{10} error$")

    # ax[0].legend()

    fig.tight_layout()

    fig.savefig(outfol / f"errs_veldx_{iconf}.png")
    fig.clf()


if __name__ == "__main__":
    # plot_configs()
    for i in range(1):
        plot_errs(i)
