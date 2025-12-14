import numpy as np

from ...util.GLL import GLL


class FieldRemapper:
    """
    Class for mapping fields of X to Y, by providing the coefficients in the different basis.
    """

    def __init__(self, pts_a, pts_b):
        # pts indices = (ispec, iz, ix, icomp)
        _nspec_a, ngllz_a, ngllx_a, ncomp_a = pts_a.shape
        nspec_b, ngllz_b, ngllx_b, ncomp_b = pts_b.shape
        ndim = 2
        assert ncomp_a == ndim, "pts must both be 2D"
        assert ncomp_b == ndim, "pts must both be 2D"

        gllx = GLL(ngllx_a)
        gllz = GLL(ngllz_a)
        self.gllx = gllx
        self.gllz = gllz

        ispec_closest = np.empty((nspec_b, ngllz_b, ngllx_b), dtype=int)
        # batch all b points in each element: find the closest a-element for each.
        # we assume b is in an a-element if distance to lerp(spec_point,spec_center, eps) is minimized
        a_centers = np.mean(pts_a, axis=(1, 2))
        b_centers = np.mean(pts_b, axis=(1, 2))
        shft = 1e-2
        x_shft = shft * a_centers[:, None, None, :] + (1 - shft) * pts_a
        y_shft = shft * b_centers[:, None, None, :] + (1 - shft) * pts_b
        for i in range(nspec_b):
            ispec_closest[i, ...] = np.argmin(
                np.min(
                    np.linalg.norm(
                        y_shft[i, np.newaxis, np.newaxis, np.newaxis, :, :, :]
                        - x_shft[:, :, :, np.newaxis, np.newaxis, :],
                        ord=2,
                        axis=-1,
                    ),  # (ispecx,izx,ixx,izy,ixy)
                    axis=(1, 2),
                ),  # (ispecx,izy,ixy)
                axis=0,
            )
        self.ispec_closest = ispec_closest

        y_interpolants = pts_a[ispec_closest, :, :, :]

        # optimize with newtons
        def refine(coords, maxiters, coordtol=1e-8, gradtol=1e-10):
            # minimize: f = 1/2 |y_interpolants[...,a,b] L_b(xi) L_a(gamma) - pts_y|^2
            # d_err = |y_interpolants[...,a,b] L_b(xi) L_a(gamma) - pts_y|

            # write Ui as the interpolation function y_interpolants[...,a,b,i] L_b(xi) L_a(gamma)
            # dj f = (Ui - yi) * (dj Ui)
            # dk dj f = (dk Ui) * (dj Ui) + (Ui - yi) * (dk dj Ui)
            for _ in range(maxiters):
                xi_pows = coords[..., 0, np.newaxis] ** np.arange(gllx.nquad)
                ga_pows = coords[..., 1, np.newaxis] ** np.arange(gllz.nquad)

                # U - y
                Umy = (
                    np.einsum(
                        "...abi,aj,bk,...j,...k->...i",
                        y_interpolants,
                        gllz.L,
                        gllx.L,
                        ga_pows,
                        xi_pows,
                    )
                    - pts_b
                )

                maxerr = np.max(np.einsum("...i,...j->...", Umy, Umy))
                if maxerr < coordtol:
                    return

                dU = np.empty((nspec_b, ngllz_b, ngllx_b, 2, 2, 1))
                dU[..., 0, 0] = np.einsum(
                    "...abi,aj,bk,...j,...k->...i",
                    y_interpolants,
                    gllz.L,
                    gllx.Lp,
                    ga_pows,
                    xi_pows[..., :-1],
                )
                dU[..., 1, 0] = np.einsum(
                    "...abi,aj,bk,...j,...k->...i",
                    y_interpolants,
                    gllz.Lp,
                    gllx.L,
                    ga_pows[..., :-1],
                    xi_pows,
                )

                maxgrad = np.max(np.einsum("...ia,...ja->...", dU, dU))
                if maxgrad < gradtol:
                    return

                # hess of f (not U)
                H = np.einsum("...bia,...bja->...ij", dU, dU)
                H[..., 0, 0] += np.einsum(
                    "...abi,aj,bk,...j,...k,...i->...",
                    y_interpolants,
                    gllz.L,
                    gllx.Lpp,
                    ga_pows,
                    xi_pows[..., :-2],
                    Umy,
                )
                H[..., 1, 0] += np.einsum(
                    "...abi,aj,bk,...j,...k,...i->...",
                    y_interpolants,
                    gllz.Lp,
                    gllx.Lp,
                    ga_pows[..., :-1],
                    xi_pows[..., :-1],
                    Umy,
                )
                H[..., 0, 1] = H[..., 1, 0]
                H[..., 1, 1] += np.einsum(
                    "...abi,aj,bk,...j,...k,...i->...",
                    y_interpolants,
                    gllz.Lpp,
                    gllx.L,
                    ga_pows[..., :-2],
                    xi_pows,
                    Umy,
                )

                coords -= np.linalg.solve(
                    H, np.einsum("...ijk,...i->...jk", dU, Umy)
                )[..., 0]

        ngllz_b = 1
        coords = np.empty((nspec_b, ngllz_b, ngllx_b, 2))
        # initial guess: closest point
        indZ, indX = np.unravel_index(
            np.argmin(
                np.linalg.norm(
                    y_interpolants - pts_b[..., None, None, :], ord=2, axis=-1
                ).reshape((nspec_b, ngllz_b, ngllx_b, ngllz_a * ngllx_a)),
                axis=-1,
            ),
            (ngllz_a, ngllx_a),
        )
        coords[..., 0] = gllx.knots[indX]
        coords[..., 1] = gllz.knots[indZ]
        refine(coords, 20)

        self.lag_xi_closest = np.einsum(
            "aj,...j->...a",
            gllx.L,
            coords[..., 0, None] ** np.arange(gllx.nquad),
        )
        self.lag_ga_closest = np.einsum(
            "aj,...j->...a",
            gllz.L,
            coords[..., 1, None] ** np.arange(gllz.nquad),
        )

    def transfer_field(self, field):
        """_summary_

        Args:
            field (ndarray): the field (indices [ispecy,iz,ix,...]) to map to Y nodes

        Returns:
            ndarray: the mapped field (indices [ispec,iz,ix,...] in Y nodes)
        """

        f_interpolants = field[self.ispec_closest, ...]
        return np.einsum(
            "suvab...,suva,suvb->suv...",
            f_interpolants,
            self.lag_ga_closest,
            self.lag_xi_closest,
        )

    def __call__(self, field):
        return self.transfer_field(field)
