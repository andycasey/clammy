#!/usr/bin/env python
"""Task 4: validate the fitter on toy data with known truth.

Headline test is leave-one-out: a held-out grid star is fit with a PCA basis
built from the *other* 197 spectra, so the basis-truncation / template-mismatch
error is included honestly (this is the realistic regime). We characterise:

  1. recovery of v, w, c versus SNR (bias, scatter, error calibration, chi2/dof);
  2. an example fit with residuals and the chi2(v) curve;
  3. RV bias as a function of the true velocity;
  4. sweeps over continuum order and basis size, to expose biases/degeneracies.

Figures are written to outputs/.
"""
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from clammy import grid, basis, toy, fitter, paths

OUT = paths.OUTPUTS
# A metal-poor RGB star -- representative of dwarf-galaxy DEIMOS targets.
TEST_STAR = "dmost_lte_4500_1.0_-2.0_.fits"
RECT_ORDER = 5
C_TRUE = np.array([0.6, -0.30, 0.12, -0.05])  # true log-continuum (Legendre, L=3)
CONT_ORDER = 3


def build_loo(test_star, max_k=30):
    """Leave-one-out basis (max_k components) + the held-out star's truth."""
    loglam, flux, params, files = grid.load_grid(paths.CONVOLVED, exclude={test_star})
    R, _ = basis.rectify_grid(loglam, flux, order=RECT_ORDER)
    mu, Phi_full, info = basis.build_pca(R, var_target=1.0, max_k=max_k)
    K99 = int(np.searchsorted(info["cumvar"], 0.99) + 1)

    w0, fl_test = grid.load_one(os.path.join(paths.CONVOLVED, test_star))
    R_test, cont_test = basis.rectify_grid(loglam, fl_test[None, :], order=RECT_ORDER)
    r_true = R_test[0]
    return dict(loglam=loglam, mu=mu, Phi_full=Phi_full, K99=K99,
                r_true=r_true, info=info)


def run_trials(loglam, mu, Phi, r_true, v_true, snr, n_trials, cont_order=CONT_ORDER,
               w_truth=None, seed0=0):
    """Fit n_trials noisy realisations; return per-trial arrays."""
    vs, verr, chi2dof, dw, dc = [], [], [], [], []
    for s in range(n_trials):
        d, sig, tr = toy.make_toy(loglam, r_true, v_true, C_TRUE, snr=snr, seed=seed0 + s)
        res = fitter.fit_rv(d, sig, loglam, mu, Phi, cont_order=cont_order,
                            vmin=-400, vmax=400)
        vs.append(res["v_kms"]); verr.append(res["v_err_kms"])
        chi2dof.append(res["chi2_dof"])
        m = min(len(res["c"]), len(C_TRUE))  # orders may differ in the sweep
        dc.append(np.max(np.abs(res["c"][:m] - C_TRUE[:m])))
        if w_truth is not None:
            dw.append(np.max(np.abs(res["w"] - w_truth)))
    return (np.array(vs), np.array(verr), np.array(chi2dof),
            np.array(dw) if dw else None, np.array(dc))


def fig_snr_recovery(b, snrs=(5, 10, 20, 40, 80, 160), n_trials=30, v_true=42.0):
    K = b["K99"]
    Phi = b["Phi_full"][:K]
    mu, loglam, r_true = b["mu"], b["loglam"], b["r_true"]
    w_truth = toy.project_onto_basis(mu, Phi, r_true)  # recoverable weight truth

    rows = []
    for snr in snrs:
        vs, verr, chi2dof, dw, dc = run_trials(loglam, mu, Phi, r_true, v_true, snr,
                                               n_trials, w_truth=w_truth)
        rows.append(dict(snr=snr, bias=vs.mean() - v_true, scat=vs.std(),
                         perr=verr.mean(), chi2=chi2dof.mean(),
                         dw=dw.mean(), dc=dc.mean()))
        print(f"  SNR={snr:4d}: v bias={1000*(vs.mean()-v_true):+8.2f} m/s  "
              f"scatter={1000*vs.std():8.2f}  pred_err={1000*verr.mean():8.2f} m/s  "
              f"chi2/dof={chi2dof.mean():.3f}  max|dw|={dw.mean():.3f}  max|dc|={dc.mean():.4f}")
    snr = np.array([r["snr"] for r in rows], float)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    a = ax[0, 0]
    a.errorbar(snr, [1000 * r["bias"] for r in rows],
               yerr=[1000 * r["scat"] / np.sqrt(n_trials) for r in rows],
               fmt="o-", label="bias +/- err on mean")
    a.fill_between(snr, [-1000 * r["scat"] for r in rows], [1000 * r["scat"] for r in rows],
                   alpha=0.2, label="+/- 1 scatter")
    a.axhline(0, color="k", lw=0.5); a.set_xscale("log")
    a.set_xlabel("SNR / px"); a.set_ylabel("RV bias [m/s]")
    a.set_title(f"RV recovery (LOO, v_true={v_true} km/s)"); a.legend(fontsize=8)

    a = ax[0, 1]
    a.loglog(snr, [1000 * r["scat"] for r in rows], "o-", label="realised scatter")
    a.loglog(snr, [1000 * r["perr"] for r in rows], "s--", label="predicted error")
    a.set_xlabel("SNR / px"); a.set_ylabel("RV uncertainty [m/s]")
    a.set_title("error calibration"); a.legend(fontsize=8)

    a = ax[1, 0]
    a.semilogx(snr, [r["chi2"] for r in rows], "o-")
    a.axhline(1.0, color="r", ls="--")
    a.set_xlabel("SNR / px"); a.set_ylabel("chi2 / dof")
    a.set_title("goodness of fit (>1 at high SNR = basis truncation floor)")

    a = ax[1, 1]
    a.loglog(snr, [r["dw"] for r in rows], "o-", label="max |w - w_true|")
    a.loglog(snr, [r["dc"] for r in rows], "s-", label="max |c - c_true|")
    a.set_xlabel("SNR / px"); a.set_ylabel("max abs error")
    a.set_title("template-weight & continuum recovery"); a.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{OUT}/validate_snr_recovery.png", dpi=120)
    plt.close(fig)
    print(f"  -> {OUT}/validate_snr_recovery.png")


def fig_example(b, snr=25, v_true=42.0):
    K = b["K99"]
    Phi = b["Phi_full"][:K]
    mu, loglam, r_true = b["mu"], b["loglam"], b["r_true"]
    d, sig, tr = toy.make_toy(loglam, r_true, v_true, C_TRUE, snr=snr, seed=7)
    res = fitter.fit_rv(d, sig, loglam, mu, Phi, cont_order=CONT_ORDER, vmin=-400, vmax=400)
    lam = np.exp(loglam)
    sl = slice(0, len(lam), 8)

    fig, ax = plt.subplots(3, 1, figsize=(13, 9))
    ax[0].plot(lam[sl], d[sl], lw=0.4, color="0.5", label="data")
    ax[0].plot(lam[sl], res["model"][sl], lw=0.5, color="C3", label="model")
    ax[0].set_ylabel("flux"); ax[0].legend()
    ax[0].set_title(f"example fit  SNR={snr}, v_true={v_true} -> "
                    f"v_fit={res['v_kms']:.3f}+/-{res['v_err_kms']:.3f} km/s, "
                    f"chi2/dof={res['chi2_dof']:.3f}")

    r = res["resid_lnd"]
    ax[1].plot(lam[sl], r[sl], lw=0.4)
    ax[1].axhline(0, color="k", lw=0.5)
    sd = np.nanstd(r)
    ax[1].set_ylim(-6 * sd, 6 * sd)
    ax[1].set_ylabel("ln-flux residual"); ax[1].set_xlabel("wavelength [A]")

    a = ax[2]
    a.plot(res["v_grid"], res["chi2_grid"] - res["chi2_grid"].min(), lw=0.8)
    a.axvline(v_true, color="g", ls=":", label="v_true")
    a.axvline(res["v_kms"], color="r", ls="--", label="v_fit")
    a.set_xlim(v_true - 60, v_true + 60)
    a.set_ylim(0, np.percentile(res["chi2_grid"] - res["chi2_grid"].min(), 60))
    a.set_xlabel("v [km/s]"); a.set_ylabel("chi2 - min"); a.set_title("RV scan")
    a.legend()

    fig.tight_layout()
    fig.savefig(f"{OUT}/validate_example_fit.png", dpi=120)
    plt.close(fig)
    print(f"  -> {OUT}/validate_example_fit.png")


def fig_v_sweep(b, vtrues=None, snr=30, n_trials=12):
    if vtrues is None:
        vtrues = np.linspace(-300, 300, 25)
    K = b["K99"]
    Phi = b["Phi_full"][:K]
    mu, loglam, r_true = b["mu"], b["loglam"], b["r_true"]
    bias, scat = [], []
    for v in vtrues:
        vs, *_ = run_trials(loglam, mu, Phi, r_true, v, snr, n_trials)
        bias.append(vs.mean() - v); scat.append(vs.std())
    bias = np.array(bias); scat = np.array(scat)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(vtrues, 1000 * bias, yerr=1000 * scat / np.sqrt(n_trials), fmt="o-")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("true v [km/s]"); ax.set_ylabel("RV bias [m/s]")
    ax.set_title(f"RV bias vs true velocity (LOO, SNR={snr})")
    fig.tight_layout()
    fig.savefig(f"{OUT}/validate_v_sweep.png", dpi=120)
    plt.close(fig)
    print(f"  RV-sweep: max|bias|={1000*np.max(np.abs(bias)):.1f} m/s, "
          f"median scatter={1000*np.median(scat):.1f} m/s")
    print(f"  -> {OUT}/validate_v_sweep.png")


def fig_sweeps(b, snr=30, n_trials=20, v_true=42.0):
    mu, loglam, r_true = b["mu"], b["loglam"], b["r_true"]

    # (a) continuum-order sweep at fixed K
    Phi = b["Phi_full"][: b["K99"]]
    orders = [0, 1, 2, 3, 4, 6, 8]
    ob, osc, oc2 = [], [], []
    for L in orders:
        vs, verr, c2, _, _ = run_trials(loglam, mu, Phi, r_true, v_true, snr, n_trials,
                                        cont_order=L)
        ob.append(vs.mean() - v_true); osc.append(vs.std()); oc2.append(c2.mean())

    # (b) basis-size sweep at fixed continuum order
    Ks = [1, 2, 4, 6, 8, 10, 14, 20, 30]
    kb, ksc, kc2 = [], [], []
    for K in Ks:
        Phi_k = b["Phi_full"][:K]
        vs, verr, c2, _, _ = run_trials(loglam, mu, Phi_k, r_true, v_true, snr, n_trials)
        kb.append(vs.mean() - v_true); ksc.append(vs.std()); kc2.append(c2.mean())

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].errorbar(orders, 1000 * np.array(ob),
                      yerr=1000 * np.array(osc) / np.sqrt(n_trials), fmt="o-")
    ax[0, 0].axhline(0, color="k", lw=0.5)
    ax[0, 0].set_xlabel("continuum order L"); ax[0, 0].set_ylabel("RV bias [m/s]")
    ax[0, 0].set_title(f"continuum-order sweep (K={b['K99']}, SNR={snr})")

    ax[0, 1].plot(orders, oc2, "o-"); ax[0, 1].axhline(1, color="r", ls="--")
    ax[0, 1].set_xlabel("continuum order L"); ax[0, 1].set_ylabel("chi2/dof")
    ax[0, 1].set_title("continuum order vs chi2/dof")

    ax[1, 0].errorbar(Ks, 1000 * np.array(kb),
                      yerr=1000 * np.array(ksc) / np.sqrt(n_trials), fmt="o-")
    ax[1, 0].axhline(0, color="k", lw=0.5)
    ax[1, 0].axvline(b["K99"], color="g", ls=":", label=f"K99={b['K99']}")
    ax[1, 0].set_xlabel("# PCA templates K"); ax[1, 0].set_ylabel("RV bias [m/s]")
    ax[1, 0].set_title("basis-size sweep"); ax[1, 0].legend()

    ax[1, 1].plot(Ks, kc2, "o-"); ax[1, 1].axhline(1, color="r", ls="--")
    ax[1, 1].axvline(b["K99"], color="g", ls=":")
    ax[1, 1].set_xlabel("# PCA templates K"); ax[1, 1].set_ylabel("chi2/dof")
    ax[1, 1].set_title("basis size vs chi2/dof (truncation floor)")

    fig.tight_layout()
    fig.savefig(f"{OUT}/validate_sweeps.png", dpi=120)
    plt.close(fig)
    print(f"  cont-order chi2/dof: {dict(zip(orders, np.round(oc2,3)))}")
    print(f"  basis-size chi2/dof: {dict(zip(Ks, np.round(kc2,3)))}")
    print(f"  -> {OUT}/validate_sweeps.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    print(f"building leave-one-out basis (excluding {TEST_STAR}) ...")
    b = build_loo(TEST_STAR)
    print(f"  K99={b['K99']} components; built in {time.time()-t0:.1f}s")

    print("\n[1] SNR recovery:")
    fig_snr_recovery(b)
    print("\n[2] example fit:")
    fig_example(b)
    print("\n[3] RV vs true-velocity sweep:")
    fig_v_sweep(b)
    print("\n[4] continuum-order & basis-size sweeps:")
    fig_sweeps(b)
    print(f"\nall validation done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
