#!/usr/bin/env python
"""Task 4 (resolution extension): validate the joint (v, R) fit and the handling
of a different wavelength array.

Headline tests (leave-one-out held-out star, exact linear-flux broadening in the
toy vs. the fitter's log-space broadening -- so any (v, R) bias from that
approximation is measured honestly):

  1. joint (v, R) recovery vs SNR at a fixed true resolution;
  2. recovered R vs true R at fixed SNR;
  3. an example where the observation is delivered on a DIFFERENT (coarser,
     offset, sub-range) log-lambda grid, resampled onto the template grid, then
     fit for (v, R) -- demonstrating the "different wavelength array" case.

Writes figures to src/outputs/.
"""
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for sibling `validate`
from clammy import basis, toy, fitter, paths, C_LIGHT_KMS
from validate import build_loo, TEST_STAR, C_TRUE

OUT = paths.OUTPUTS
V_TRUE = 42.0


def run_joint(loglam, mu, Phi, r_true, v_true, R_true, snr, n_trials, seed0=0):
    vs, verr, Rs, Rerr, c2 = [], [], [], [], []
    for s in range(n_trials):
        d, sig, _ = toy.make_toy(loglam, r_true, v_true, C_TRUE, snr=snr,
                                 resolution_R=R_true, seed=seed0 + s)
        res = fitter.fit_rv(d, sig, loglam, mu, Phi, cont_order=3, fit_resolution=True,
                            R_bounds=(2500.0, 25000.0), vmin=-300, vmax=300)
        vs.append(res["v_kms"]); verr.append(res["v_err_kms"])
        Rs.append(res["resolution_R"]); Rerr.append(res["R_err"]); c2.append(res["chi2_dof"])
    return map(np.array, (vs, verr, Rs, Rerr, c2))


def fig_snr(b, R_true=6000.0, snrs=(10, 20, 40, 80, 160), n_trials=12):
    K = b["K99"]; Phi = b["Phi_full"][:K]; mu, loglam, r_true = b["mu"], b["loglam"], b["r_true"]
    rows = []
    for snr in snrs:
        vs, verr, Rs, Rerr, c2 = run_joint(loglam, mu, Phi, r_true, V_TRUE, R_true, snr, n_trials)
        rows.append(dict(snr=snr, vb=vs.mean() - V_TRUE, vsc=vs.std(), vpe=np.nanmean(verr),
                         Rb=np.nanmean(Rs) - R_true, Rsc=np.nanstd(Rs), Rpe=np.nanmean(Rerr),
                         c2=np.nanmean(c2)))
        print(f"  SNR={snr:4d}: v bias={1000*rows[-1]['vb']:+7.1f} m/s (sc {1000*rows[-1]['vsc']:.1f}, "
              f"pred {1000*rows[-1]['vpe']:.1f}) | R={np.nanmean(Rs):.0f} bias={rows[-1]['Rb']:+.0f} "
              f"({100*rows[-1]['Rb']/R_true:+.1f}%, sc {rows[-1]['Rsc']:.0f}, pred {rows[-1]['Rpe']:.0f}) | chi2/dof={rows[-1]['c2']:.2f}")
    snr = np.array([r["snr"] for r in rows], float)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].errorbar(snr, [1000 * r["vb"] for r in rows],
                      yerr=[1000 * r["vsc"] / np.sqrt(n_trials) for r in rows], fmt="o-")
    ax[0, 0].fill_between(snr, [-1000 * r["vsc"] for r in rows], [1000 * r["vsc"] for r in rows], alpha=0.2)
    ax[0, 0].axhline(0, color="k", lw=0.5); ax[0, 0].set_xscale("log")
    ax[0, 0].set_xlabel("SNR/px"); ax[0, 0].set_ylabel("RV bias [m/s]")
    ax[0, 0].set_title(f"joint (v,R): RV recovery (R_true={R_true:.0f})")

    ax[0, 1].errorbar(snr, [100 * r["Rb"] / R_true for r in rows],
                      yerr=[100 * r["Rsc"] / R_true / np.sqrt(n_trials) for r in rows], fmt="o-")
    ax[0, 1].axhline(0, color="k", lw=0.5)
    ax[0, 1].axhspan(-2.5, 0, color="orange", alpha=0.15, label="log-conv approx (~ -2%)")
    ax[0, 1].set_xscale("log"); ax[0, 1].set_xlabel("SNR/px"); ax[0, 1].set_ylabel("R bias [%]")
    ax[0, 1].set_title("resolution recovery"); ax[0, 1].legend(fontsize=8)

    ax[1, 0].loglog(snr, [1000 * r["vsc"] for r in rows], "o-", label="RV scatter")
    ax[1, 0].loglog(snr, [1000 * r["vpe"] for r in rows], "s--", label="RV pred err")
    ax[1, 0].loglog(snr, [r["Rsc"] for r in rows], "o-", label="R scatter")
    ax[1, 0].loglog(snr, [r["Rpe"] for r in rows], "s--", label="R pred err")
    ax[1, 0].set_xlabel("SNR/px"); ax[1, 0].set_ylabel("uncertainty (m/s ; R)")
    ax[1, 0].set_title("error calibration"); ax[1, 0].legend(fontsize=8)

    ax[1, 1].semilogx(snr, [r["c2"] for r in rows], "o-"); ax[1, 1].axhline(1, color="r", ls="--")
    ax[1, 1].set_xlabel("SNR/px"); ax[1, 1].set_ylabel("chi2/dof")
    ax[1, 1].set_title("goodness of fit")
    fig.tight_layout(); fig.savefig(f"{OUT}/validate_resolution_snr.png", dpi=120); plt.close(fig)
    print(f"  -> {OUT}/validate_resolution_snr.png")


def fig_R_vs_R(b, R_trues=(3000, 4500, 6000, 9000, 15000), snr=120, n_trials=8):
    K = b["K99"]; Phi = b["Phi_full"][:K]; mu, loglam, r_true = b["mu"], b["loglam"], b["r_true"]
    Rfit, Rsc = [], []
    for Rt in R_trues:
        vs, verr, Rs, Rerr, c2 = run_joint(loglam, mu, Phi, r_true, V_TRUE, float(Rt), snr, n_trials)
        Rfit.append(np.nanmean(Rs)); Rsc.append(np.nanstd(Rs))
        print(f"  R_true={Rt:6d}: R_fit={np.nanmean(Rs):7.0f} +/- {np.nanstd(Rs):4.0f}  ({100*(np.nanmean(Rs)-Rt)/Rt:+.1f}%)")
    Rt = np.array(R_trues, float)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.errorbar(Rt, Rfit, yerr=Rsc, fmt="o", ms=6, capsize=3, label="recovered")
    lim = [0.9 * Rt.min(), 1.1 * Rt.max()]
    ax.plot(lim, lim, "k--", lw=1, label="1:1")
    ax.set_xlabel("true R"); ax.set_ylabel("recovered R")
    ax.set_title(f"resolution recovery vs true R (SNR={snr})\n(slight low bias = log-space convolution approx)")
    ax.legend(); ax.set_xlim(lim); ax.set_ylim(lim)
    fig.tight_layout(); fig.savefig(f"{OUT}/validate_resolution_RvR.png", dpi=120); plt.close(fig)
    print(f"  -> {OUT}/validate_resolution_RvR.png")


def fig_resampled_example(b, R_true=6500.0, snr=40, v_true=42.0):
    """Observation delivered on a different log grid, resampled, then fit for (v,R)."""
    K = b["K99"]; Phi = b["Phi_full"][:K]; mu, loglam, r_true = b["mu"], b["loglam"], b["r_true"]
    # a different observed grid: coarser (1.5 km/s/px), offset, covering a sub-range
    velscale_obs = 1.5
    dln_obs = velscale_obs / C_LIGHT_KMS
    lo, hi = np.log(6600.0), np.log(9100.0)
    obs_loglam = np.arange(lo + 0.3 * dln_obs, hi, dln_obs)
    d_obs, sig_obs, tr = toy.make_toy(loglam, r_true, v_true, C_TRUE, snr=snr,
                                      resolution_R=R_true, out_loglam=obs_loglam, seed=5)
    # resample observation onto the template grid for fitting
    d, sig, good = toy.resample_to_loglam(obs_loglam, d_obs, sig_obs, loglam)
    res = fitter.fit_rv(d, sig, loglam, mu, Phi, cont_order=3, fit_resolution=True,
                        R_bounds=(2500.0, 25000.0), mask=good, vmin=-300, vmax=300)
    print(f"  resampled example: v_true={v_true} R_true={R_true:.0f} -> "
          f"v={res['v_kms']:.3f}+/-{res['v_err_kms']:.3f} km/s, R={res['resolution_R']:.0f}+/-{res['R_err']:.0f}, "
          f"chi2/dof={res['chi2_dof']:.2f}")

    lam = np.exp(loglam); sl = slice(0, len(lam), 6)
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0, 0].plot(np.exp(obs_loglam), d_obs, ".", ms=1, color="0.6", label=f"obs grid ({velscale_obs} km/s/px)")
    ax[0, 0].plot(lam[sl], res["model"][sl], lw=0.5, color="C3", label="best-fit model")
    ax[0, 0].set_xlim(6600, 9100); ax[0, 0].set_ylabel("flux"); ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_title(f"degraded (R={R_true:.0f}) + resampled obs; fit v={res['v_kms']:.2f}, R={res['resolution_R']:.0f}")

    rr = res["resid_lnd"]; ax[1, 0].plot(lam[sl], rr[sl], lw=0.4); ax[1, 0].axhline(0, color="k", lw=0.5)
    sd = np.nanstd(rr); ax[1, 0].set_ylim(-5 * sd, 5 * sd); ax[1, 0].set_xlim(6600, 9100)
    ax[1, 0].set_xlabel("wavelength [A]"); ax[1, 0].set_ylabel("ln-flux residual")

    # 2-D chi2(v,R) surface from the coarse scan
    chi2_2d = res["chi2_2d"]; sp = res["sp_grid"]
    Rgrid = res["R_grid"]  # = R for sp_grid[1:] (excludes the sigma=0 row)
    vgrid = res["v_grid"]
    extent = [vgrid.min(), vgrid.max(), 0, len(sp) - 1]
    im = ax[0, 1].imshow(chi2_2d - chi2_2d.min(), aspect="auto", origin="lower",
                         extent=extent, vmax=np.percentile(chi2_2d - chi2_2d.min(), 40))
    ax[0, 1].set_xlabel("v [km/s]"); ax[0, 1].set_ylabel("R grid index")
    ax[0, 1].set_title("coarse chi2(v, R) surface"); ax[0, 1].axvline(v_true, color="g", ls=":")
    plt.colorbar(im, ax=ax[0, 1], label="chi2 - min")

    ax[1, 1].plot(Rgrid, chi2_2d[1:].min(axis=1) - chi2_2d.min(), "o-")
    ax[1, 1].axvline(R_true, color="g", ls=":", label="R_true")
    ax[1, 1].axvline(res["resolution_R"], color="r", ls="--", label="R_fit")
    ax[1, 1].set_xlabel("R"); ax[1, 1].set_ylabel("profiled chi2 - min"); ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_title("chi2 vs R (profiled over v, w, c)")
    fig.tight_layout(); fig.savefig(f"{OUT}/validate_resolution_resampled.png", dpi=120); plt.close(fig)
    print(f"  -> {OUT}/validate_resolution_resampled.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    print(f"building leave-one-out basis (excluding {TEST_STAR}) ...")
    b = build_loo(TEST_STAR)
    print(f"  K99={b['K99']}; {time.time()-t0:.1f}s")
    print("\n[1] joint (v,R) recovery vs SNR:")
    fig_snr(b)
    print("\n[2] recovered R vs true R:")
    fig_R_vs_R(b)
    print("\n[3] different-wavelength-grid (resampled) example:")
    fig_resampled_example(b)
    print(f"\nresolution validation done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
