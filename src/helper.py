import numpy as np
from inspect import signature
from numpy.linalg import inv
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def printRes(A, dA, prt=False):
    dDec = np.array(np.log10(dA), dtype=int)
    output = f"{np.round(A, -dDec+2)}({np.array(np.round(dA, -dDec+2)*10**(-dDec+2), dtype=int)})"
    if prt:
        print(output)
    return output



def delta(obs: np.ndarray, ax=0):
    return np.std(obs, ddof=1, axis=ax) / np.sqrt(obs.shape[ax])


def correlationErrorPerBinSize(
    correlator,
    lattice_times=np.array([0]),
    bin_sizes=np.arange(1, 100, 1, dtype=np.uint),
    path="data/CEPBS.csv",
):
    bin_count = np.array(np.floor(correlator.shape[1] / bin_sizes), dtype=np.uint)
    lengths = np.array(bin_count * bin_sizes, dtype=np.uint)
    deltas = np.zeros((lattice_times.shape[0], bin_sizes.shape[0]))
    for tidx in range(lattice_times.shape[0]):
        for i in range(bin_sizes.shape[0]):
            cropped_correlator = correlator[tidx, : int(lengths[i])]
            bined_correlator = np.reshape(
                cropped_correlator, (bin_count[i], bin_sizes[i])
            ).mean(1)
            deltas[tidx, i] = delta(bined_correlator)
    np.savetxt(path, deltas.T, delimiter=",")
    return deltas, (lattice_times, bin_sizes, path)


def calcCorrelation(
    x: np.ndarray,
    y: np.ndarray,
    func,
    p0,
    bin_size=1,
    resamples=100,
    options = {'xtol': 1e-14, 'ftol': 1e-14, 'gtol': 1e-14, 'maxfev': 2000}
):

    # cutting and binning data
    if bin_size != 1:
        bin_count = int(y.shape[1] / bin_size)
        length = bin_count * bin_size
        y = y[:, :length]
        y = np.reshape(
            y, (y.shape[0], bin_count, bin_size)
        ).mean(2)

    # setting arrays
    sigma = np.cov(y)/y.shape[1]
    print(y.shape[1])
    popt = np.zeros((resamples + 1, len(p0)))
    pcov = np.zeros((resamples + 1, len(p0), len(p0)))
    red_chi_sq = np.zeros(resamples + 1)
    ydata = y.mean(1)

    # first fit
    popt[0], pcov[0] = curve_fit(
        func,
        x,
        ydata,
        p0=p0,
        sigma=sigma,
        absolute_sigma=False,
        **options
    )
    
    # chisq calc with svd
    r = ydata - func(x, *popt[0])
    chi_sq = r.T @ np.linalg.pinv(sigma) @ r
    red_chi_sq[0] = chi_sq/(x.shape[0] - len(p0)) # reduced chisq
    if resamples == 0:
        return (popt, pcov, red_chi_sq)

    # Bootstraping
    for i in range(1, resamples+1):
        ysample = y[:, np.random.choice(y.shape[1], size=y.shape[1], replace=True)]
        ydata = ysample.mean(1)
        sigma = np.cov(ysample)/ysample.shape[1]

        # fit to sample
        popt[i], pcov[i] = curve_fit(
            func,
            x,
            ydata,
            p0=p0,
            sigma=sigma,
            absolute_sigma=False,
            **options
        )

        r = ydata - func(x, *popt[i])
        red_chi_sq[i] = r.T @ np.linalg.pinv(sigma) @ r / (x.shape[0] - len(p0))


    return (
        popt,
        pcov,
        red_chi_sq,
    )


def plotCorrelator(corr, corr_err, path="plots/Correlator.pdf"):
    plt.errorbar(
        np.arange(corr.shape[0]),
        corr,
        yerr=corr_err,
        label="Correlators",
        marker="x",
        ls="none",
        linewidth=0.5,
    )
    plt.grid()
    plt.title("Average correlator")
    plt.ylabel(r"$\langle x(0)x(\tau)\rangle$")
    plt.xlabel(r"$\tau$")
    plt.yscale("log")
    plt.legend()
    plt.savefig(path, dpi=500)
    plt.clf()


def plotCEPBS(
    corr,
    datapath,
    figpath="plots/CEPBS.pdf",
    lattice_times=np.arange(0, 6, 1),
    bin_sizes=np.arange(1, 11, 1),
):
    deltas, (lattice_times, bin_sizes, _) = helper.correlationErrorPerBinSize(
        data, path=datapath, lattice_times=latticre_times, bin_sizes=bin_sizes
    )
    for i in range(deltas.shape[0]):
        plt.scatter(bin_sizes, deltas[i], marker="x", label=f"$t=${lattice_times[i]}")

    plt.xlabel(r"Bin sizes $N_B$")
    plt.ylabel(r"Uncertenty $\Delta$")
    plt.title(r"Estimation of proper $N_B$")
    plt.legend()
    plt.savefig(figpath, dpi=500)
    plt.clf()
