import sys
import numpy as np
import re
import pandas as pd
import h5py as h5
import matplotlib.pyplot as plt
import helper

# fit function
def func(x, *args):
    return args[0] * np.cosh(-args[1] * (x-80))


path = "data/p2gg_local_neutral_light.p-lvc-lvc.fl1.qx0_qy0_qz0.gseq_4.tseq_15.px0_py-2_pz2.h5"
# load and order data
f = h5.File(path, "r")
px = int(path.split("px")[1].split("_")[0])
py = int(path.split("py")[1].split("_")[0])
pz = int(path.split("pz")[1].split(".")[0])
p = np.array([px, py, pz])
T = 160

file = f["stream_a/"]
data = []

for keyD0 in file:
    config = []
    for keyD1 in file[keyD0]:
        t = int(keyD1.split("t")[1].split("_")[0])
        x = int(keyD1.split("x")[1].split("_")[0])
        y = int(keyD1.split("y")[1].split("_")[0])
        z = int(keyD1.split("z")[1])
        r = np.array([x, y, z])

        arr = file[keyD0][keyD1][:]
        arr = arr[:, :, ::2]                        # nur den Realteil
        arr = arr * np.cos(-p@r)

        config.append(np.roll(arr, shift=-t, axis=-1))

    data.append(config)

data = np.array(data)
data = np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2], data.shape[3], data.shape[4]))

p3f_std = data.std(0, ddof=1) / np.sqrt(data.shape[0])
p3f = data.mean(0)

# -- making data export ready for plot --
p3f_pd = np.array([p3f, p3f_std])
p3f_pd = np.reshape(p3f_pd, (p3f_pd.shape[0]*p3f_pd.shape[1]*p3f_pd.shape[2], p3f_pd.shape[3])).T
temp = np.array(p3f_pd)
p3f_pd[:,::2] = temp[:,:16]
p3f_pd[:,1::2] = temp[:,16:]
print(p3f_pd.shape)
np.savetxt("data/p3f_pd.dat", p3f_pd)
# ---------------------------------------



exit()

for keyD0 in file:
    config = []
    for keyD1 in file[keyD0]:
        t = int(keyD1.split("t")[1].split("_")[0])
        # print(t)
        arr = file[keyD0][keyD1][:].squeeze()
        arr = arr[::2]
        config.append(np.roll(arr, shift=-t))
        # data.append(arr)
    data.append(config)




# reorder and fold data 
data = np.array(data).squeeze()
data = -1 * data

first_half_data = np.ndarray([data.shape[0], data.shape[1], int(data.shape[2] / 2) + 1])
second_half_data = np.ndarray(
    [data.shape[0], data.shape[1], int(data.shape[2] / 2) + 1]
)
first_half_data[:, :, :-1] = data[:, :, : int(data.shape[2] / 2)]
first_half_data[:, :, -1] = data[:, :, int(data.shape[2] / 2)]
second_half_data[:, :, :-1] = data[:, :, int(data.shape[2] / 2) :]
second_half_data[:, :, -1] = data[:, :, 0]
second_half_data = np.flip(second_half_data, 2)

data = np.array([first_half_data, second_half_data]).mean(0)
data_err = data.std(1, ddof=1)
data = data.mean(1)

data = data.T
data_err = data_err.T


# helper.plotCEPBS(corr, path=argv[1], lattice_times=np.arange(0, 6, 1), bin_sizes=np.arange(1, 11, 1))

'''
stop = 80
start = np.arange(1, 75)

chi_sq = np.zeros(start.shape[0])
param_E = np.zeros(start.shape[0])
dChi_sq = np.zeros(start.shape[0])
dE = np.zeros(start.shape[0])

for i in range(start.shape[0]):
    print(i)
    _, _, a = helper.calcCorrelation(
        np.arange(start[i], stop, 1),
        data[start[i] : stop, :],
        func,
        p0 = [3e-3, 0.047],
        resamples=0,
        bin_size=1,
    )
    chi_sq[i] = a[0]

np.savetxt("../data/LowerBound.csv", np.array([start, chi_sq]).T, delimiter=" ")
exit()
'''
'''
start = 16
stop = np.arange(80, 40, -1)
print(stop)

chi_sq = np.zeros(stop.shape[0])
param_E = np.zeros(stop.shape[0])
dChi_sq = np.zeros(stop.shape[0])
dE = np.zeros(stop.shape[0])

for i in range(stop.shape[0]):
    print(i)
    _, _, all_chi_sq = helper.calcCorrelation(
        np.arange(start, stop[i], 1),
        data[start : stop[i], :],
        func,
        [3e-3, 0.047],
        resamples=0,
        bin_size=1,
    )
    chi_sq[i] = all_chi_sq[0]

np.savetxt(
    "../data/UpperBound.csv",
    np.array([stop, chi_sq]).T,
    delimiter=" ",
)
'''

"""
plt.scatter(start, ddE, marker="x")
plt.xlabel(r"Lower bound $t_{\textrm{lower}}$")
plt.ylabel(r"$\delta E$")
plt.yscale("log")
plt.title("Uncertenty")
plt.savefig("plots/LowerBound.pdf", dpi=500)
#plt.show()
plt.clf()
"""

"""

end = np.arange(20, 100)
ddE = np.zeros(end.shape[0])
for i in range(end.shape[0]):
    _, _, ddE[i], _, _ = helper.calcCorrelation(data[5:end[i]], lattice_time=np.arange(5, end[i], 1), resamples=100, isCorr=True, bin_size=16)

plt.scatter(end, ddE, marker="x")
plt.xlabel(r"Upper bound $t_{\textrm{upper}}$")
plt.ylabel(r"$\delta E$")
plt.yscale("log")
plt.title("Uncertenty for upper bound")
plt.savefig("plots/UpperBound.pdf", dpi=500)
#plt.show()
plt.clf()

print(ddE)
"""
'''

start = 30
stop = 60
param_C = np.zeros(1000)
rand_p0 = np.random.rand(1000)*1e-1
#print(rand_p0)
for i in range(1000):
    popt, _, _ = helper.calcCorrelation(
        np.arange(start, stop, 1),
        data[start:stop, :],
        func,
        [0 + rand_p0[i], 0.047],
        bin_size=1,
        resamples=0,
    )
    param_C[i] = popt[0, 0]

print(param_C)
print(param_C[param_C > 3.37e-3])
exit()

'''

# fit and bootstrap parameters
start = 16
stop = 64
popt, pcov, chisq = helper.calcCorrelation(
    np.arange(start, stop, 1),
    data[start:stop, :],
    func,
    [3.38e-3, 0.047],
    bin_size=1,
    resamples=1000,
)

# Setzte alles für den Output
popt = popt.T
samp_popt_mean = np.mean(popt[:, 1:], axis=1)
print(popt.shape)
samp_popt_std = np.std(popt[:, 1:], axis=1, ddof=1)
print(samp_popt_std)
result = np.array([[0, popt[0, 0], np.sqrt(pcov[0, 0, 0]), samp_popt_mean[0], samp_popt_std[0], (popt[0, 0]-samp_popt_mean[0])/popt[0, 0], samp_popt_std[0]/popt[0, 0]], 
                   [0, popt[1, 0], np.sqrt(pcov[0, 1, 1]), samp_popt_mean[1], samp_popt_std[1], (popt[1, 0]-samp_popt_mean[1])/popt[1, 0], samp_popt_std[1]/popt[1, 0]]])

np.savetxt("../data/results", result, delimiter=", ", fmt='%0.4e')

print("chisq/dof: ", chisq[0])




# Calculate predicted correlator
x = np.arange(start, stop)
pred = func(
    np.expand_dims(x, 1),
    np.tile(popt[0, 1:], x.shape[0]).T,
    np.tile(popt[1, 1:], x.shape[0]).T,
)
dPred = pred.std(1)
pred = pred.mean(1)

print(data.shape[1])
np.savetxt(
    "../data/CorrelationFunction.csv",
    np.array([x, data[start:stop, :].mean(1), data[start:stop, :].std(1, ddof=1)/np.sqrt(data.shape[1])]).T,
    delimiter=", ",
)

np.savetxt(
    "../data/parameter_covariancematrix.csv",
    pcov[0],
    delimiter=", ",
)
np.savetxt(
    "../data/covariancezmatrix.csv",
    np.cov(data[start:stop])/data.shape[1],
    delimiter=", ",
)

np.savetxt(
    "../data/CorrelationFunctionPrediction.csv",
    np.array([x, pred, dPred]).T,
    delimiter=", ",
)

np.savetxt(
    "../data/CorrelationFunction0Prediction.csv",
        np.array([x, func(x, *popt[:, 0]), func(x, *(popt[:, 0] + np.sqrt(np.diag(pcov[0])))), func(x, *(popt[:, 0] - np.sqrt(np.diag(pcov[0]))))]).T,
    delimiter=", ",
)
