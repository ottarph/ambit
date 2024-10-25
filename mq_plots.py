import numpy as np
import matplotlib.pyplot as plt

results_base = "demos/fsi/tmp/results_fsi_channel_flag_turek_FSI2_r2000_"
results_base = "demos/fsi/tmp/results_fsi_channel_flag_turek_FSI2_r2500_"
results_base = "demos/fsi/tmp_1proc/results_fsi_channel_flag_turek_FSI2_"
results_base = "demos/fsi/tmp_tri/16proc_mqtest/results_fsi_channel_flag_turek_FSI2_r3000_"
results_post = ".txt"

ADD_REF = False
TIME_CROP = True
REFERENCE_SLICE = 1
if ADD_REF:
    DON_STYLE = "r:"
    REF_STYLE = "k-"
else:
    DON_STYLE = "k-"
crop_a, crob_b = 10.0, 11.0



time = np.concatenate(([12.0], np.loadtxt(results_base+"drag"+results_post, skiprows=1, delimiter=',')[:,0]))
newton_iters = np.loadtxt(results_base+"newton_iter_min_scaled_jacobian.txt", skiprows=1, delimiter=',')[:,0]
min_sj = np.loadtxt(results_base+"newton_iter_min_scaled_jacobian.txt", skiprows=1, delimiter=',')[:,1]


newton_iters_t = np.zeros_like(newton_iters)
new_steps = np.flatnonzero(newton_iters == 0)


for i in range(new_steps.shape[0]-1):
    k0, k1 = new_steps[i], new_steps[i+1]
    t0, t1 = time[i], time[i+1]
    newton_iters_t[k0:k1] = t0 + (t1-t0) * newton_iters[k0:k1] / (k1-k0)
k0 = k1 * 1
k1 = newton_iters.shape[0]
t0 = time[i]
t1 = time[-1]
newton_iters_t[k0:k1] = t0 + (t1-t0) * newton_iters[k0:k1] / (k1-k0)


plt.figure()

plt.plot(np.arange(newton_iters.shape[0]), min_sj, "k-", label="min scaled Jacobian")

plt.xlabel("Nonlinear iterations")
plt.ylabel("minimum scaled Jacobian")
plt.xlim(0, newton_iters.shape[0])

plt.savefig("newton_mq.pdf")


plt.figure()

plt.plot(newton_iters_t, min_sj, "k-", label="min scaled Jacobian")

plt.xlabel("time (newton iterations fractional)")
plt.ylabel("minimum scaled Jacobian")
plt.xlim(newton_iters_t.min(), newton_iters_t.max())

plt.savefig("newton_mq_t.pdf")

