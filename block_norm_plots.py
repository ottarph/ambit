import numpy as np
import matplotlib.pyplot as plt

results_base = "demos/fsi/tmp/results_fsi_channel_flag_turek_FSI2_r2000_"
results_base = "demos/fsi/tmp/results_fsi_channel_flag_turek_FSI2_r2500_"
results_base = "demos/fsi/tmp_1proc/results_fsi_channel_flag_turek_FSI2_"
results_base = "demos/fsi/tmp_tri/16proc_mqtest/results_fsi_channel_flag_turek_FSI2_r3000_"
results_post = ".txt"




time = np.concatenate(([12.0], np.loadtxt(results_base+"drag"+results_post, skiprows=1, delimiter=',')[:,0]))
newton_iters = np.loadtxt(results_base+"newton_iter_min_scaled_jacobian.txt", skiprows=1, delimiter=',')[:,0]
min_sj = np.loadtxt(results_base+"newton_iter_min_scaled_jacobian.txt", skiprows=1, delimiter=',')[:,1]
block_norms_arr = np.loadtxt(results_base+"newton_iter_blocknorms.txt", skiprows=1, delimiter=',')[:,1:]

blocks =                ((0,0),               (0,3), 
                                (1,1), (1,2), (1,3), (1,4), 
                                (2,1),               (2,4), # (2,2) is nonzero only when using stabilization for pressure
                         (3,0), (3,1),
                                (4,1),               (4,4))


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


max_T = 12.3
crop = np.flatnonzero(newton_iters_t <= max_T)
newton_iters = newton_iters[crop]
newton_iters_t = newton_iters_t[crop]
min_sj = min_sj[crop]
new_steps = np.flatnonzero(np.logical_and(newton_iters == 0, newton_iters_t <= max_T))

block_norms_arr = block_norms_arr[crop,:]

block_norms = [[None for _ in range(5)] for __ in range(5)]
for e, (i,j) in enumerate(blocks):
    block_norms[i][j] = block_norms_arr[:,e]

for e, (i,j) in enumerate(blocks):
    assert block_norms[i][j].shape == block_norms_arr[:,e].shape

A = np.zeros((5,5))
for e, (i,j) in enumerate(blocks):
    A[i,j] = np.mean(block_norms_arr[:,e])

from matplotlib import colors as mcolors
norm = mcolors.LogNorm()
im = plt.imshow(A, norm=norm)
plt.colorbar(im)
plt.savefig("mean_blocknorm_logscale.pdf")

