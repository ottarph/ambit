import numpy as np
import matplotlib.pyplot as plt

results_base = "demos/fsi/tmp/results_fsi_channel_flag_turek_FSI2_r2000_"
results_base = "demos/fsi/tmp/results_fsi_channel_flag_turek_FSI2_r2500_"
results_base = "demos/fsi/tmp_1proc/results_fsi_channel_flag_turek_FSI2_"
results_base = "demos/fsi/tmp_tri/16proc/results_fsi_channel_flag_turek_FSI2_"
results_base = "demos/fsi/tmp_tri/16proc/results_fsi_channel_flag_turek_FSI2_r3500_"
results_post = ".txt"

ADD_REF = True
TIME_CROP = False
REFERENCE_SLICE = 1
if ADD_REF:
    DON_STYLE = "r:"
    REF_STYLE = "k-"
else:
    DON_STYLE = "k-"
crop_a, crob_b = 12.0, 14.0


time = np.loadtxt(results_base+"drag"+results_post, skiprows=1, delimiter=',')[:,0]
drag = np.loadtxt(results_base+"drag"+results_post, skiprows=1, delimiter=',')[:,1]
lift = np.loadtxt(results_base+"lift"+results_post, skiprows=1, delimiter=',')[:,1] * -1
dragcr = np.loadtxt(results_base+"drag_corner"+results_post, skiprows=1, delimiter=',')[:,1]
minimumJ = np.loadtxt(results_base+"minimumDetF"+results_post, skiprows=1, delimiter=',')[:,1]
minimizerJ_val = np.loadtxt(results_base+"minimizerDetF"+results_post, skiprows=1, delimiter=',')[:,1]
minimizerJ_x = np.loadtxt(results_base+"minimizerDetF"+results_post, skiprows=1, delimiter=',')[:,2]
minimizerJ_y = np.loadtxt(results_base+"minimizerDetF"+results_post, skiprows=1, delimiter=',')[:,3]

if TIME_CROP:
    eps = 1e-6
    inds = np.logical_and(time >= crop_a - eps, time <= crob_b + eps)

    time = time[inds]
    drag = drag[inds]
    lift = lift[inds]
    dragcr = dragcr[inds]


if ADD_REF:
    ref = np.loadtxt("fsi2_reference.txt")[::REFERENCE_SLICE,:] * 1.0

    if TIME_CROP:
        ref_inds = np.logical_and(ref[:,0] >= crop_a - eps, ref[:,0] <= crob_b + eps)
        time_r = ref[ref_inds,0]
        drag_r = ref[ref_inds,4] + ref[ref_inds,6]
        lift_r = ref[ref_inds,5] + ref[ref_inds,7]
        xdisp_r = ref[ref_inds,10]
        ydisp_r = ref[ref_inds,11]

    else:
        time_r = ref[:,0]
        drag_r = ref[:,4] + ref[:,6]
        lift_r = ref[:,5] + ref[:,7]
        xdisp_r = ref[:,10]
        ydisp_r = ref[:,11]



print(f"{drag.mean() = }")

plt.figure()

plt.plot(time_r, drag_r, REF_STYLE, label="reference") if ADD_REF else None
plt.plot(time, drag, DON_STYLE, label="deeponet")

plt.xlabel("time")
plt.ylabel("drag")
plt.xlim(time.min(), time.max())
plt.legend() if ADD_REF else None

plt.savefig("drag.pdf")


print(f"{lift.mean() = }")

plt.figure()

plt.plot(time_r, lift_r, REF_STYLE, label="reference") if ADD_REF else None
plt.plot(time, lift, DON_STYLE, label="deeponet")

plt.xlabel("time")
plt.ylabel("lift")
plt.xlim(time.min(), time.max())
plt.legend() if ADD_REF else None

plt.savefig("lift.pdf")


print(f"{dragcr.mean() = }")

plt.figure()

plt.plot(time, dragcr, 'k-')

plt.xlabel("time")
plt.ylabel("drag corner")
plt.xlim(time.min(), time.max())
plt.title(f"quad_deg = {results_post[2]}" if len(results_post) > 4 else None)
plt.savefig("dragcr.pdf")


plt.figure()

plt.plot(time, minimizerJ_val, 'k-')

plt.xlabel("time")
plt.ylabel(r"$\mathrm{min}_x \; \mathrm{det} \; F(u(t,x))$")
plt.xlim(time.min(), time.max())

plt.savefig("minimumJ.pdf")

min_ind = np.argmin(minimizerJ_val)
min_x = minimizerJ_x[min_ind]
min_y = minimizerJ_y[min_ind]
print(f"{min_x = }")
print(f"{min_y = }")