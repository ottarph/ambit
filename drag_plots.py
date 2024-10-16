import numpy as np
import matplotlib.pyplot as plt

results_base = "demos/fsi/tmp/results_fsi_channel_flag_turek_FSI2_r2000_"
results_base = "demos/fsi/tmp/results_fsi_channel_flag_turek_FSI2_r2500_"
# results_base = "demos/fsi/tmp/results_fsi_channel_flag_turek_FSI2_r3000_"
results_post = ".txt"

TIME_CROP = False
crop_a, crob_b = 10.4, 10.5

drag_arr = np.loadtxt(results_base+"drag"+results_post, skiprows=1, delimiter=',')
drag_arr = drag_arr[np.logical_and(drag_arr[:,0] >= crop_a, drag_arr[:,0] <= crob_b),:] if TIME_CROP else drag_arr

print(f"{drag_arr[:,1].mean() = }")

plt.figure()

plt.plot(drag_arr[:,0], drag_arr[:,1], 'k-')

plt.xlabel("time")
plt.ylabel("drag")
plt.xlim(drag_arr[:,0].min(), drag_arr[:,0].max())

plt.savefig("drag.pdf")

lift_arr = np.loadtxt(results_base+"lift"+results_post, skiprows=1, delimiter=',')
lift_arr = lift_arr[np.logical_and(lift_arr[:,0] >= crop_a, lift_arr[:,0] <= crob_b),:] if TIME_CROP else lift_arr

print(f"{lift_arr[:,1].mean() = }")

plt.figure()

plt.plot(lift_arr[:,0], lift_arr[:,1], 'k-')

plt.xlabel("time")
plt.ylabel("lift")
plt.xlim(lift_arr[:,0].min(), lift_arr[:,0].max())

plt.savefig("lift.pdf")


dragcr_arr = np.loadtxt(results_base+"drag_corner"+results_post, skiprows=1, delimiter=',')
dragcr_arr = dragcr_arr[np.logical_and(dragcr_arr[:,0] >= crop_a, dragcr_arr[:,0] <= crob_b),:] if TIME_CROP else dragcr_arr

print(f"{dragcr_arr[:,1].mean() = }")

plt.figure()

plt.plot(dragcr_arr[:,0], dragcr_arr[:,1], 'k-')

plt.xlabel("time")
plt.ylabel("drag corner")
plt.xlim(dragcr_arr[:,0].min(), dragcr_arr[:,0].max())
plt.title(f"quad_deg = {results_post[2]}")
plt.savefig("dragcr_arr.pdf")


plt.figure()

plt.plot(drag_arr[:,0], (drag_arr-dragcr_arr)[:,1], 'k-')

plt.xlabel("time")
plt.ylabel("drag minus corner contribution")
plt.xlim(drag_arr[:,0].min(), drag_arr[:,0].max())

plt.savefig("drag_diff_arr.pdf")


detF_arr = np.loadtxt(results_base+"detf_corner"+results_post, skiprows=1, delimiter=',')
detF_arr = detF_arr[np.logical_and(detF_arr[:,0] >= crop_a, detF_arr[:,0] <= crob_b),:] if TIME_CROP else detF_arr


print(f"{detF_arr[:,1].mean() = }")

plt.figure()

plt.plot(detF_arr[:,0], detF_arr[:,1], 'k-')

plt.xlabel("time")
plt.ylabel("detF")
plt.xlim(detF_arr[:,0].min(), detF_arr[:,0].max())

plt.savefig("detF.pdf")

