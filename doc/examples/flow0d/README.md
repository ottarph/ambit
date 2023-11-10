# README #

This example demonstrates how to simulate a cardiac cycle using a lumped-parameter (0D) model for the heart chambers and the entire circulation. Multiple heart beats are run
until a periodic state criterion is met (which compares variable values at the beginning to those at the end of a cycle, and stops if the relative change is less than
a specified value, here 'eps_periodic' in the TIME_PARAMS dictionary). The problem is set up such that periodicity is reached after 5 heart cycles.

### Instructions ###
Run the simulation, either in one of the provided Docker containers or using your own FEniCSx/Ambit installation, using the command
```
python3 flow0d_heart_cycle.py
```

For postprocessing of the time courses of pressures, volumes, and fluxes of the 0D model, make sure to have Gnuplot (and TeX) installed.
Navigate to the output folder (tmp/) and execute the script flow0d_plot.py (which lies in ambit/src/ambit_fe/postprocess/):
```
flow0d_plot.py -s flow0d_heart_cycle -n 100
```
A folder 'plot_flow0d_heart_cycle' is created inside tmp/. Look at the results of pressures (p), volumes (V), and fluxes (q,Q) over time.
Subscripts v, at, ar, ven refer to 'ventricular', 'atrial', 'arterial', and 'venous', respectively. Superscripts l, r, sys, pul refer to 'left', 'right', 'systemic', and
'pulmonary', respectively.
Try to understand the time courses of the respective pressures, as well as the plots of ventricular pressure over volume.
Check that the overall system volume is constant and around 4-5 liters.

### Solution ###

The solution is depicted in the following figure, showing the time course of volumes and pressures of the circulatory system.

![A. Left heart and systemic pressures over time. B. Right heart and pulmonary pressures over time. C. Left and right ventricular and atrial volumes over time. D. Left and right ventricular pressure-volume relationships of periodic (5th) cycle.](https://github.com/marchirschvogel/ambit/assets/52761273/9df6b4a8-2acc-453d-b79b-fc797abcb2c9)
**A. Left heart and systemic pressures over time. B. Right heart and pulmonary pressures over time. C. Left and right ventricular and atrial volumes over time. D. Left and right ventricular pressure-volume relationships of periodic (5th) cycle.**