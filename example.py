from potentialflow import geometry2D
from potentialflow.vortexpanelmethod import VPM
from potentialflow.potentialflow2D import JoukowskiAirfoil, plot

import matplotlib.pyplot as plt

# jkwski = JoukowskiAirfoil(cl_d=.8, thick_d=.14)
# cl, cm4 = jkwski.performance(alpha=5.)
# 
# plot(jkwski)
# # TODO: Cm4 is off for Joukowski Airfoil
# print("Joukoski Airfoil")
# print("CL: ", cl)
# print("Cm4: ", cm4)


# naca2412 = geometry2D.naca4digit('2412', 200)
ul04 = geometry2D.uniformload(CL=.45, thick=.04, n_points=200,
                              thickness_type='open_te')

plt.plot(ul04[:, 0], ul04[:, 1])
plt.scatter(ul04[0, 0], ul04[0, 1])
plt.show()

# vpm_naca = VPM(naca2412)
# results = vpm_naca.solve([0.])
vpm_ul04 = VPM(ul04)
results = vpm_ul04.solve([4.])

print(results)
