from Phase_shifters import ps
import numpy as np


#
# # test
V = list(np.arange(0, 5.01, 0.01))
Theta = []

P_MZI = []
P_MRR = []

Device_MZI = ps()
Device_MRR = ps()

for v_bias in V:
    # print(v)
    # Theta.append(Device_MZI.delta_theta_heater(v_bias))
    P_MZI.append(Device_MZI.MZI(v_bias))
    P_MRR.append(Device_MRR.MRR(0.98, 0.98, v_bias))

plt.plot(V,P_MZI)
plt.plot(V,P_MRR)