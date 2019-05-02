from Phase_shifters import ps

import numpy as np
import matplotlib.pyplot as plt


#
# # test
V = list(np.arange(0, 5.01, 0.01))
Theta = []

P_MZI = []
P_MRR = []

Device_MZI = ps()
Device_MRR = ps()

for v_bias in V:
    # print(v_bias)
    # Theta.append(Device_MZI.delta_theta_heater(v_bias))
    P_MZI.append(Device_MZI.MZI(v_bias, 10))
    P_MRR.append(Device_MRR.MRR(0.98, 0.98, v_bias, 10))


plt.plot(V, P_MZI)
plt.plot(V, P_MRR)
# plt.show()


print (Device_MZI.MZI(0, 10))

f = open("result.csv", 'w')
for i in list(range(len(V))):
    data = "%s\t %s\t  %s\n" %(V[i], P_MZI[i], P_MRR[i])
    print (data)
    f.write(data)
f.close()

# print (Device_MZI.MZI(10, 0))