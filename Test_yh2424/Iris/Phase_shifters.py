####
# Output power through Phase-shifter

import numpy as np
import matplotlib.pyplot as plt


class ps:
    def __init__(self):
        self.output_power_bar = 0
        self.output_power_cross = 0
        self.output_power = 0
        self.gamma = 2 * np.pi / (7 ** 2)

    def delta_theta_heater(self, v_bias):
        delta_theta = self.gamma * v_bias ** 2
        return delta_theta

    def MZI(self, v_bias, Pmax):
        delta_theta = self.delta_theta_heater(v_bias)
        output_power_cross = Pmax * np.cos(delta_theta / 2) ** 2
        # print ('Here')
        return output_power_cross

    def MRR(self, r, a, v_bias, Pmax):
        delta_theta = self.delta_theta_heater(v_bias)
        Ipass = a**2 - 2*r*a*np.cos(delta_theta) + r**2
        Iinput = 1 - 2*a*r*np.cos(delta_theta) + (r*a)**2
        output_power = Pmax*(1 - Ipass/Iinput)
        # print('Here2')
        return output_power


#
# # test
# V = list(np.arange(0, 5.01, 0.01))
# Theta = []
#
# P_MZI = []
# P_MRR = []
#
# Device_MZI = ps()
# Device_MRR = ps()
#
# for v_bias in V:
#     # print(v)
#     # Theta.append(Device_MZI.delta_theta_heater(v_bias))
#     P_MZI.append(Device_MZI.MZI(v_bias, 10))
#     P_MRR.append(Device_MRR.MRR(0.98, 0.98, v_bias, 10))
#
# plt.plot(V,P_MZI)
# plt.plot(V,P_MRR)
plt.show()