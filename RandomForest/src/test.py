# -*- coding: utf-8 -*-
"""
Created on Wed Aug 06 15:08:04 2014

@author: KrerKait
"""

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Testing: Flat map
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

from master import Master

master = Master() 
roots = master.load_trees(prefix='tree_5_100_devie_by_2_evaluation')

plt.hold(True)
plt.figure(1)
for t in range(3):
    plt.subplot(2, 2, t+1)
    x, y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    z = np.array([], dtype=np.float)
    for i in range(100):
        cls = []
        for j in range(100):
            cls.append(np.argmax(master.get_result(roots[t], (x[i][j], y[i][j]))))
        z = np.append(z, cls)
    z = z.reshape(100, 100)
    plt.pcolormesh(x, y, z)
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Testing: Inter map
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import matplotlib.pyplot as plt
import numpy as np

plt.hold(True)
plt.figure(1)
plt.subplot(2, 2, 4)

x, y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
z = np.array([], dtype=np.float)
for i in range(100):
    cls = []
    for j in range(100):
        cls.append(np.argmax(master.get_results(roots, (x[i][j], y[i][j]))))
    z = np.append(z, cls)
z = z.reshape(100, 100)
plt.pcolormesh(x, y, z)

plt.axis([-1, 1, -1, 1])
plt.show()