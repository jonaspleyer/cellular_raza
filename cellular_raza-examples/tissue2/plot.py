import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

v1 = np.array([-8.6522, 11.7445])
v2 = np.array([0.2234, 5.5635])

fig, ax = plt.subplots(figsize=(8, 8))

ax.set_xlim(-12, 12)
ax.set_ylim(-12, 12)

ax.plot([v1[0], v2[0]], [v1[1], v2[1]], color="k")
ax.add_patch(Circle((0, 0), 4.8))

plt.show()
