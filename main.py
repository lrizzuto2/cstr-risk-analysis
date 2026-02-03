import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 10, 100)
y = np.exp(-t)

plt.plot(t, y)
plt.title("System response test")
plt.xlabel("Time")
plt.ylabel("Response")
plt.show()
