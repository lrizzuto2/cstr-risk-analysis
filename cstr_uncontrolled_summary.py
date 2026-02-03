import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

p = {
    "V": 100.0, "F": 10.0,
    "CA_in": 1.0, "T_in": 350.0,
    "k0": 7.2e10, "E": 8.314e4, "R": 8.314,
    "dH": -5.0e4, "rho": 1000.0, "Cp": 4.184,
    "UA": 2.0e4, "T_c": 300.0,
}

def cstr_odes(t, y, p):
    CA, T = y
    V, F = p["V"], p["F"]
    CA_in, T_in = p["CA_in"], p["T_in"]
    k = p["k0"] * np.exp(-p["E"] / (p["R"] * T))
    rA = k * CA

    dCA_dt = (F / V) * (CA_in - CA) - rA

    rhoCp = p["rho"] * p["Cp"]
    reaction_heat = (-p["dH"]) * rA
    heat_removal = (p["UA"] / V) * (T - p["T_c"])
    dT_dt = (F / V) * (T_in - T) + (reaction_heat - heat_removal) / rhoCp

    return [dCA_dt, dT_dt]

y0 = [1.0, 330.0]
t_span = (0.0, 20.0)
t_eval = np.linspace(*t_span, 400)

sol = solve_ivp(cstr_odes, t_span, y0, t_eval=t_eval, args=(p,), rtol=1e-7, atol=1e-9)

t = sol.t
CA = sol.y[0]
T = sol.y[1]

print(f"Final at t={t[-1]:.1f} min: CA={CA[-1]:.4f} mol/L, T={T[-1]:.2f} K")

plt.figure()
plt.plot(t, CA)
plt.xlabel("Time (min)"); plt.ylabel("C_A (mol/L)")
plt.title("CSTR: Concentration vs time")

plt.figure()
plt.plot(t, T)
plt.xlabel("Time (min)"); plt.ylabel("Temperature (K)")
plt.title("CSTR: Temperature vs time")

plt.show()
