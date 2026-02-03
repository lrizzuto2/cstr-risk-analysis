import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -----------------------------
# CSTR with exothermic reaction: A -> products
# States: y = [CA, T]
# -----------------------------

# Parameters (reasonable demo values; we can refine later)
p = {
    "V": 100.0,          # reactor volume (L)
    "F": 10.0,           # volumetric flow rate (L/min)
    "CA_in": 1.0,        # feed concentration of A (mol/L)
    "T_in": 350.0,       # feed temperature (K)

    "k0": 7.2e10,        # pre-exponential factor (1/min)
    "E": 8.314e4,        # activation energy (J/mol)
    "R": 8.314,          # gas constant (J/mol/K)

    "dH": -5.0e4,        # heat of reaction (J/mol) (negative = exothermic)
    "rho": 1000.0,       # density (g/L) approx water (1000 g/L)
    "Cp": 4.184,         # heat capacity (J/g/K) approx water

    "UA": 2.0e4,         # heat transfer coefficient*area (J/min/K)
    "T_c": 300.0,        # coolant temperature (K)
}

def cstr_odes(t, y, p):
    CA, T = y

    V, F = p["V"], p["F"]
    CA_in, T_in = p["CA_in"], p["T_in"]

    k0, E, R = p["k0"], p["E"], p["R"]
    dH, rho, Cp = p["dH"], p["rho"], p["Cp"]
    UA, T_c = p["UA"], p["T_c"]

    # Arrhenius rate constant and reaction rate
    k = k0 * np.exp(-E / (R * T))
    rA = k * CA  # first-order in A (mol/L/min)

    # Mass balance
    dCA_dt = (F / V) * (CA_in - CA) - rA

    # Energy balance
    # Convert to consistent units:
    # rho (g/L), Cp (J/g/K) => rho*Cp (J/L/K)
    rhoCp = rho * Cp
    reaction_heat = (-dH) * rA          # J/L/min (since -dH positive for exo)
    heat_removal = (UA / V) * (T - T_c) # J/L/min

    dT_dt = (F / V) * (T_in - T) + (reaction_heat - heat_removal) / rhoCp

    return [dCA_dt, dT_dt]

# Initial conditions
CA0 = 1.0   # mol/L
T0 = 330.0  # K
y0 = [CA0, T0]

# Time span (minutes)
t_span = (0.0, 20.0)
t_eval = np.linspace(t_span[0], t_span[1], 400)

sol = solve_ivp(cstr_odes, t_span, y0, t_eval=t_eval, args=(p,), rtol=1e-7, atol=1e-9)

if not sol.success:
    raise RuntimeError("ODE solver failed: " + str(sol.message))

t = sol.t
CA = sol.y[0]
T = sol.y[1]

# Plot results
plt.figure()
plt.plot(t, CA)
plt.xlabel("Time (min)")
plt.ylabel("C_A (mol/L)")
plt.title("CSTR: Concentration vs time")

plt.figure()
plt.plot(t, T)
plt.xlabel("Time (min)")
plt.ylabel("Temperature (K)")
plt.title("CSTR: Temperature vs time")

plt.show()
