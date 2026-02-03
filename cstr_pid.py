import numpy as np
import matplotlib.pyplot as plt

p = {
    "V": 100.0,
    "F": 10.0,
    "CA_in": 1.0,
    "T_in": 350.0,

    "k0": 7.2e10,
    "E": 8.314e4,
    "R": 8.314,

    "dH": -5.0e4,
    "rho": 1000.0,
    "Cp": 4.184,

    "UA": 2.0e4,

    "T_set": 333.0,
    "Tc_min": 280.0,
    "Tc_max": 330.0,
}

Kp = 4.0
Ki = 0.40
Kd = 0.0

def reaction_rate(CA, T, p):
    k = p["k0"] * np.exp(-p["E"] / (p["R"] * T))
    return k * CA

def plant_derivatives(CA, T, Tc, p):
    V, F = p["V"], p["F"]
    rhoCp = p["rho"] * p["Cp"]

    rA = reaction_rate(CA, T, p)

    dCA_dt = (F / V) * (p["CA_in"] - CA) - rA
    reaction_heat = (-p["dH"]) * rA
    heat_removal = (p["UA"] / V) * (T - Tc)
    dT_dt = (F / V) * (p["T_in"] - T) + (reaction_heat - heat_removal) / rhoCp

    return dCA_dt, dT_dt

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

t0, tf = 0.0, 60.0
dt = 0.01
N = int((tf - t0)/dt) + 1

t = np.linspace(t0, tf, N)
CA = np.zeros(N)
T  = np.zeros(N)
Tc = np.zeros(N)
e  = np.zeros(N)

CA[0] = 1.0
T[0]  = 330.0

Tc_bias = 300.0
Tc[0] = Tc_bias

integral_e = 0.0
prev_e = p["T_set"] - T[0]

for i in range(N - 1):
    e[i] = p["T_set"] - T[i]
    deriv_e = (e[i] - prev_e) / dt

    Tc_cmd = Tc_bias + (Kp * e[i] + Ki * integral_e + Kd * deriv_e)
    Tc_cmd = clamp(Tc_cmd, p["Tc_min"], p["Tc_max"])

    if p["Tc_min"] < Tc_cmd < p["Tc_max"]:
        integral_e += e[i] * dt

    Tc[i] = Tc_cmd

    dCA_dt, dT_dt = plant_derivatives(CA[i], T[i], Tc[i], p)
    CA[i+1] = CA[i] + dCA_dt * dt
    T[i+1]  = T[i]  + dT_dt  * dt

    prev_e = e[i]

# Fix the last points so arrays don't end with zeros
e[-1] = p["T_set"] - T[-1]
Tc[-1] = Tc[-2]

print(f"Final: CA={CA[-1]:.4f}, T={T[-1]:.2f}, Tc={Tc[-1]:.2f}")

plt.figure()
plt.plot(t, T, label="Reactor T")
plt.axhline(p["T_set"], linestyle="--", label="T_set")
plt.legend()
plt.title("CSTR Temperature Control (PID)")
plt.xlabel("Time (min)")
plt.ylabel("Temperature (K)")

plt.figure()
plt.plot(t, Tc)
plt.title("Control Action (Tc)")
plt.xlabel("Time (min)")
plt.ylabel("Coolant Temperature (K)")

plt.figure()
plt.plot(t, e)
plt.title("Tracking Error")
plt.xlabel("Time (min)")
plt.ylabel("Error (K)")

plt.show()
