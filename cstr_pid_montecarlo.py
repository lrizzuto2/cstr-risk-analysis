import numpy as np
import matplotlib.pyplot as plt

# --- Base parameters (same plant) ---
p = {
    "V": 100.0,
    "F": 10.0,
    "CA_in_mean": 1.0,
    "T_in_mean": 350.0,

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

# --- Controller gains (from your working run) ---
Kp = 4.0
Ki = 0.40
Kd = 0.0
Tc_bias = 300.0

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def reaction_rate(CA, T, p):
    k = p["k0"] * np.exp(-p["E"] / (p["R"] * T))
    return k * CA

def plant_step(CA, T, Tc, CA_in, T_in, dt, p):
    V, F = p["V"], p["F"]
    rhoCp = p["rho"] * p["Cp"]

    rA = reaction_rate(CA, T, p)

    dCA_dt = (F / V) * (CA_in - CA) - rA
    reaction_heat = (-p["dH"]) * rA
    heat_removal = (p["UA"] / V) * (T - Tc)
    dT_dt = (F / V) * (T_in - T) + (reaction_heat - heat_removal) / rhoCp

    return CA + dCA_dt * dt, T + dT_dt * dt

def run_one(seed, sigma_T=3.0, sigma_CA=0.02):
    rng = np.random.default_rng(seed)

    t0, tf = 0.0, 60.0
    dt = 0.01
    N = int((tf - t0)/dt) + 1
    t = np.linspace(t0, tf, N)

    CA = np.zeros(N)
    T  = np.zeros(N)
    Tc = np.zeros(N)

    CA[0] = 1.0
    T[0]  = 330.0
    Tc[0] = Tc_bias

    integral_e = 0.0
    prev_e = p["T_set"] - T[0]

    Tmax = T[0]

    for i in range(N - 1):
        # stochastic disturbances (CHE but quant-style noise)
        CA_in = p["CA_in_mean"] + rng.normal(0.0, sigma_CA)
        T_in  = p["T_in_mean"]  + rng.normal(0.0, sigma_T)

        e = p["T_set"] - T[i]
        deriv_e = (e - prev_e) / dt

        Tc_cmd = Tc_bias + (Kp*e + Ki*integral_e + Kd*deriv_e)
        Tc_cmd = clamp(Tc_cmd, p["Tc_min"], p["Tc_max"])

        # integrate only if not saturated
        if p["Tc_min"] < Tc_cmd < p["Tc_max"]:
            integral_e += e * dt

        Tc[i] = Tc_cmd
        CA[i+1], T[i+1] = plant_step(CA[i], T[i], Tc[i], CA_in, T_in, dt, p)

        if T[i+1] > Tmax:
            Tmax = T[i+1]

        prev_e = e

    Tc[-1] = Tc[-2]
    return t, T, Tc, Tmax, T[-1]

# --- Monte Carlo ---
M = 300  # number of scenarios (increase later)
Tmax_list = []
Tfinal_list = []
sample_paths = []

for m in range(M):
    t, T, Tc, Tmax, Tfinal = run_one(seed=1000+m)
    Tmax_list.append(Tmax)
    Tfinal_list.append(Tfinal)
    if m < 20:  # store a few paths for a fan chart
        sample_paths.append(T)

Tmax_arr = np.array(Tmax_list)
Tfinal_arr = np.array(Tfinal_list)

# Risk metrics (quant-style)
limit = 334.5
prob_exceed = np.mean(Tmax_arr > limit)
mean_final_err = np.mean(p["T_set"] - Tfinal_arr)

print(f"Monte Carlo scenarios: {M}")
print(f"P(max T > {limit} K) = {prob_exceed:.3f}")
print(f"Mean final error (T_set - T_final) = {mean_final_err:.3f} K")
print(f"Max of Tmax across scenarios = {Tmax_arr.max():.2f} K")

# Fan chart
plt.figure()
for path in sample_paths:
    plt.plot(t, path, alpha=0.25)
plt.axhline(p["T_set"], linestyle="--", label="T_set")
plt.axhline(limit, linestyle="--", label="Risk limit 334.5K")
plt.title("Monte Carlo Temperature Paths (sample)")
plt.xlabel("Time (min)")
plt.ylabel("Reactor Temperature (K)")
plt.legend()
plt.show()
