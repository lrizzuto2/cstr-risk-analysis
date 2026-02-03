import numpy as np
import matplotlib.pyplot as plt
import csv

# -----------------------
# Plant + controller setup
# -----------------------
p = {
    "V": 100.0,
    "F": 10.0,
    "CA_in_mean": 1.0,
    "T_in_mean": 350.0,

    "k0": 7.2e10,
    "E": 8.314e4,
    "R": 8.314,

    # "Moderately risky" regime (good S-curve)
    "dH": -7.0e4,     # J/mol (more exothermic than base)
    "rho": 1000.0,    # g/L
    "Cp": 4.184,      # J/g/K
    "UA": 1.2e4,      # J/min/K (weaker heat removal than base)

    "T_set": 333.0,
    "Tc_min": 280.0,
    "Tc_max": 330.0,
}

# Controller gains (stable)
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

def run_one(seed, sigma_T, sigma_CA=0.02, tf=60.0):
    rng = np.random.default_rng(seed)

    dt = 0.01
    N = int(tf / dt) + 1

    CA = 1.0
    T = 330.0

    integral_e = 0.0
    prev_e = p["T_set"] - T

    Tmax = T

    for _ in range(N - 1):
        # stochastic disturbances
        CA_in = p["CA_in_mean"] + rng.normal(0.0, sigma_CA)
        T_in  = p["T_in_mean"]  + rng.normal(0.0, sigma_T)

        # PID on temperature
        e = p["T_set"] - T
        deriv_e = (e - prev_e) / dt

        Tc_cmd = Tc_bias + (Kp * e + Ki * integral_e + Kd * deriv_e)
        Tc_cmd = clamp(Tc_cmd, p["Tc_min"], p["Tc_max"])

        # anti-windup
        if p["Tc_min"] < Tc_cmd < p["Tc_max"]:
            integral_e += e * dt

        CA, T = plant_step(CA, T, Tc_cmd, CA_in, T_in, dt, p)

        Tmax = max(Tmax, T)
        prev_e = e

    return Tmax

def estimate_risk_metrics(sigma_T, M, limit):
    Tmax = np.array([run_one(1000 + i, sigma_T=sigma_T) for i in range(M)])

    p_exceed = float(np.mean(Tmax > limit))
    tmax = float(Tmax.max())
    var95 = float(np.percentile(Tmax, 95))
    cvar95 = float(Tmax[Tmax >= var95].mean())
    return p_exceed, tmax, var95, cvar95

# -----------------------
# Risk curve settings
# -----------------------
sigmas = [0.5, 1, 2, 3, 4, 5, 6, 8, 10]
limit = 336.0
M = 300

probs = []
maxes = []
vars95 = []
cvars95 = []

print(f"Monte Carlo per sigma: M={M}, limit={limit} K\n")
print("sigma_T(K)\tP(maxT>limit)\tmax(Tmax)\tVaR95\tCVaR95")

for s in sigmas:
    p_exceed, tmax, var95, cvar95 = estimate_risk_metrics(sigma_T=s, M=M, limit=limit)

    probs.append(p_exceed)
    maxes.append(tmax)
    vars95.append(var95)
    cvars95.append(cvar95)

    print(f"{s:>7.2f}\t\t{p_exceed:>10.3f}\t\t{tmax:>7.2f}\t{var95:>7.2f}\t{cvar95:>7.2f}")

# -----------------------
# Save table to CSV
# -----------------------
with open("risk_table.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["sigma_T", "P_exceed", "max_Tmax", "VaR95", "CVaR95"])
    for s, pex, tmx, v, cv in zip(sigmas, probs, maxes, vars95, cvars95):
        w.writerow([s, pex, tmx, v, cv])

# -----------------------
# Plot + save PNG
# -----------------------
plt.figure()
plt.plot(sigmas, probs, marker="o")
plt.xlabel("sigma_T (K)  [feed temperature volatility]")
plt.ylabel(f"P(max T > {limit} K)")
plt.title("Risk Curve: Temperature Excursion Probability vs Volatility")
plt.ylim(-0.02, 1.02)
plt.tight_layout()
plt.savefig("risk_curve.png", dpi=200)
plt.show()
