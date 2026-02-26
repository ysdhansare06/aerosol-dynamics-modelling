# streamlit_app_two_species_first_order_depletion.py
# Run: streamlit run streamlit_app_two_species_first_order_depletion.py
#
# Two-species condensation + coagulation PSD tuner (FIRST-ORDER vapor depletion ONLY).
# - No aerosol-sink depletion (per your request).
# - Two sliders: k_depl_A and k_depl_B (1/s), with Cvap_A(t)=Cvap_A0*exp(-k_depl_A*t),
#   Cvap_B(t)=Cvap_B0*exp(-k_depl_B*t)
#
# Files required in same folder:
#   initial_psd.csv, final_psd.csv
# Columns (all files):
#   Dp (nm), dN/dlogDp (#/cc), dN (#/cc per bin)
#
# Curves:
#   initial = green, modeled = black, final = red

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from typing import Union, Tuple

# =========================
# Constants
# =========================
KB = 1.380649e-23
NA = 6.02214076e23
PI = np.pi


# =========================
# Grid helpers
# =========================
def build_edges_from_midpoints(Dp_m: np.ndarray) -> np.ndarray:
    lnD = np.log(Dp_m)
    edges = np.empty(len(Dp_m) + 1, dtype=float)
    edges[1:-1] = np.exp(0.5 * (lnD[:-1] + lnD[1:]))
    edges[0] = np.exp(lnD[0] - 0.5 * (lnD[1] - lnD[0]))
    edges[-1] = np.exp(lnD[-1] + 0.5 * (lnD[-1] - lnD[-2]))
    return edges

def dlog10_widths_from_edges(edges_m: np.ndarray) -> np.ndarray:
    return np.log10(edges_m[1:] / edges_m[:-1])

def volume_from_Dp(Dp_m: np.ndarray) -> np.ndarray:
    return (PI / 6.0) * Dp_m**3


# =========================
# Condensation physics (AAQRL-inspired, simplified)
# =========================
def mu_air_sutherland_like(T_K: float) -> float:
    return 1.716e-5 * (T_K / 273.0) ** (2.0 / 3.0)

def mean_free_path_air(T_K: float, P_Pa: float, Mg_kg_per_mol: float) -> float:
    mu = mu_air_sutherland_like(T_K)
    lam = (mu / P_Pa) * np.sqrt(PI * KB * T_K / (2.0 * (Mg_kg_per_mol / NA)))
    return lam

def fuchs_sutugin_mass_transfer_factor(Kn: np.ndarray, alpha: float) -> np.ndarray:
    a = max(float(alpha), 1e-6)
    return (1.0 + Kn) / (1.0 + (4.0 / (3.0 * a) + 0.377) * Kn + (4.0 / (3.0 * a)) * Kn**2)

def C_from_P(P_Pa: Union[float, np.ndarray], T_K: float) -> Union[float, np.ndarray]:
    return np.asarray(P_Pa) / (KB * T_K)

def kelvin_corrected_Psat(
    Psat_Pa: float,
    sigma_N_m: float,
    V1_m3_per_molecule: float,
    T_K: float,
    Dp_m: np.ndarray
) -> np.ndarray:
    expo = (4.0 * sigma_N_m * V1_m3_per_molecule) / (KB * T_K * np.maximum(Dp_m, 1e-30))
    return Psat_Pa * np.exp(expo)

def dv_dt_per_particle_species(
    Dp_m: np.ndarray,
    *,
    T_K: float,
    P_Pa: float,
    Cvap_num_m3: float,
    Psat_Pa: float,
    sigma_N_m: float,
    Mw_kg_per_mol: float,
    rho_cond_kg_m3: float,
    Dg_m2_s: float,
    alpha: float,
    Mg_air_kg_per_mol: float
) -> np.ndarray:
    """
    dv/dt (m^3/s) per particle for ONE species.
    """
    # molecular volume per molecule
    V1 = (Mw_kg_per_mol / rho_cond_kg_m3) / NA  # m^3 per molecule

    # Kelvin-corrected saturation pressure -> saturation number concentration
    Psat_eff = kelvin_corrected_Psat(Psat_Pa, sigma_N_m, V1, T_K, Dp_m)
    Csat = C_from_P(Psat_eff, T_K)  # #/m^3

    # driving force (only condense, no evaporation in this simplified model)
    driving = np.maximum(Cvap_num_m3 - Csat, 0.0)

    # transition-regime correction (Fuchs–Sutugin-type mass transfer factor)
    lam = mean_free_path_air(T_K, P_Pa, Mg_air_kg_per_mol)
    Kn = 2.0 * lam / np.maximum(Dp_m, 1e-30)
    F = fuchs_sutugin_mass_transfer_factor(Kn, alpha)

    # mass flux to one particle then to volume flux
    m_molecule = Mw_kg_per_mol / NA  # kg/molecule
    dm_dt = 2.0 * PI * Dp_m * Dg_m2_s * F * driving * m_molecule  # kg/s per particle
    dv_dt = dm_dt / rho_cond_kg_m3                                # m^3/s per particle
    return dv_dt

def remap_number_by_volume_shift(n_m3_bin: np.ndarray, v_mid: np.ndarray, dv: np.ndarray) -> np.ndarray:
    """
    Move number between bins based on volume increase per particle.
    Conservative in number.
    """
    N = len(n_m3_bin)
    ln_v = np.log(v_mid)
    n_new = np.zeros_like(n_m3_bin)

    for i in range(N):
        ni = n_m3_bin[i]
        if ni <= 0.0:
            continue

        v_new = v_mid[i] + dv[i]
        if v_new <= 0.0:
            n_new[i] += ni
            continue

        ln_v_new = np.log(v_new)
        if ln_v_new <= ln_v[0]:
            n_new[0] += ni
        elif ln_v_new >= ln_v[-1]:
            n_new[-1] += ni
        else:
            k = int(np.searchsorted(ln_v, ln_v_new) - 1)
            k = max(0, min(k, N - 2))
            w = (ln_v_new - ln_v[k]) / (ln_v[k + 1] - ln_v[k])
            n_new[k] += ni * (1.0 - w)
            n_new[k + 1] += ni * w

    return n_new


# =========================
# Coagulation physics (AAQRL-ADM beta_safe port)
# =========================
def Diff_FuchsSutugin(dp_m: float, T_K: float, P_Pa: float, Mg_kg_per_mol: float) -> float:
    mu = mu_air_sutherland_like(T_K)
    lam = (mu / P_Pa) * np.sqrt(PI * KB * T_K / (2.0 * (Mg_kg_per_mol / NA)))
    Kn = 2.0 * lam / dp_m
    num = (5.0 + 4.0 * Kn + 6.0 * Kn**2 + 18.0 * Kn**3)
    den = (5.0 - Kn + (8.0 + PI) * Kn**2)
    return (KB * T_K / (3.0 * PI * mu * dp_m)) * (num / den)

def beta_kernel_from_volumes(
    v1: float, v2: float, flag: int,
    *,
    v0: float,
    T_K: float,
    P_Pa: float,
    Mg_kg_per_mol: float,
    mw_kg_per_mol: float,
    rho_p_kg_m3: float
) -> float:
    dp0 = (6.0 / PI * v0) ** (1.0 / 3.0)
    dp1 = (6.0 / PI * v1) ** (1.0 / 3.0)
    dp2 = (6.0 / PI * v2) ** (1.0 / 3.0)

    if (dp1 < dp0) or (dp2 < dp0):
        return 0.0

    if flag == 0:
        c1 = np.sqrt(8.0 * KB * T_K / (PI * (mw_kg_per_mol / NA)))
        c2 = c1
        D1 = Diff_FuchsSutugin(dp1, T_K, P_Pa, Mg_kg_per_mol)
        D2 = Diff_FuchsSutugin(dp2, T_K, P_Pa, Mg_kg_per_mol)

        l1 = max(8.0 * D1 / (PI * c1), 1e-300)
        l2 = max(8.0 * D2 / (PI * c2), 1e-300)

        g1 = (1.0 / (3.0 * dp1 * l1)) * ((dp1 + l1) ** 3 - (dp1 * dp1 + l1 * l1) ** (3.0 / 2.0)) - dp1
        g2 = (1.0 / (3.0 * dp2 * l2)) * ((dp2 + l2) ** 3 - (dp2 * dp2 + l2 * l2) ** (3.0 / 2.0)) - dp2

        coeff1 = 2.0 * PI * (dp1 + dp2) * (D1 + D2)
        coeff2 = ((dp1 + dp2) / (dp1 + dp2 + 2.0 * np.sqrt(g1 * g1 + g2 * g2))
                  + (8.0 * (D1 + D2)) / ((dp1 + dp2) * np.sqrt(c1 * c1 + c2 * c2)))
        return coeff1 / max(coeff2, 1e-300)

    if flag == 1:
        pref = (3.0 / (4.0 * PI)) ** (1.0 / 6.0) * np.sqrt(6.0 * KB * T_K / rho_p_kg_m3)
        term1 = np.sqrt((1.0 / v1) + (1.0 / v2))
        term2 = (v1 ** (1.0 / 3.0) + v2 ** (1.0 / 3.0)) ** 2
        return pref * term1 * term2

    if flag == 2:
        return 1.0

    raise ValueError("coag_flag must be 0, 1, or 2")

def build_beta_matrix(
    v_mid: np.ndarray,
    flag: int,
    *,
    v0: float,
    T_K: float,
    P_Pa: float,
    Mg_kg_per_mol: float,
    mw_kg_per_mol: float,
    rho_p_kg_m3: float
) -> np.ndarray:
    N = len(v_mid)
    beta = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(i, N):
            bij = beta_kernel_from_volumes(
                v_mid[i], v_mid[j], flag,
                v0=v0, T_K=T_K, P_Pa=P_Pa,
                Mg_kg_per_mol=Mg_kg_per_mol,
                mw_kg_per_mol=mw_kg_per_mol,
                rho_p_kg_m3=rho_p_kg_m3,
            )
            beta[i, j] = bij
            beta[j, i] = bij
    return beta

def coagulation_step(
    n_m3_bin: np.ndarray,
    v_mid: np.ndarray,
    beta: np.ndarray,
    dt: float
) -> np.ndarray:
    """
    Conservative number update with pairwise collisions.
    Product placed by ln(volume) interpolation.
    """
    n = n_m3_bin
    N = len(n)
    ln_v = np.log(v_mid)
    dn = np.zeros_like(n)

    for i in range(N):
        if n[i] <= 0.0:
            continue
        for j in range(i, N):
            if n[j] <= 0.0:
                continue
            rate = beta[i, j] * n[i] * n[j]
            if i == j:
                rate *= 0.5
            dcoll = rate * dt

            # limiter to prevent negative bins
            if i == j:
                dcoll = min(dcoll, 0.5 * n[i])
            else:
                dcoll = min(dcoll, n[i], n[j])

            if dcoll <= 0.0:
                continue

            dn[i] -= dcoll
            dn[j] -= dcoll

            v_new = v_mid[i] + v_mid[j]
            ln_v_new = np.log(v_new)

            if ln_v_new <= ln_v[0]:
                dn[0] += dcoll
            elif ln_v_new >= ln_v[-1]:
                dn[-1] += dcoll
            else:
                k = int(np.searchsorted(ln_v, ln_v_new) - 1)
                k = max(0, min(k, N - 2))
                w = (ln_v_new - ln_v[k]) / (ln_v[k + 1] - ln_v[k])
                dn[k] += dcoll * (1.0 - w)
                dn[k + 1] += dcoll * w

    return np.maximum(n + dn, 0.0)


# =========================
# Model runner (2 species + first-order depletion)
# =========================
def run_model_two_species_first_order_depl(
    Dp_nm: np.ndarray,
    dN_cm3_bin: np.ndarray,
    *,
    time_s: float,
    dt_s: float,
    T_C: float,
    P_Pa: float,
    enable_cond: bool,
    enable_coag: bool,
    # Species A
    CvapA0: float, PsatA: float, k_depl_A_s1: float,
    # Species B
    CvapB0: float, PsatB: float, k_depl_B_s1: float,
    # Shared condensable props (surrogate)
    Mw_kg_per_mol: float,
    rho_cond_kg_m3: float,
    sigma_N_per_m: float,
    Dg_m2_s: float,
    alpha: float,
    Mg_air_kg_per_mol: float,
    # Coag
    coag_flag: int,
    mw_gas_kg_per_mol: float,
    rho_p_kg_m3: float,
) -> Tuple[np.ndarray, np.ndarray]:
    Dp_m = Dp_nm * 1e-9
    n = dN_cm3_bin.astype(float) * 1e6  # #/m^3 per bin

    edges = build_edges_from_midpoints(Dp_m)
    dlog10 = dlog10_widths_from_edges(edges)
    v_mid = volume_from_Dp(Dp_m)

    T_K = T_C + 273.15

    beta = None
    if enable_coag:
        v0 = float(v_mid.min())
        beta = build_beta_matrix(
            v_mid, coag_flag,
            v0=v0, T_K=T_K, P_Pa=P_Pa,
            Mg_kg_per_mol=Mg_air_kg_per_mol,
            mw_kg_per_mol=mw_gas_kg_per_mol,
            rho_p_kg_m3=rho_p_kg_m3,
        )

    n_steps = int(np.ceil(time_s / dt_s))
    dt = time_s / n_steps

    for step in range(n_steps):
        t_mid = (step + 0.5) * dt

        # ---- Condensation (two species with first-order vapor decay) ----
        if enable_cond:
            CvapA_t = float(CvapA0 * np.exp(-max(k_depl_A_s1, 0.0) * t_mid))
            CvapB_t = float(CvapB0 * np.exp(-max(k_depl_B_s1, 0.0) * t_mid))

            dvdt_A = dv_dt_per_particle_species(
                Dp_m,
                T_K=T_K, P_Pa=P_Pa,
                Cvap_num_m3=CvapA_t,
                Psat_Pa=PsatA,
                sigma_N_m=sigma_N_per_m,
                Mw_kg_per_mol=Mw_kg_per_mol,
                rho_cond_kg_m3=rho_cond_kg_m3,
                Dg_m2_s=Dg_m2_s,
                alpha=alpha,
                Mg_air_kg_per_mol=Mg_air_kg_per_mol,
            )
            dvdt_B = dv_dt_per_particle_species(
                Dp_m,
                T_K=T_K, P_Pa=P_Pa,
                Cvap_num_m3=CvapB_t,
                Psat_Pa=PsatB,
                sigma_N_m=sigma_N_per_m,
                Mw_kg_per_mol=Mw_kg_per_mol,
                rho_cond_kg_m3=rho_cond_kg_m3,
                Dg_m2_s=Dg_m2_s,
                alpha=alpha,
                Mg_air_kg_per_mol=Mg_air_kg_per_mol,
            )

            dv = (dvdt_A + dvdt_B) * dt
            n = remap_number_by_volume_shift(n, v_mid, dv)

        # ---- Coagulation ----
        if enable_coag and beta is not None:
            n = coagulation_step(n, v_mid, beta, dt)

    dN_out_cm3_bin = n / 1e6
    dNdlog_out = dN_out_cm3_bin / np.maximum(dlog10, 1e-30)
    return dNdlog_out, dN_out_cm3_bin


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Two-species + First-order Vapor Depletion", layout="wide")
st.title("Two-species Condensation + Coagulation (First-order Vapor Depletion)")

@st.cache_data
def load_psd(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Dp", "dN/dlogDp", "dN"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    df = df.sort_values("Dp").reset_index(drop=True)
    df["Dp"] = df["Dp"].astype(float)
    df["dN/dlogDp"] = df["dN/dlogDp"].astype(float)
    df["dN"] = df["dN"].astype(float)
    return df

try:
    df_init = load_psd("initial_psd.csv")
    df_final = load_psd("final_psd.csv")
except Exception as e:
    st.error(
        "Could not load CSVs. Put initial_psd.csv and final_psd.csv in the same folder as this app.\n\n"
        f"Error: {e}"
    )
    st.stop()

Dp_nm = df_init["Dp"].to_numpy(dtype=float)
dN_init_cm3 = df_init["dN"].to_numpy(dtype=float)

# Sidebar sliders (only relevant)
with st.sidebar:
    st.header("Controls")

    enable_cond = st.checkbox("Enable condensation", value=True)
    enable_coag = st.checkbox("Enable coagulation", value=True)

    st.subheader("Time")
    time_s = st.slider("Residence time (s)", 1.0, 400.0, 106.0, 1.0)
    dt_s = st.slider("Time step dt (s)", 0.005, 5.00, 5.00, 0.005)

    st.subheader("Environment")
    T_C = st.slider("Temperature (°C)", 0.0, 40.0, 22.0, 0.5)
    P_Pa = st.slider("Pressure (Pa)", 80000.0, 110000.0, 101325.0, 500.0)

    st.subheader("Species A (LVOC/ELVOC-like)")
    log10_CvapA = st.slider("log10(Cvap_A) [#/m³]", 5.0, 50.0, float(np.log10(1e17)), 0.05)
    log10_PsatA = st.slider("log10(Psat_A) [Pa]", -30.0, -1.0, float(np.log10(1e-8)), 0.05)
    k_depl_A = st.slider("k_depl_A (1/s)", 0.0, 5.0, 0.11, 0.01)

    st.subheader("Species B (SVOC-like)")
    log10_CvapB = st.slider("log10(Cvap_B) [#/m³]", 5.0, 50.0, float(np.log10(5e17)), 0.05)
    log10_PsatB = st.slider("log10(Psat_B) [Pa]", -30.0, -1.0, float(np.log10(1e-6)), 0.05)
    k_depl_B = st.slider("k_depl_B (1/s)", 0.0, 5.0, 0.11, 0.01)

    st.subheader("Coagulation")
    coag_flag = st.selectbox(
        "Kernel flag",
        options=[0, 1, 2],
        index=0,
        format_func=lambda x: {0: "0: Fuchs–Sutugin", 1: "1: Free-molecular", 2: "2: Constant"}[x]
    )

CvapA0 = float(10 ** log10_CvapA)
PsatA = float(10 ** log10_PsatA)
CvapB0 = float(10 ** log10_CvapB)
PsatB = float(10 ** log10_PsatB)

# Fixed surrogate properties (keep in code)
Mw_kg_per_mol = 0.15
rho_cond_kg_m3 = 1200.0
sigma_N_per_m = 0.03
Dg_m2_s = 6e-6
alpha = 1.0
Mg_air_kg_per_mol = 0.02897
mw_gas_kg_per_mol = 0.02897
rho_p_kg_m3 = 1800.0

# Run model
modeled_dNdlog, _ = run_model_two_species_first_order_depl(
    Dp_nm=Dp_nm,
    dN_cm3_bin=dN_init_cm3,
    time_s=time_s,
    dt_s=dt_s,
    T_C=T_C,
    P_Pa=P_Pa,
    enable_cond=enable_cond,
    enable_coag=enable_coag,
    CvapA0=CvapA0, PsatA=PsatA, k_depl_A_s1=k_depl_A,
    CvapB0=CvapB0, PsatB=PsatB, k_depl_B_s1=k_depl_B,
    Mw_kg_per_mol=Mw_kg_per_mol,
    rho_cond_kg_m3=rho_cond_kg_m3,
    sigma_N_per_m=sigma_N_per_m,
    Dg_m2_s=Dg_m2_s,
    alpha=alpha,
    Mg_air_kg_per_mol=Mg_air_kg_per_mol,
    coag_flag=coag_flag,
    mw_gas_kg_per_mol=mw_gas_kg_per_mol,
    rho_p_kg_m3=rho_p_kg_m3,
)

# Plot
fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(
    df_init["Dp"].to_numpy(dtype=float),
    df_init["dN/dlogDp"].to_numpy(dtype=float),
    marker="o", markersize=2.5, linewidth=1.2,
    label="Initial (measured)",
    color="green",
)

ax.plot(
    Dp_nm,
    modeled_dNdlog,
    marker="o", markersize=2.2, linewidth=1.2,
    label="Modeled",
    color="black",
)

ax.plot(
    df_final["Dp"].to_numpy(dtype=float),
    df_final["dN/dlogDp"].to_numpy(dtype=float),
    marker="o", markersize=2.5, linewidth=1.2,
    label="Final (measured)",
    color="red",
)

ax.set_xscale("log")
ax.set_xlabel(r"Particle Diameter, $D_p$ (nm)")
ax.set_ylabel(r"$dN/d\log D_p$ (#/cc)")
ax.grid(True, which="both", linestyle="--", linewidth=0.5)
ax.legend()

st.pyplot(fig, clear_figure=True)

# Parameter readout
st.markdown("### Current parameters")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Residence time (s)", f"{time_s:.1f}")
c2.metric("Cvap_A0 (#/m³)", f"{CvapA0:.2e}")
c3.metric("Psat_A (Pa)", f"{PsatA:.2e}")
c4.metric("k_depl_A (1/s)", f"{k_depl_A:.3f}")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Cvap_B0 (#/m³)", f"{CvapB0:.2e}")
c6.metric("Psat_B (Pa)", f"{PsatB:.2e}")
c7.metric("k_depl_B (1/s)", f"{k_depl_B:.3f}")
c8.metric("Coag kernel", {0: "F-S", 1: "FM", 2: "Const"}[coag_flag])

st.caption(
    "Notes: Vapor depletion here is purely first-order in time (Cvap_A(t)=Cvap_A0*exp(-kA*t), "
    "Cvap_B(t)=Cvap_B0*exp(-kB*t)). No aerosol-sink depletion is included."
)
