import streamlit as st
import numpy as np
import pandas as pd
from pso import PSO
from gwo import GWO

st.set_page_config(page_title="Optimasi Fungsi Sphere", layout="wide")
st.title("Optimasi Fungsi Sphere dengan PSO dan GWO")
st.markdown("""
Aplikasi ini mendemonstrasikan proses optimasi fungsi Sphere 2-dimensi menggunakan **Particle Swarm Optimization (PSO)** dan **Grey Wolf Optimizer (GWO)**. 
""")

st.header("Definisi Masalah: Fungsi Sphere")
st.latex(r"f(x) = \sum_{i=1}^{n} x_i^2 = x_1^2 + x_2^2")
st.markdown(r"""
- **Dimensi (n):** 2
- **Rentang Pencarian:** $x_1, x_2 \in [-5.12, 5.12]$
- **Global Optimum:** $f(x) = 0$ pada $x = (0, 0)$
""")

N_PARTICLES = 5
N_DIMENSIONS = 2
SEARCH_RANGE = (-5.12, 5.12)
MAX_ITER = 5

INITIAL_POSITIONS = [
    [1.0, -2.0],
    [-3.0, 1.5],
    [0.5, 0.5],
    [4.0, -1.0],
    [-1.0, 3.0]
]

st.sidebar.title("Pengaturan Simulasi")
algo_choice = st.sidebar.selectbox("Pilih Algoritma Optimasi:", ["PSO", "GWO"])


# LOGIKA UNTUK PSO
if algo_choice == "PSO":
    st.header("Particle Swarm Optimization (PSO)")
    W = 0.7
    C1 = 2.0
    C2 = 2.0
    INITIAL_VELOCITIES = [[0.0, 0.0]] * N_PARTICLES
    FIXED_R_PSO = [(0.7, 0.4), (0.2, 0.9), (0.8, 0.1), (0.3, 0.6), (0.9, 0.5)]
    st.subheader("Parameter Inisialisasi PSO")
    param_data_pso = {
        "Parameter": ["Jumlah Partikel (N)", "Bobot Inersia (w)", "Koefisien Kognitif (c1)", "Koefisien Sosial (c2)", "Jumlah Iterasi (T)"],
        "Nilai": [N_PARTICLES, W, C1, C2, MAX_ITER]
    }
    st.table(pd.DataFrame(param_data_pso))
    st.subheader("Posisi & Kecepatan Awal Partikel")
    initial_df_pso = pd.DataFrame(INITIAL_POSITIONS, columns=["x1", "x2"])
    initial_df_pso["v1"] = [v[0] for v in INITIAL_VELOCITIES]
    initial_df_pso["v2"] = [v[1] for v in INITIAL_VELOCITIES]
    initial_df_pso.index = [f"Partikel {i+1}" for i in range(N_PARTICLES)]
    st.dataframe(initial_df_pso)

    if st.button("Jalankan Simulasi PSO"):
        pso = PSO(n_particles=N_PARTICLES, n_dimensions=N_DIMENSIONS, search_range=SEARCH_RANGE, w=W, c1=C1, c2=C2)
        pso.initialize_population_manual(INITIAL_POSITIONS, INITIAL_VELOCITIES)
        gbest_pos, gbest_val, history = pso.optimize(max_iter=MAX_ITER, fixed_r_values=FIXED_R_PSO)
        st.subheader("Hasil Perhitungan Langkah demi Langkah")
        convergence_data = []

        for log in history:
            st.markdown(f"---")
            st.markdown(f"### Iterasi {log['iteration']}")
            st.markdown(f"**Nilai Acak:** `r1 = {log['r1']}`, `r2 = {log['r2']}`")
            st.markdown(f"**Global Best Sebelum Update:** Posisi = `{np.round(log['gbest_position_before'], 4)}`, Fitness = `{log['gbest_value_before']:.4f}`")
            iter_df = pd.DataFrame(log['positions_before'], columns=["Posisi Awal x1", "Posisi Awal x2"])
            iter_df["Kecepatan Awal v1"] = [v[0] for v in log['velocities_before']]
            iter_df["Kecepatan Awal v2"] = [v[1] for v in log['velocities_before']]
            iter_df["Kecepatan Baru v1"] = [v[0] for v in log['velocities_after']]
            iter_df["Kecepatan Baru v2"] = [v[1] for v in log['velocities_after']]
            iter_df["Posisi Baru x1"] = [p[0] for p in log['positions_after']]
            iter_df["Posisi Baru x2"] = [p[1] for p in log['positions_after']]
            iter_df["Fitness Baru"] = [pso.sphere_function(p) for p in log['positions_after']]
            iter_df["PBest Fitness"] = log['pbest_values_after']
            iter_df.index = [f"Partikel {i+1}" for i in range(N_PARTICLES)]
            st.dataframe(iter_df.style.format("{:.4f}"))
            st.markdown(f"**Global Best Setelah Update:** Posisi = `{np.round(log['gbest_position_after'], 4)}`, Fitness = `{log['gbest_value_after']:.4f}`")
            convergence_data.append({'Iterasi': log['iteration'], 'Best Fitness': log['gbest_value_after']})

        st.subheader("Hasil Akhir Optimasi PSO")
        st.success(f"Posisi Terbaik (gbest): {np.round(gbest_pos, 6)}")
        st.success(f"Nilai Fitness Terbaik: {gbest_val:.6f}")
        st.subheader("Grafik Konvergensi PSO")
        convergence_df = pd.DataFrame(convergence_data).set_index('Iterasi')
        st.line_chart(convergence_df)




# LOGIKA UNTUK GWO

elif algo_choice == "GWO":
    st.header("Grey Wolf Optimizer (GWO)")
    FIXED_R_GWO = [(0.6, 0.3), (0.1, 0.8), (0.7, 0.2), (0.4, 0.9), (0.5, 0.1)]
    st.subheader("Parameter Inisialisasi GWO")
    param_data_gwo = {
        "Parameter": ["Jumlah Serigala (N)", "Jumlah Iterasi (T)"],
        "Nilai": [N_PARTICLES, MAX_ITER]
    }
    st.table(pd.DataFrame(param_data_gwo))
    st.subheader("Posisi Awal Serigala")
    initial_df_gwo = pd.DataFrame(INITIAL_POSITIONS, columns=["x1", "x2"])
    initial_df_gwo.index = [f"Serigala {i+1}" for i in range(N_PARTICLES)]
    st.dataframe(initial_df_gwo)

    if st.button("Jalankan Simulasi GWO"):
        gwo = GWO(n_wolves=N_PARTICLES, n_dimensions=N_DIMENSIONS, search_range=SEARCH_RANGE)
        gwo.initialize_population_manual(INITIAL_POSITIONS)
        alpha_pos, alpha_score, history = gwo.optimize(max_iter=MAX_ITER, fixed_r_values=FIXED_R_GWO)
        st.subheader("Hasil Perhitungan Langkah demi Langkah")
        convergence_data = []
        for log in history:
            st.markdown(f"---")
            st.markdown(f"### Iterasi {log['iteration']}")
            st.markdown(f"**Parameter:** `a = {log['a']:.4f}`, `r1 = {log['r1']}`, `r2 = {log['r2']}`")
            st.markdown("**Hirarki Serigala (Pemimpin):**")
            leader_df = pd.DataFrame({
                "Alpha": np.round(log['alpha_pos'], 4),
                "Beta": np.round(log['beta_pos'], 4),
                "Delta": np.round(log['delta_pos'], 4)
            }, index=["x1", "x2"])
            st.dataframe(leader_df)
            iter_df = pd.DataFrame(log['positions_before'], columns=["Posisi Awal x1", "Posisi Awal x2"])
            iter_df["Fitness Awal"] = log['fitness_before']
            iter_df["Posisi Baru x1"] = [p[0] for p in log['positions_after']]
            iter_df["Posisi Baru x2"] = [p[1] for p in log['positions_after']]
            iter_df["Fitness Baru"] = [gwo.sphere_function(p) for p in log['positions_after']]
            iter_df.index = [f"Serigala {i+1}" for i in range(N_PARTICLES)]
            st.dataframe(iter_df.style.format("{:.4f}"))
            st.markdown(f"**Best Fitness (Alpha Score) di akhir iterasi:** `{log['best_fitness']:.4f}`")
            convergence_data.append({'Iterasi': log['iteration'], 'Best Fitness': log['best_fitness']})

        st.subheader("Hasil Akhir Optimasi GWO")
        st.success(f"Posisi Terbaik (Alpha): {np.round(alpha_pos, 6)}")
        st.success(f"Nilai Fitness Terbaik: {alpha_score:.6f}")
        st.subheader("Grafik Konvergensi GWO")
        convergence_df = pd.DataFrame(convergence_data).set_index('Iterasi')
        st.line_chart(convergence_df)
