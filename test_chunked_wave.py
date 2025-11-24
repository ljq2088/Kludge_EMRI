import numpy as np
import matplotlib.pyplot as plt
import time
import h5py
import os
from dataclasses import dataclass

# Import core modules
from src.emrikludge.orbits.nk_geodesic_orbit import BabakNKOrbit
from src.emrikludge.waveforms.nk_waveform import compute_nk_waveform, ObserverInfo
from src.emrikludge.orbits.nk_mapping import get_conserved_quantities

# Try importing C++ extension
try:
    from src.emrikludge._emrikludge import BabakNKOrbit_CPP
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    print("⚠️ Warning: C++ extension not found.")

# ==========================================
# 0. Helpers & Config
# ==========================================
G_SI = 6.67430e-11
C_SI = 299792458.0
M_SUN_SI = 1.989e30
SEC_PER_YEAR = 31536000.0

@dataclass
class CppTrajectoryAdapter:
    """Adapter to make C++ vector data look like Python objects"""
    t: np.ndarray
    p: np.ndarray
    e: np.ndarray
    iota: np.ndarray
    r_over_M: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    psi: np.ndarray
    chi: np.ndarray
    
    @property
    def r(self):
        return self.r_over_M

def save_trajectory_to_h5(filename, cpp_states, M, mu, a):
    """Save C++ results to HDF5"""
    n = len(cpp_states)
    print(f"[IO] Saving {n} steps to {filename}...")
    
    with h5py.File(filename, 'w') as f:
        f.attrs['M'] = M
        f.attrs['mu'] = mu
        f.attrs['a'] = a
        # Save dt if we have points, else 0
        f.attrs['dt'] = cpp_states[1].t - cpp_states[0].t if n > 1 else 0
        
        # Create datasets with compression
        f.create_dataset('t', data=[s.t for s in cpp_states], compression="gzip")
        f.create_dataset('p', data=[s.p for s in cpp_states], compression="gzip")
        f.create_dataset('e', data=[s.e for s in cpp_states], compression="gzip")
        f.create_dataset('iota', data=[s.iota for s in cpp_states], compression="gzip")
        f.create_dataset('r', data=[s.r for s in cpp_states], compression="gzip")
        f.create_dataset('theta', data=[s.theta for s in cpp_states], compression="gzip")
        f.create_dataset('phi', data=[s.phi for s in cpp_states], compression="gzip")
        f.create_dataset('psi', data=[s.psi for s in cpp_states], compression="gzip")
        f.create_dataset('chi', data=[s.chi for s in cpp_states], compression="gzip")
    
    print("[IO] Trajectory saved.")

def compute_waveform_io_stream(h5_filename, observer, batch_size=500000, overlap=2000):
    """Streamed waveform calculation: Read -> Calc -> Write"""
    print(f"[Waveform] Starting stream processing on {h5_filename}")
    
    with h5py.File(h5_filename, 'r+') as f:
        M = f.attrs['M']
        mu = f.attrs['mu']
        total_points = f['t'].shape[0]
        
        if total_points < 2:
            print("Error: Trajectory data is empty or too short!")
            return

        dt = f['t'][1] - f['t'][0]
        print(f"  Detected dt = {dt:.4f} M. Total points: {total_points}")

        if 'h_plus' in f:
            print("  Overwriting existing waveform data...")
            del f['h_plus'], f['h_cross']
            
        dset_h_plus = f.create_dataset('h_plus', shape=(0,), maxshape=(None,), chunks=True, compression="gzip")
        dset_h_cross = f.create_dataset('h_cross', shape=(0,), maxshape=(None,), chunks=True, compression="gzip")
        
        for start_idx in range(0, total_points, batch_size):
            end_idx = min(start_idx + batch_size, total_points)
            slice_end = min(end_idx + overlap, total_points)
            
            # Safety check for small tails
            chunk_len = slice_end - start_idx
            if chunk_len < 4:
                print(f"  [Warning] Skipping tiny tail batch (size {chunk_len}).")
                continue

            # Load chunk
            traj_chunk = CppTrajectoryAdapter(
                t=f['t'][start_idx:slice_end],
                p=f['p'][start_idx:slice_end],
                e=f['e'][start_idx:slice_end],
                iota=f['iota'][start_idx:slice_end],
                r_over_M=f['r'][start_idx:slice_end],
                theta=f['theta'][start_idx:slice_end],
                phi=f['phi'][start_idx:slice_end],
                psi=f['psi'][start_idx:slice_end],
                chi=f['chi'][start_idx:slice_end]
            )
            
            # Compute waveform
            hp_chunk, hc_chunk = compute_nk_waveform(traj_chunk, mu, M, observer, dt)
            
            # Write valid part
            valid_len = end_idx - start_idx
            valid_len = min(valid_len, len(hp_chunk))
            
            current_size = dset_h_plus.shape[0]
            new_size = current_size + valid_len
            dset_h_plus.resize((new_size,))
            dset_h_cross.resize((new_size,))
            
            dset_h_plus[current_size:] = hp_chunk[:valid_len]
            dset_h_cross[current_size:] = hc_chunk[:valid_len]
            
            print(f"  [Batch] Processed {new_size}/{total_points} ({new_size/total_points*100:.1f}%)")

    print("[Waveform] All done. Data saved to H5.")

def check_existing_file(filename):
    """Check if file exists and has valid data"""
    if not os.path.exists(filename):
        return False
    try:
        with h5py.File(filename, 'r') as f:
            if 't' not in f or f['t'].shape[0] < 10:
                print(f"[Setup] Existing file {filename} is empty or corrupt. Re-running evolution.")
                return False
            print(f"[Setup] Found valid existing file with {f['t'].shape[0]} steps.")
            return True
    except OSError:
        print(f"[Setup] File {filename} is corrupt. Re-running.")
        return False

def run_production_pipeline():
    filename = "emri_waveform_complete.h5"
    
    # ==========================================
    # 1. Setup
    # ==========================================
    M_phys_solar = 1e6
    mu_phys_solar = 10.0
    a_spin = 0.7
    p_init, e_init, iota_init = 10.0, 0.6, np.radians(30.0)
    
    M_kg = M_phys_solar * M_SUN_SI
    T_unit_sec = G_SI * M_kg / (C_SI**3)
    
    target_years = 1.0
    target_sec = target_years * SEC_PER_YEAR
    total_duration_M = target_sec / T_unit_sec
    
    # To fix index errors, ensure dt_M isn't too small relative to duration
    # But for 1 year run, 0.1-2.0 is fine.
    dt_M = 0.1014826  # Using your previous specific value or set to 1.0/2.0
    
    print(f"=== EMRI Chunked Waveform Generation ===")
    print(f"[Physics] M = {M_phys_solar:.1e} M_sun, mu = {mu_phys_solar:.1f} M_sun")
    print(f"[Units] 1 M = {T_unit_sec:.4f} s")
    print(f"[Setup] Target Duration: {target_years} Year = {target_sec:.2e} s")
    print(f"[Setup] In Geometric Units: {total_duration_M:.2e} M")
    print(f"[Setup] Sampling dt: {dt_M} M -> Total points ~ {int(total_duration_M/dt_M)}")

    # -------------------------------------------------
    # Step 1: Evolution (C++)
    # -------------------------------------------------
    # Robust check: Only skip if file exists AND has data
    file_is_valid = check_existing_file(filename)

    if not file_is_valid:
        print("\n[1/3] Running Evolution (C++)...")
        if not CPP_AVAILABLE:
            raise RuntimeError("Need C++ extension for long run!")
            
        start_t = time.time()
        orbiter = BabakNKOrbit_CPP(M_phys_solar, a_spin, p_init, e_init, iota_init, mu_phys_solar)
        
        # Run evolution
        cpp_results = orbiter.evolve(total_duration_M, dt_M)
        
        print(f"[C++] Evolution finished in {time.time()-start_t:.2f} s")
        
        if len(cpp_results) == 0:
            print("Error: C++ evolution returned 0 steps! Check parameters or C++ logic.")
            return

        save_trajectory_to_h5(filename, cpp_results, M_phys_solar, mu_phys_solar, a_spin)
        
        del cpp_results
        del orbiter
        import gc; gc.collect()
    else:
        print(f"\n[1/3] Using existing valid data from {filename}.")

    # -------------------------------------------------
    # Step 2: Waveform (IO Stream)
    # -------------------------------------------------
    print("\n[2/3] Computing Waveform (Streamed)...")
    compute_waveform_io_stream(filename, ObserverInfo(1e9*3e22, np.pi/4, 0), batch_size=500000)
    
    # -------------------------------------------------
    # Step 3: Plot
    # -------------------------------------------------
    print("\n[3/3] Plotting check...")
    plot_results(filename, T_unit_sec)

def plot_results(filename, T_unit):
    with h5py.File(filename, 'r') as f:
        t = f['t']
        p = f['p']
        e = f['e']
        
        total = t.shape[0]
        if total == 0: return

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Downsample for plotting
        step = max(1, total // 2000)
        t_days = np.array(t[::step]) * T_unit / 86400.0
        
        axes[0].plot(t_days, p[::step], label='p')
        axes[0].set_ylabel('p/M')
        axes[0].set_xlabel('Time (Days)')
        axes[0].set_title("Long-term Evolution")
        ax0r = axes[0].twinx()
        ax0r.plot(t_days, e[::step], color='orange', label='e', linestyle='--')
        ax0r.set_ylabel('Eccentricity')
        
        # Plot waveform if exists
        if 'h_plus' in f:
            h = f['h_plus']
            view_len = 2000
            start = max(0, h.shape[0] - view_len)
            t_wave = np.array(t[start:start+view_len]) * T_unit
            h_wave = np.array(h[start:start+view_len])
            
            if len(h_wave) > 0:
                axes[1].plot(t_wave - t_wave[0], h_wave)
                axes[1].set_ylabel("Strain")
                axes[1].set_title("Waveform (Tail)")
        
        plt.tight_layout()
        plt.savefig("Chunked_Result.png")
        print("[Plot] Saved Chunked_Result.png")

if __name__ == "__main__":
    run_production_pipeline()