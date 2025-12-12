import argparse
import os

import torch
import numpy as np
import matplotlib.pyplot as plt


def load_metrics(path):
    data = torch.load(path)

    out = {
        # per-env scalars
        "slip_distance": data["slip_distance"].numpy(),
        "max_contact_force": data["max_contact_force"].numpy(),
        "N_min": data["N_min"].numpy(),
        "gentle_ratio": data["gentle_ratio"].numpy(),
        "success_mask": data["success_mask"].numpy().astype(bool),
        "success_rate": float(data["success_rate"]),
        # time series
        "ts_contact_force": data["ts_contact_force"].numpy(),  # (T, num_envs)
        "ts_stiffness": data["ts_stiffness"].numpy(),
        "ts_slip_speed": data["ts_slip_speed"].numpy(),
        "ts_slip_flag": data["ts_slip_flag"].numpy().astype(bool),
        "ts_lifted": data["ts_lifted"].numpy().astype(bool),
        # meta
        "dt_step": float(data["dt_step"]),
        "SLIP_THRESH": float(data["SLIP_THRESH"]),
        "RATIO_THRESH": float(data["RATIO_THRESH"]),
        "LIFT_HEIGHT_THRESH": float(data["LIFT_HEIGHT_THRESH"]),
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline_file",
        type=str,
        default="baseline.pt",
        help="Path to baseline controller metrics (.pt)",
    )
    parser.add_argument(
        "--adaptive_file",
        type=str,
        default="adaptive.pt",
        help="Path to adaptive controller metrics (.pt)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="figs_compare",
        help="Directory to save comparison figures",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load both data files
    # ------------------------------------------------------------------
    base = load_metrics(args.baseline_file)
    adap = load_metrics(args.adaptive_file)

    # Basic checks
    num_envs_base = base["slip_distance"].shape[0]
    num_envs_adap = adap["slip_distance"].shape[0]

    if num_envs_base != num_envs_adap:
        print(f"WARNING: num_envs differ: baseline={num_envs_base}, adaptive={num_envs_adap}")
    num_envs = min(num_envs_base, num_envs_adap)

    # Truncate to common env count if needed
    def trunc_env(x):
        # x is (num_envs, ...) or (T, num_envs)
        if x.ndim == 1:
            return x[:num_envs]
        elif x.ndim == 2:
            return x[:, :num_envs]
        else:
            return x  # not expected, but safe

    for key in ["slip_distance", "max_contact_force", "N_min", "gentle_ratio",
                "success_mask"]:
        base[key] = trunc_env(base[key])
        adap[key] = trunc_env(adap[key])

    # time series
    # match time lengths too (in case episodes ended at slightly different step)
    T_base = base["ts_contact_force"].shape[0]
    T_adap = adap["ts_contact_force"].shape[0]
    T = min(T_base, T_adap)

    for key in ["ts_contact_force", "ts_stiffness", "ts_slip_speed",
                "ts_slip_flag", "ts_lifted"]:
        base[key] = base[key][:T, :num_envs]
        adap[key] = adap[key][:T, :num_envs]

    dt_step = base["dt_step"]  # assume same
    time = np.arange(T) * dt_step

    SLIP_THRESH = base["SLIP_THRESH"]    # assume same
    RATIO_THRESH = base["RATIO_THRESH"]  # assume same

    print(f"Loaded:")
    print(f"  baseline: {args.baseline_file}")
    print(f"  adaptive: {args.adaptive_file}")
    print(f"Common num_envs = {num_envs}, common T = {T}")
    print(f"Baseline success_rate = {base['success_rate']:.3f}")
    print(f"Adaptive success_rate = {adap['success_rate']:.3f}")

    # ------------------------------------------------------------------
    # 2. Figure 1: Histogram of gentle_ratio (baseline vs adaptive)
    # ------------------------------------------------------------------
    plt.figure()
    plt.hist(
        base["gentle_ratio"],
        bins=20,
        alpha=0.6,
        label="Baseline",
        edgecolor="black",
    )
    plt.hist(
        adap["gentle_ratio"],
        bins=20,
        alpha=0.6,
        label="Adaptive",
        edgecolor="black",
    )
    plt.axvline(RATIO_THRESH, linestyle="--", color="gray",
                label=f"Threshold = {RATIO_THRESH:.2f}")
    plt.xlabel("F_max / N_min (gentle ratio)")
    plt.ylabel("Count")
    plt.title("Distribution of Gentle Grasp Ratios")
    plt.legend()
    plt.tight_layout()
    fpath1 = os.path.join(args.out_dir, "gentle_ratio_hist_compare.png")
    plt.savefig(fpath1, dpi=200)
    plt.close()
    print(f"Saved: {fpath1}")

    # ------------------------------------------------------------------
    # 3. Figure 2: Slip distance vs gentle_ratio scatter (colored by controller)
    # ------------------------------------------------------------------
    plt.figure()

    # Baseline
    plt.scatter(
        base["gentle_ratio"],
        base["slip_distance"],
        label="Baseline",
        alpha=0.7,
        marker="o",
    )
    # Adaptive
    plt.scatter(
        adap["gentle_ratio"],
        adap["slip_distance"],
        label="Adaptive",
        alpha=0.7,
        marker="x",
    )

    plt.axvline(RATIO_THRESH, linestyle="--", color="gray")
    plt.axhline(SLIP_THRESH, linestyle="--", color="gray")

    plt.xlabel("F_max / N_min (gentle ratio)")
    plt.ylabel("Slip distance (m)")
    plt.title("Slip Distance vs Gentle Ratio (Baseline vs Adaptive)")
    plt.legend()
    plt.tight_layout()
    fpath2 = os.path.join(args.out_dir, "slip_vs_gentle_ratio_compare.png")
    plt.savefig(fpath2, dpi=200)
    plt.close()
    print(f"Saved: {fpath2}")

    # ------------------------------------------------------------------
    # 4. Figure 3: Success rate bar chart (baseline vs adaptive)
    # ------------------------------------------------------------------
    plt.figure()
    success_rates = [base["success_rate"], adap["success_rate"]]
    labels = ["Baseline", "Adaptive"]
    colors = ["C0", "C1"]

    plt.bar(labels, success_rates, color=colors, alpha=0.6)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Success rate")
    plt.title("Success Rate Comparison")
    plt.tight_layout()
    fpath3 = os.path.join(args.out_dir, "success_rate_bar_compare.png")
    plt.savefig(fpath3, dpi=200)
    plt.close()
    print(f"Saved: {fpath3}")

        # ------------------------------------------------------------------
    # 5. Figure 4: Time series with mean ± std across all envs
    # ------------------------------------------------------------------
    # We aggregate over envs to see the distribution:
    #   - mean over envs at each timestep
    #   - shaded band = mean ± std over envs
    # for: contact force, stiffness, slip speed
    #   and a separate lifted fraction curve.

    # Helpers to compute mean and std over envs (axis=1: env index)
    force_base_mean = base["ts_contact_force"].mean(axis=1)
    force_base_std  = base["ts_contact_force"].std(axis=1)
    force_adap_mean = adap["ts_contact_force"].mean(axis=1)
    force_adap_std  = adap["ts_contact_force"].std(axis=1)

    stiff_base_mean = base["ts_stiffness"].mean(axis=1)
    stiff_base_std  = base["ts_stiffness"].std(axis=1)
    stiff_adap_mean = adap["ts_stiffness"].mean(axis=1)
    stiff_adap_std  = adap["ts_stiffness"].std(axis=1)

    slip_base_mean = base["ts_slip_speed"].mean(axis=1)
    slip_base_std  = base["ts_slip_speed"].std(axis=1)
    slip_adap_mean = adap["ts_slip_speed"].mean(axis=1)
    slip_adap_std  = adap["ts_slip_speed"].std(axis=1)

    # NEW: lifted fraction over time (mean of boolean in [0,1])
    lifted_base_mean = base["ts_lifted"].mean(axis=1)
    lifted_adap_mean = adap["ts_lifted"].mean(axis=1)

    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 10))

    # --- Panel 1: Contact force ---
    axs[0].plot(time, force_base_mean, label="Baseline")
    axs[0].fill_between(
        time,
        force_base_mean - force_base_std,
        force_base_mean + force_base_std,
        alpha=0.3,
    )

    axs[0].plot(time, force_adap_mean, label="Adaptive")
    axs[0].fill_between(
        time,
        force_adap_mean - force_adap_std,
        force_adap_mean + force_adap_std,
        alpha=0.3,
    )

    axs[0].set_ylabel("Force (N)")
    axs[0].set_title("Time series across envs (mean ± std)")
    axs[0].legend(loc="upper right")

    # --- Panel 2: Stiffness ---
    axs[1].plot(time, stiff_base_mean, label="Baseline")
    axs[1].fill_between(
        time,
        stiff_base_mean - stiff_base_std,
        stiff_base_mean + stiff_base_std,
        alpha=0.3,
    )

    axs[1].plot(time, stiff_adap_mean, label="Adaptive")
    axs[1].fill_between(
        time,
        stiff_adap_mean - stiff_adap_std,
        stiff_adap_mean + stiff_adap_std,
        alpha=0.3,
    )

    axs[1].set_ylabel("Stiffness")
    axs[1].legend(loc="upper right")

    # --- Panel 3: Slip speed ---
    axs[2].plot(time, slip_base_mean, label="Baseline")
    axs[2].fill_between(
        time,
        slip_base_mean - slip_base_std,
        slip_base_mean + slip_base_std,
        alpha=0.3,
    )

    axs[2].plot(time, slip_adap_mean, label="Adaptive")
    axs[2].fill_between(
        time,
        slip_adap_mean - slip_adap_std,
        slip_adap_mean + slip_adap_std,
        alpha=0.3,
    )

    axs[2].axhline(SLIP_THRESH, linestyle="--")
    axs[2].set_ylabel("Slip speed (m/s)")
    axs[2].legend(loc="upper right")

    # --- Panel 4: Lifted fraction (new) ---
    axs[3].plot(time, lifted_base_mean, label="Baseline")
    axs[3].plot(time, lifted_adap_mean, label="Adaptive")

    axs[3].set_ylabel("Lifted fraction")
    axs[3].set_xlabel("Time (s)")
    axs[3].set_ylim(-0.1, 1.1)  # in [0,1]
    axs[3].legend(loc="upper right")

    plt.tight_layout()
    fpath4 = os.path.join(args.out_dir, "time_series_mean_std_compare.png")
    plt.savefig(fpath4, dpi=200)
    plt.close()
    print(f"saved {fpath4}")


if __name__ == "__main__":
    main()
