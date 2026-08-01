"""Telemetry plot generator for Mario Kart DS RL training runs.

Reads a ``telemetry_log.csv`` produced by the agent during a training run and
generates five diagnostic PNG plots saved alongside the run's other outputs.

Plots produced:
    1. **heatmap.png**       – 2-D kernel-density estimate of track position
                               (pos_x vs pos_z) showing where the kart spent
                               most of its time.
    2. **actions.png**       – Bar chart of how frequently each discrete action
                               (Gas / Gas+Left / Gas+Right) was chosen.
    3. **reasons.png**       – Pie chart of episode-termination reasons logged
                               in the ``reason`` column.
    4. **speed_offroad.png** – Scatter plot correlating the offroad modifier
                               with the kart's speed.
    5. **cumulative_reward.png** – Line chart of cumulative reward over all
                                   logged training steps (only if a ``reward``
                                   column is present).

Usage::

    python analysis/plot_generator.py   # from project root
    python plot_generator.py            # from analysis/ directory

The script resolves the ``outputs/`` directory relative to its own location,
so it works correctly regardless of the calling working directory.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path


def generate_plots():
    """Generate performance plots from a user-selected telemetry CSV log.

    Presents an interactive CLI menu listing every subdirectory found inside
    ``<project_root>/outputs/``.  The user selects a run by index; the
    function then reads ``<run>/logs/telemetry_log.csv`` and writes five PNG
    plots to ``<run>/plots/``.

    Plots generated:
        1. **heatmap.png** - KDE density of kart position (pos_x x pos_z).
           Raw coordinates are NDS fixed-point units (divide by 4096.0 for
           real-world metres).
        2. **actions.png** - Bar chart of discrete-action frequencies with
           human-readable labels (0 -> "Gas", 1 -> "Gas + Left",
           2 -> "Gas + Right").
        3. **reasons.png** - Pie chart of episode-termination reasons drawn
           from the non-null rows of the ``reason`` column.
        4. **speed_offroad.png** - Scatter of ``offroad`` modifier vs
           ``speed``; low offroad values indicate the kart is on grass.
        5. **cumulative_reward.png** - Cumulative sum of the ``reward`` column
           plotted against ``step`` (skipped when ``reward`` column absent).

    Returns:
        None: All output is written to disk; nothing is returned.

    Raises:
        FileNotFoundError: Implicitly via ``pd.read_csv`` if the selected
            run's ``telemetry_log.csv`` does not exist.

    Note:
        The function returns early (with an error message) if the ``outputs/``
        directory does not exist or contains no run subdirectories.

    Example::

        >>> generate_plots()
        0: run_20240101_120000
        1: run_20240102_090000
        Select Run Index: 0
        Plots saved to .../outputs/run_20240101_120000/plots/
    """
    # Apply a clean whitegrid theme globally before any figure is created.
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 15,
        'axes.titleweight': 'bold',
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.autolayout': True  # Prevents labels from being clipped on save.
    })

    # Anchor to project root via __file__ so the script works whether run from
    # the project root (`python analysis/plot_generator.py`) or from analysis/.
    base_dir = str(Path(__file__).resolve().parent.parent / "outputs")

    if not os.path.exists(base_dir):
        print(f"Error: Directory '{base_dir}' not found. Current path: {os.getcwd()}")
        return

    # Collect only immediate subdirectories -- each represents one training run.
    runs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    if not runs:
        print("No runs found in outputs directory.")
        return

    # Print an indexed menu so the user can identify runs by name.
    print("\n".join([f"{i}: {r}" for i, r in enumerate(runs)]))
    try:
        choice = int(input("Select Run Index: "))
        run_path = os.path.join(base_dir, runs[choice])
    except (ValueError, IndexError):
        print("Invalid selection.")
        return

    # Load the flat telemetry CSV; every row is one environment step.
    csv_path = os.path.join(run_path, "logs/telemetry_log.csv")
    df = pd.read_csv(csv_path)

    # Ensure the output directory exists before writing any figures.
    plot_dir = os.path.join(run_path, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Position Heatmap                                                  #
    # ------------------------------------------------------------------ #
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.set_facecolor('white')  # Override whitegrid grey to keep KDE colours vivid.

    # Scale coordinates from NDS fixed-point to meters
    df_scaled = df.copy()
    df_scaled['pos_x_m'] = df_scaled['pos_x'] / 4096.0
    df_scaled['pos_z_m'] = df_scaled['pos_z'] / 4096.0

    sns.kdeplot(
        data=df_scaled, x="pos_x_m", y="pos_z_m",
        fill=True,         # Shade the density regions rather than just contour lines.
        thresh=0.05,       # Drop the bottom 5 % density band to reduce noise.
        levels=100,        # High level count gives a smooth gradient appearance.
        cmap="mako",       # Dark sequential palette -- good contrast on white bg.
        antialiased=True   # Smooth the contour edges for publication quality.
    )

    plt.title("Track Position Density")
    plt.xlabel("Position X (meters)")
    plt.ylabel("Position Z (meters)")
    ax = plt.gca()
    ax.set_axisbelow(False)
    ax.grid(True, linestyle='-', alpha=0.5, color='black')
    # Note: pos_x/pos_z are scaled by 4096.0 to convert raw NDS fixed-point to real-world meters.
    plt.savefig(os.path.join(plot_dir, "heatmap.png"), dpi=300, facecolor='white')
    plt.close()

    # ------------------------------------------------------------------ #
    # 2. Action Frequency (With Descriptive Labels)                        #
    # ------------------------------------------------------------------ #
    plt.figure(figsize=(8, 6))
    # Map integer action IDs to human-readable strings for axis tick labels.
    action_map = {0: "Gas", 1: "Gas + Left", 2: "Gas + Right", 3: "Drift", 4: "Drift + Left", 5: "Drift + Right"}
    action_counts = df['action'].value_counts().sort_index()
    labels = [action_map.get(x, str(x)) for x in action_counts.index]
    # hue=labels + legend=False: satisfies seaborn's categorical colour API
    # while still applying distinct palette colours without a redundant legend.
    sns.barplot(x=labels, y=action_counts.values, palette="viridis", hue=labels, legend=False)
    plt.title("Action Distribution")
    plt.xlabel("Action Type")
    plt.ylabel("Frequency")
    ax = plt.gca()
    ax.set_axisbelow(False)
    ax.grid(True, linestyle='-', alpha=0.7, color='black')
    plt.savefig(os.path.join(plot_dir, "actions.png"), dpi=300)
    plt.close()

    # ------------------------------------------------------------------ #
    # 3. Terminal Reasons                                                  #
    # ------------------------------------------------------------------ #
    plt.figure(figsize=(8, 8))
    # Filter to rows where a termination reason was recorded (non-NaN).
    reason_counts = df[df['reason'].notna()]['reason'].value_counts()
    if not reason_counts.empty:
        plt.pie(
            reason_counts,
            labels=reason_counts.index,
            autopct='%1.1f%%',              # Display percentage with one decimal place.
            startangle=140,                 # Rotate start so the largest slice is at the top-left.
            colors=sns.color_palette("pastel"),  # Soft palette avoids harsh colours in a pie.
            wedgeprops={'edgecolor': 'white'},   # White borders separate adjacent slices cleanly.
            textprops={'fontsize': 14, 'weight': 'bold'}
        )
        plt.title("Episode Termination Reasons", fontsize=20, fontweight='bold')
    plt.savefig(os.path.join(plot_dir, "reasons.png"), dpi=300)
    plt.close()

    # ------------------------------------------------------------------ #
    # 4. Speed vs Offroad Correlation                                      #
    # ------------------------------------------------------------------ #
    plt.figure(figsize=(8, 6))
    
    # Scale from 12-bit fixed point
    df['offroad_scaled'] = df['offroad'] / 4096.0
    df['speed_scaled'] = df['speed'] / 4096.0
    
    # Use hexbin to handle overplotting of millions of steps
    plt.hexbin(df['offroad_scaled'], df['speed_scaled'], gridsize=50, cmap='YlOrRd', mincnt=1)
    cb = plt.colorbar(label='Count')
    
    plt.title("Speed vs. Offroad Performance")
    plt.xlabel("Offroad Modifier (Scaled, Lower = More Grass)")
    plt.ylabel("Speed (Scaled)")
    ax = plt.gca()
    ax.set_axisbelow(False)
    ax.grid(True, linestyle='-', alpha=0.5, color='black')
    plt.savefig(os.path.join(plot_dir, "speed_offroad.png"), dpi=300)
    plt.close()

    # ------------------------------------------------------------------ #
    # 5. Reward Progress                                                   #
    # ------------------------------------------------------------------ #
    if 'reward' in df.columns:
        plt.figure(figsize=(10, 6))
        
        # Calculate rolling average over 1000 steps
        window = 1000
        df['rolling_reward'] = df['reward'].rolling(window=window).mean()

        plt.plot(df['step'], df['rolling_reward'], color='green', linewidth=2)
        plt.title(f"Rolling Average Reward (Window = {window} Steps)")
        plt.xlabel("Step")
        plt.ylabel("Average Reward")
        ax = plt.gca()
        ax.set_axisbelow(False)
        ax.grid(True, linestyle='-', alpha=0.5, color='black')
        plt.savefig(os.path.join(plot_dir, "rolling_reward.png"), dpi=300)

    print(f"Plots saved to {plot_dir}/")


if __name__ == "__main__":
    generate_plots()