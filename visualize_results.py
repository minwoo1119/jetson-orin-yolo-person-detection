# visualize_results.py
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import glob

def get_statistics(row, column_name, timeseries_dir):
    # Use glob to find the log file, accommodating for variations like _(6)
    pattern = f"{timeseries_dir}/{row['video']}_{row['proto']}_{row['cipher']}*.csv"
    log_files = glob.glob(pattern)

    if not log_files:
        return pd.Series([np.nan, np.nan, np.nan, np.nan])

    log_filename = log_files[0]  # Use the first match

    try:
        ts_df = pd.read_csv(log_filename)
        # Assuming 5s warmup, filter out initial data
        ts_df = ts_df[ts_df['time_sec'] > 5]
        if column_name in ts_df.columns and not ts_df[column_name].empty:
            # Replace 0s with NaN before calculating stats to avoid them being min
            values = ts_df[column_name].replace(0, np.nan).dropna()
            if values.empty:
                return pd.Series([np.nan, np.nan, np.nan, np.nan])

            p25 = values.quantile(0.25)
            p75 = values.quantile(0.75)
            min_val = values.min()
            max_val = values.max()
            return pd.Series([min_val, p25, p75, max_val])
        else:
            return pd.Series([np.nan, np.nan, np.nan, np.nan])
    except Exception:
        return pd.Series([np.nan, np.nan, np.nan, np.nan])

def main():
    sns.set(font_scale=1.5, style="whitegrid")
    parser = argparse.ArgumentParser(description="Visualize benchmark results from a timestamped result directory.")
    parser.add_argument("--result-dir", required=True, help="Path to the result directory (e.g., results/20250105_120000_baseline)")
    parser.add_argument("--protocol", help="Filter by specific protocol (e.g., https, mqtt, dtls). If not specified, all protocols will be included.")
    args = parser.parse_args()

    result_dir = args.result_dir
    if not os.path.exists(result_dir):
        print(f"Error: Result directory not found at {result_dir}")
        return

    # Find results.csv in the result directory
    csv_file = os.path.join(result_dir, "results.csv")
    if not os.path.exists(csv_file):
        print(f"Error: results.csv not found in {result_dir}")
        return

    df = pd.read_csv(csv_file)

    # Set up directories
    timeseries_dir = os.path.join(result_dir, "timeseries_logs")
    output_dir = os.path.join(result_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    # Filter by protocol if specified
    if args.protocol:
        if 'proto' in df.columns:
            df = df[df['proto'] == args.protocol]
            if df.empty:
                print(f"No data found for protocol '{args.protocol}' in the CSV file.")
                return

    if df.empty:
        print("No data to visualize.")
        return

    # Calculate statistics from timeseries logs
    df[['fps_min', 'fps_p25', 'fps_p75', 'fps_max']] = df.apply(get_statistics, axis=1, column_name='interval_fps', timeseries_dir=timeseries_dir)
    df[['rtt_min', 'rtt_p25', 'rtt_p75', 'rtt_max']] = df.apply(get_statistics, axis=1, column_name='rtt_ms', timeseries_dir=timeseries_dir)
    df.dropna(subset=['fps_min', 'rtt_min'], inplace=True)

    if df.empty:
        print("Could not calculate statistics. Check timeseries logs.")
        return

    unique_ciphers = sorted(df['cipher'].unique())
    spring_warm_palette = ["#f56368", "#f7de6f", "#51daa7", "#7c8ed6", "#d473c7"]
    palette = sns.color_palette(spring_warm_palette, len(unique_ciphers))
    cipher_color_map = {cipher: color for cipher, color in zip(unique_ciphers, palette)}

    # --- Plotting FPS Custom Box Plot ---
    plt.figure(figsize=(12, 7))
    x_pos = np.arange(len(unique_ciphers))

    # Draw whiskers (min to max lines)
    for i, cipher in enumerate(unique_ciphers):
        stats = df[df['cipher'] == cipher].iloc[0]
        plt.vlines(x=i, ymin=stats['fps_min'], ymax=stats['fps_max'], color=cipher_color_map[cipher], linewidth=2)

    # Draw IQR bars (p25 to p75)
    bar_heights = df['fps_p75'] - df['fps_p25']
    bar_bottoms = df['fps_p25']
    plt.bar(x=x_pos, height=bar_heights, bottom=bar_bottoms, color=[cipher_color_map[c] for c in unique_ciphers], alpha=0.7, width=0.5)

    plt.xlabel('Cipher Suite')
    plt.ylabel('FPS')
    plt.title('FPS Distribution (Min, P25-P75, Max) per Cipher Suite')
    plt.xticks(x_pos, unique_ciphers, rotation=45, ha='right')

    min_val = df['fps_min'].min()
    max_val = df['fps_max'].max()
    plt.ylim(bottom=min_val * 0.9, top=max_val * 1.1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fps_range_comparison.png"))
    plt.close()
    print(f"Saved plot: {os.path.join(output_dir, 'fps_range_comparison.png')}")

    # --- Plotting RTT Custom Box Plot ---
    plt.figure(figsize=(12, 7))

    for i, cipher in enumerate(unique_ciphers):
        stats = df[df['cipher'] == cipher].iloc[0]
        plt.vlines(x=i, ymin=stats['rtt_min'], ymax=stats['rtt_max'], color=cipher_color_map[cipher], linewidth=2)

    bar_heights = df['rtt_p75'] - df['rtt_p25']
    bar_bottoms = df['rtt_p25']
    plt.bar(x=x_pos, height=bar_heights, bottom=bar_bottoms, color=[cipher_color_map[c] for c in unique_ciphers], alpha=0.7, width=0.5)

    plt.xlabel('Cipher Suite')
    plt.ylabel('RTT (ms)')
    plt.title('RTT Distribution (Min, P25-P75, Max) per Cipher Suite')
    plt.xticks(x_pos, unique_ciphers, rotation=45, ha='right')

    min_val = df['rtt_min'].min()
    max_val = df['rtt_max'].max()
    plt.ylim(bottom=min_val * 0.9, top=max_val * 1.1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rtt_range_comparison.png"))
    plt.close()
    print(f"Saved plot: {os.path.join(output_dir, 'rtt_range_comparison.png')}")

    # --- Keep other plots as they were ---
    metrics_to_plot = {
        'cpu_pct': 'CPU Usage (%)',
        'gpu_pct': 'GPU Usage (%)',
    }
    for metric, title in metrics_to_plot.items():
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='cipher', y=metric, data=df, hue='cipher', dodge=False, palette=palette)
        if metric == 'cpu_pct':
            plt.ylim(bottom=200)
        plt.xlabel('Cipher Suite')
        plt.ylabel(title)
        plt.title(f'{title} per Cipher Suite')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_filename = os.path.join(output_dir, f"{metric}_comparison.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved plot: {plot_filename}")

    print(f"\nAll plots saved to: {output_dir}")

if __name__ == "__main__":
    main()
