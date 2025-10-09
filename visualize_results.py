# visualize_results.py
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob

sns.set_theme(style="whitegrid")

def plot_summary_comparisons(df, output_dir):
    """요약 CSV를 기반으로 프로토콜/암호별 평균 성능을 비교합니다."""
    if df.empty:
        print("Summary data is empty, skipping summary plots.")
        return

    plt.figure(figsize=(15, 8))
    sns.barplot(x='proto', y='avg_fps', hue='cipher', data=df, palette='viridis')
    plt.title('Average FPS Comparison', fontsize=16)
    plt.ylabel('Average FPS')
    plt.xlabel('Protocol')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_fps_comparison.png"))
    plt.close()

    plt.figure(figsize=(15, 8))
    sns.barplot(x='proto', y='rtt_avg_ms', hue='cipher', data=df, palette='plasma')
    plt.title('Average RTT (ms) Comparison', fontsize=16)
    plt.ylabel('Average RTT (ms)')
    plt.xlabel('Protocol')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_rtt_comparison.png"))
    plt.close()
    print(f"Saved summary plots to {output_dir}")

def plot_timeseries(log_dir, output_dir):
    """상세 로그들을 기반으로 시계열 꺾은선 그래프를 생성합니다."""
    log_files = glob.glob(os.path.join(log_dir, "*_timeseries.csv"))
    if not log_files:
        print(f"No time-series logs found in {log_dir}. Skipping time-series plots.")
        return

    all_data = []
    for f in log_files:
        df = pd.read_csv(f)
        run_name = os.path.basename(f).replace("_timeseries.csv", "")
        df['run'] = run_name
        all_data.append(df)
    
    if not all_data:
        print("No data to plot in time-series logs.")
        return
        
    full_df = pd.concat(all_data, ignore_index=True)

    metrics_to_plot = ["cpu_pct", "gpu_pct", "mem_pct", "interval_fps", "rtt_ms"]
    for metric in metrics_to_plot:
        if metric not in full_df.columns or full_df[metric].isnull().all():
            print(f"Skipping plot for '{metric}' as it contains no data.")
            continue

        plt.figure(figsize=(18, 9))
        sns.lineplot(data=full_df, x='time_sec', y=metric, hue='run', marker='o', markersize=4, linestyle='-')
        plt.title(f'Time-Series: {metric.upper()} Over Time', fontsize=16)
        plt.xlabel('Time (seconds)')
        plt.ylabel(metric)
        plt.legend(title='Test Run', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        save_path = os.path.join(output_dir, f"timeseries_{metric}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved time-series plot for {metric} to {save_path}")

def plot_percentile_comparisons(df, output_dir):
    """Percentile CSV를 기반으로 p50, p95, p99 비교 그래프를 생성합니다."""
    if df.empty:
        print("Percentile data is empty, skipping percentile plots.")
        return

    metrics = [
        ('cpu', 'CPU Usage (%)', ['cpu_p50', 'cpu_p95', 'cpu_p99']),
        ('gpu', 'GPU Usage (%)', ['gpu_p50', 'gpu_p95', 'gpu_p99']),
        ('mem', 'Memory Usage (%)', ['mem_p50', 'mem_p95', 'mem_p99']),
        ('fps', 'FPS', ['fps_p50', 'fps_p95', 'fps_p99']),
    ]

    for metric_key, metric_title, columns in metrics:
        if not all(col in df.columns for col in columns):
            print(f"Skipping {metric_key} percentile plot - missing columns")
            continue

        # Reshape data for grouped bar plot
        plot_data = []
        for _, row in df.iterrows():
            label = f"{row['proto']}_{row['cipher']}"
            plot_data.append({'config': label, 'p50': row[columns[0]], 'p95': row[columns[1]], 'p99': row[columns[2]]})

        plot_df = pd.DataFrame(plot_data)

        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(len(plot_df))
        width = 0.25

        ax.bar([i - width for i in x], plot_df['p50'], width, label='p50', alpha=0.8)
        ax.bar([i for i in x], plot_df['p95'], width, label='p95', alpha=0.8)
        ax.bar([i + width for i in x], plot_df['p99'], width, label='p99', alpha=0.8)

        ax.set_xlabel('Configuration')
        ax.set_ylabel(metric_title)
        ax.set_title(f'{metric_title} Percentiles Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df['config'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"percentile_{metric_key}_comparison.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved percentile plot for {metric_key} to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize benchmark results.")
    parser.add_argument("summary_csv_file", help="Path to the input CSV file (e.g., out_results.csv)")
    parser.add_argument("--log-dir", default="timeseries_logs", help="Directory containing the time-series log files")
    parser.add_argument("--out-dir", default="plots", help="Directory to save plot images")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if os.path.exists(args.summary_csv_file):
        summary_df = pd.read_csv(args.summary_csv_file)
        plot_summary_comparisons(summary_df, args.out_dir)
    else:
        print(f"Summary file not found: {args.summary_csv_file}")

    # Plot percentile comparisons
    percentile_file = args.summary_csv_file.replace(".csv", "_percentiles.csv")
    if os.path.exists(percentile_file):
        percentile_df = pd.read_csv(percentile_file)
        plot_percentile_comparisons(percentile_df, args.out_dir)
    else:
        print(f"Percentile file not found: {percentile_file}")

    plot_timeseries(args.log_dir, args.out_dir)

    print("\nAll plots generated.")

if __name__ == "__main__":
    main()
