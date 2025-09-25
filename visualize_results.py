# visualize_results.py
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set plot style
sns.set_theme(style="whitegrid")

def plot_by_cipher(df, output_dir):
    """암호 알고리즘별 성능 비교 그래프 생성"""
    # Filter for relevant protocols where cipher matters most
    df_filtered = df[df['proto'].isin(['https', 'dtls', 'http3', 'mqtt'])]
    if df_filtered.empty:
        print("Skipping cipher plots: No relevant data found.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(15, 20), sharex=True)
    fig.suptitle('Performance Comparison by Cipher Suite', fontsize=16, y=0.95)

    # 1. Average FPS by Cipher
    sns.barplot(ax=axes[0], x='proto', y='avg_fps', hue='cipher', data=df_filtered, palette='viridis')
    axes[0].set_title('Average FPS (Higher is Better)')
    axes[0].set_ylabel('Average FPS')

    # 2. CPU Percentage by Cipher
    sns.barplot(ax=axes[1], x='proto', y='cpu_pct', hue='cipher', data=df_filtered, palette='viridis')
    axes[1].set_title('CPU Usage % (Lower is Better)')
    axes[1].set_ylabel('CPU Usage %')

    # 3. RTT p50 by Cipher
    if 'rtt_p50_ms' in df.columns:
        sns.barplot(ax=axes[2], x='proto', y='rtt_p50_ms', hue='cipher', data=df_filtered, palette='viridis')
        axes[2].set_title('p50 Latency (ms) (Lower is Better)')
        axes[2].set_ylabel('RTT p50 (ms)')
    
    plt.xlabel('Protocol')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(output_dir, "plot_by_cipher.png")
    plt.savefig(save_path)
    print(f"Saved cipher comparison plot to {save_path}")
    plt.close()

def plot_by_protocol(df, output_dir):
    """프로토콜별 성능 비교 그래프 생성"""
    if df.empty:
        print("Skipping protocol plots: No data found.")
        return

    # Use a consistent cipher for comparison, e.g., the first one in the list
    # Or choose a common one like 'aesgcm'
    comp_cipher = df['cipher'].unique()[0] if len(df['cipher'].unique()) > 0 else ''
    df_filtered = df[df['cipher'] == comp_cipher]

    fig, axes = plt.subplots(3, 1, figsize=(15, 20), sharex=True)
    fig.suptitle(f'Performance Comparison by Protocol (Cipher: {comp_cipher})', fontsize=16, y=0.95)

    # 1. Average FPS by Protocol
    sns.barplot(ax=axes[0], x='proto', y='avg_fps', data=df_filtered, palette='plasma')
    axes[0].set_title('Average FPS (Higher is Better)')
    axes[0].set_ylabel('Average FPS')

    # 2. CPU Percentage by Protocol
    sns.barplot(ax=axes[1], x='proto', y='cpu_pct', data=df_filtered, palette='plasma')
    axes[1].set_title('CPU Usage % (Lower is Better)')
    axes[1].set_ylabel('CPU Usage %')

    # 3. RTT p50 by Protocol
    if 'rtt_p50_ms' in df.columns:
        sns.barplot(ax=axes[2], x='proto', y='rtt_p50_ms', data=df_filtered, palette='plasma')
        axes[2].set_title('p50 Latency (ms) (Lower is Better)')
        axes[2].set_ylabel('RTT p50 (ms)')

    plt.xlabel('Protocol')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(output_dir, "plot_by_protocol.png")
    plt.savefig(save_path)
    print(f"Saved protocol comparison plot to {save_path}")
    plt.close()

def plot_by_board(df, output_dir):
    """보드별 성능 비교 그래프 생성"""
    if 'board' not in df.columns or len(df['board'].unique()) < 2:
        print("Skipping board plots: Requires data from multiple boards.")
        return

    fig, axes = plt.subplots(1, 1, figsize=(15, 8))
    sns.barplot(ax=axes, x='proto', y='avg_fps', hue='board', data=df, palette='coolwarm')
    axes.set_title('Average FPS Comparison Across Boards')
    axes.set_ylabel('Average FPS')
    axes.set_xlabel('Protocol')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "plot_by_board.png")
    plt.savefig(save_path)
    print(f"Saved board comparison plot to {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize benchmark results.")
    parser.add_argument("csv_file", help="Path to the input CSV file (e.g., out_results.csv)")
    parser.add_argument("--out-dir", default="./plots", help="Directory to save plot images")
    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file not found at {args.csv_file}")
        return

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        print(f"Created output directory: {args.out_dir}")

    # Load data
    df = pd.read_csv(args.csv_file)
    print("CSV data loaded successfully.")

    # Generate plots
    plot_by_cipher(df, args.out_dir)
    plot_by_protocol(df, args.out_dir)
    plot_by_board(df, args.out_dir)

    print("\nAll plots generated.")

if __name__ == "__main__":
    main()
