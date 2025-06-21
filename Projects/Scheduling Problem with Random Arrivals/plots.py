import matplotlib.pyplot as plt
import pandas as pd


colors = plt.cm.tab10.colors  # a palette with enough distinct colors


def plot_gantt_chart(intervals, n_servers, save_path=None, title = "Gantt Chart of Job Service Intervals", show = True):
    # Plot Gantt chart
    fig, ax = plt.subplots(figsize=(10, 4))
    for server_id, start, end, name in intervals:
        ax.barh(server_id,
                end - start,
                left=start,
                height=0.8,
                color=colors[server_id % len(colors)],  # pick color by server
                edgecolor='black')

    # Formatting
    ax.set_ylabel("Server")
    ax.set_xlabel("Time")
    ax.set_yticks(range(n_servers))
    ax.set_yticklabels([f"Server {i+1}" for i in range(n_servers)])
    ax.set_title(title)
    plt.tight_layout()
    if show:
        plt.show()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Gantt chart saved to {save_path}")