import matplotlib.pyplot as plt
from collections import Counter
import matplotlib

colors = plt.cm.tab20.colors  # up to 20 distinct colors; if you have more jobs, switch to 'hsv'

def plot_gantt_chart(intervals, n_servers, save_path=None,
                     title="Gantt Chart of Job Service Intervals", show=True):
    """
    Plot a Gantt chart assigning each job a unique color, cycling through the palette
    as new jobs appear. All segments of the same job share the same color.
    """
    # Build mapping from job name to color, in order of first appearance
    job_to_color = {}
    next_color_idx = 0

    # Determine color for each job based on appearance
    for _, start, end, name in sorted(intervals, key=lambda x: x[1]):
        if name not in job_to_color:
            job_to_color[name] = colors[next_color_idx % len(colors)]
            next_color_idx += 1

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    for server_id, start, end, name in intervals:
        ax.barh(server_id,
                end - start,
                left=start,
                height=0.8,
                color=job_to_color[name],
                edgecolor='black')

    # Optional: print job counts for sanity
    counts = Counter(sid for sid, *_ in intervals)
    print("jobs per server index:", counts)

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
    