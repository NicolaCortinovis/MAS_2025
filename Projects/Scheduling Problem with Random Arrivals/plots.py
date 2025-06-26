import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import scipy.stats as stats
import matplotlib.cm as cm
import matplotlib.colors as mcolors


colors = plt.cm.tab20.colors  # up to 20 distinct colors; if you have more jobs, switch to 'hsv'


def plot_gantt_chart(intervals, n_servers, arrival_times_by_job, save_path=None, 
                     title="Gantt Chart of Job Service Intervals", show=True):
    """
    Plot a Gantt chart assigning each job a unique color + hatch pattern, based on arrival time.
    All segments of the same job share the same color and hatch.
    """

    hatches = ['', '/', '.', '\\']

    # Map each job to a (color, hatch) pair based on order of arrival
    job_to_style = {}
    for i, name in enumerate(sorted(arrival_times_by_job, key=arrival_times_by_job.get)):
        color = colors[i % len(colors)]
        hatch = hatches[(i // len(colors)) % len(hatches)]
        job_to_style[name] = (color, hatch)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    for server_id, start, end, name in intervals:
        color, hatch = job_to_style.get(name, ('gray', ''))
        ax.barh(server_id,
                end - start,
                left=start,
                height=0.8,
                color=color,
                hatch=hatch,
                edgecolor='black')

    # Optional sanity check
    counts = Counter(sid for sid, *_ in intervals)

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


# Exponential and Lognormal distributions plots
def plot_service_distribution(service_dist1, params1, service_dist2=None, params2=None, x_range=None, num_points=1000):
    """
    Plot one or two service distributions with given parameters on the same axes.
    :param service_dist1: 'exponential' or 'lognormal' for the first distribution
    :param params1: dictionary of parameters for the first distribution
    :param service_dist2: (optional) 'exponential' or 'lognormal' for the second distribution
    :param params2: (optional) dictionary of parameters for the second distribution
    :param x_range: tuple (min, max) for x-axis limits; if None, auto-calculate
    :param num_points: number of points to plot
    """

    # Helper to get x and y for a distribution, and its mean
    def get_xy_and_mean(dist, params):
        if dist == 'exponential':
            rate = params.get('rate', 1.0)
            x = np.linspace(0, 5 / rate, num_points)
            y = stats.expon.pdf(x, scale=1/rate)
            mean = 1 / rate
        elif dist == 'lognormal':
            mu = params.get('mu', 0.0)
            sigma = params.get('sigma', 1.0)
            x = np.linspace(0, 5 * np.exp(mu + sigma**2), num_points)
            y = stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu))
            mean = np.exp(mu + sigma**2 / 2)
        else:
            raise ValueError(f"Unknown distribution: {dist}")
        return x, y, mean

    x1, y1, mean1 = get_xy_and_mean(service_dist1, params1)
    x2, y2, mean2 = (None, None, None)
    if service_dist2 and params2:
        x2, y2, mean2 = get_xy_and_mean(service_dist2, params2)

    # Determine x_range
    if x_range is None:
        x_min = x1.min()
        x_max = x1.max()
        if x2 is not None:
            x_min = min(x_min, x2.min())
            x_max = max(x_max, x2.max())
        x_range = (x_min, x_max)

    plt.figure(figsize=(10, 6))
    plt.plot(x1, y1, label=f"{service_dist1.capitalize()} {params1}")
    plt.scatter([mean1], [stats.expon.pdf(mean1, scale=1/params1.get('rate', 1.0)) if service_dist1 == 'exponential'
                          else stats.lognorm.pdf(mean1, s=params1.get('sigma', 1.0), scale=np.exp(params1.get('mu', 0.0)))],
                color='C0', marker='o', zorder=5, label=f"{service_dist1.capitalize()} Mean")

    if x2 is not None:
        plt.plot(x2, y2, label=f"{service_dist2.capitalize()} {params2}")
        plt.scatter([mean2], [stats.expon.pdf(mean2, scale=1/params2.get('rate', 1.0)) if service_dist2 == 'exponential'
                              else stats.lognorm.pdf(mean2, s=params2.get('sigma', 1.0), scale=np.exp(params2.get('mu', 0.0)))],
                    color='C1', marker='o', zorder=5, label=f"{service_dist2.capitalize()} Mean")

    plt.xlim(x_range)
    plt.xlabel("Service Time")
    plt.ylabel("Probability Density")
    plt.title("Service Distribution(s)")
    plt.legend()
    plt.grid()
    plt.show()


# Plot various exponential distributions
def plot_exponential_distributions(lambdas, x_range=(0, 5), num_points=1000):
    """
    Plot multiple exponential distributions with different lambda (rate) values on the same axes.
    :param lambdas: list of lambda (rate) parameters for the exponential distributions
    :param x_range: tuple (min, max) for x-axis limits; if None, auto-calculate
    :param num_points: number of points to plot
    """
    plt.figure(figsize=(10, 6))
    
    for lam in lambdas:
        x = np.linspace(x_range[0], x_range[1], num_points)
        y = stats.expon.pdf(x, scale=1/lam)
        plt.plot(x, y, label=f"$\lambda$ = {lam}")


    plt.ylabel("Probability density")
    plt.title("Exponential densities")
    plt.legend()
    plt.show()


# Same thing for lognormal distributions
def plot_lognormal_distributions(mu_sigma_pairs, x_range=(0, 5),
                                    num_points=1000):
        """
        Plot multiple lognormal distributions with different mu and sigma parameters on the same axes.
        :param mu_sigma_pairs: list of tuples (mu, sigma) for the lognormal distributions
        :param x_range: tuple (min, max) for x-axis limits; if None, auto-calculate
        :param num_points: number of points to plot
        """
        plt.figure(figsize=(10, 6))
        
        for mu, sigma in mu_sigma_pairs:
            x = np.linspace(x_range[0], x_range[1], num_points)
            y = stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu))
            plt.plot(x, y, label=f"μ = {mu}, σ = {sigma}")
    
        plt.ylabel("Probability density")
        plt.title("Lognormal densities")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    # Example usage of the service distribution plotting function


    plot_lognormal_distributions(
        mu_sigma_pairs=[(1.5, 0.5), (1, 0.5), (1.0, 1.0)],
        x_range=(0, 15),
        num_points=1000
    )