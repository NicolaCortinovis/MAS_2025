import simpy as sp
from process_handling import arrival_process, make_servers
from plots import plot_gantt_chart
import statistics
import pandas as pd
import os
import glob
import random


def run_simulation(arrival_rate, sim_time, n_servers, strategy, save_results=True, show_plot=False, seed=15, service_dist = "exponential", **dist_parameters):
        """
        Run a scheduling simulation with the specified parameters. We track the following metrics:
        - Scheduling strategy used
        - Busy times of each server
        - Number of jobs completed by each server
        - Utilization of each server
        - Mean, standard deviation, and coefficient of variation of server utilization
        - Total number of jobs completed
        - CV Utilization (standard deviation / mean utilization) (It tracks the variability of server utilization)
        - Throughput (total jobs completed / simulation time)
        - Arrival rate 
        - Simulation time
        - Number of servers
        - Service distribution
        - Service distribution parameters (if applicable)
    

        Args:
            arrival_rate (float): The rate at which jobs arrive.
            sim_time (float): Total simulation time.
            n_servers (int): Number of servers to handle jobs.
            strategy (str): Scheduling strategy to use ('FIFO', 'LIFO', 'SJF', 'Preemptive LIFO').
            save_results (bool): Whether to save results to files.
            show_plot (bool): Whether to display the Gantt chart plot.
            seed (int): Random seed for reproducibility.
            service_dist (str): Distribution type for service times ('exponential', 'uniform', etc.).
            **dist_parameters: Additional parameters for the service distribution (e.g., rate for exponential, mu and sigma for lognormal).
        Returns:
            pd.DataFrame: DataFrame containing simulation metrics.
        """
        # Set seed
        random.seed(seed)

        # Setup
        intervals = []
        latencies = []
        remaining_times = [0.0] * n_servers
        busy_times = [0.0] * n_servers
        jobs_done = [0] * n_servers
        arrival_times_by_job = {}

        env = sp.Environment()
        servers = make_servers(env, n_servers, strategy)

        env.process(arrival_process(
            env=env,
            arrival_rate=arrival_rate,
            service_dist=service_dist,
            servers=servers,
            busy_times=busy_times,
            jobs_done=jobs_done,
            strategy=strategy,
            intervals=intervals,
            sim_time=sim_time,
            remaining_times=remaining_times,
            latencies=latencies,
            arrival_times_by_job=arrival_times_by_job,
            **dist_parameters
        ))

        env.run(until=sim_time)

        # Metrics
        utilizations = [bt / sim_time for bt in busy_times]
        mean_U = statistics.mean(utilizations)
        std_U = statistics.pstdev(utilizations)
        cv_U = std_U / mean_U if mean_U else float('nan')
        total_jobs = sum(jobs_done)
        throughput = total_jobs / sim_time
        mean_latencies = statistics.mean(latencies) if latencies else float('nan')

        metrics = pd.DataFrame({
            'Strategy': [strategy],
            'Busy Times': [busy_times],
            'Jobs Done': [jobs_done],
            'Utilizations': [utilizations],
            'Mean Util': [mean_U],
            'Std Dev Util': [std_U],
            'CV Util': [cv_U],
            'Total Jobs': [total_jobs],
            'Throughput': [throughput],
            'Mean Latencies': [mean_latencies],
            'Arrival Rate': [arrival_rate],
            'Sim Time': [sim_time],
            'N Servers': [n_servers],
            'Service Distribution': [service_dist],
            'Service Distribution Params': [dist_parameters],
        })

        if save_results:
            dir_path = f"Results/{strategy}"
            os.makedirs(dir_path, exist_ok=True)

            param_str = '_'.join(f"{k}{v}" for k, v in dist_parameters.items())
            fname_base = f"{strategy}_AR_{arrival_rate}_SR_{service_dist}_Params_{param_str}"
            metrics.to_csv(f"{dir_path}/metrics_{fname_base}.csv", index=False)
            plot_gantt_chart(
                intervals,
                n_servers,
                arrival_times_by_job = arrival_times_by_job,
                save_path=f"{dir_path}/gantt_{fname_base}.png",
                title=f"Gantt Chart of {strategy} Scheduling with {service_dist} Service with Params {param_str}",
                show=show_plot
            )

        return metrics



def evaluation_metrics_reports(results_root="Results"):
    """
    Generate summary metric reports for each strategy based on previously saved simulation CSVs.

    Creates 4 files:
        - Results/FIFO_summary.csv
        - Results/LIFO_summary.csv
        - Results/SJF_summary.csv
        - Results/Preemptive LIFO_summary.csv
    Each report contains:
        - Arrival Rate
        - Service Distribution
        - Service Distribution Params
        - Mean Latencies
        - Throughput
        - Utilizations
        - CV Util

    Args:
        results_root (str): Path to the base results directory where strategy folders are located.
    """
    strategies = ["FIFO", "LIFO", "SJF", "Preemptive LIFO"]

    for strategy in strategies:
        strategy_path = os.path.join(results_root, strategy)
        if not os.path.isdir(strategy_path):
            print(f"Directory not found: {strategy_path}")
            continue

        summary_rows = []
        for file_path in glob.glob(os.path.join(strategy_path, "metrics_*.csv")):
            df = pd.read_csv(file_path)

            if df.empty:
                continue

            row = {
                "Arrival Rate": df.loc[0, "Arrival Rate"],
                "Service Distribution": df.loc[0, "Service Distribution"],
                "Service Distribution Params": df.loc[0, "Service Distribution Params"],
                "Mean Latencies": df.loc[0, "Mean Latencies"],
                "Throughput": df.loc[0, "Throughput"],
                "CV Util": df.loc[0, "CV Util"],
                "Utilizations": df.loc[0, "Utilizations"],
            }

            summary_rows.append(row)

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            save_path = os.path.join(results_root, f"{strategy}_summary.csv")
            summary_df.to_csv(save_path, index=False)
            print(f"Saved summary to {save_path}")
        else:
            print(f"No valid CSV files found for strategy {strategy}")




if __name__ == "__main__":
    
    # Service distribution parameters

    random.seed(15)  # Set seed for reproducibility

    mu = 0.0     # Mean for lognormal distribution
    sigma = 0.5  # Standard deviation for lognormal distribution
    rho = 1.0    # Rate for exponential distribution

    simulation_times = 25

    lambdas = [1.5, 3.5]  # Low load and high load arrival 
    strategies = ["FIFO", "LIFO", "SJF", "Preemptive LIFO"]
    service_dists = [
        ("lognormal", {"mu": mu, "sigma": sigma}),
        ("exponential", {"rate": rho}),
    ]

    for arrival_rate in lambdas:
        for strategy in strategies:
            for service_dist, dist_params in service_dists:

                # Run the simulation
                metrics = run_simulation(
                    arrival_rate=arrival_rate,
                    sim_time= simulation_times,  # Total simulation time
                    n_servers=3,    # Number of servers
                    strategy=strategy,
                    service_dist=service_dist,
                    save_results=True,
                    show_plot=False,  # Set to True if you want to see the plot
                    **dist_params
                )


    evaluation_metrics_reports(results_root="Results")