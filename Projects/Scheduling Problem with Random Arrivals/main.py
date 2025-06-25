import simpy as sp
from process_handling import arrival_process, make_servers
from plots import plot_gantt_chart
import statistics
import pandas as pd
import os
import random
import argparse


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
        remaining_times = [0.0] * n_servers
        busy_times = [0.0] * n_servers
        jobs_done = [0] * n_servers

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
            remaining_times=remaining_times,  # 👈 Add this line
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
                save_path=f"{dir_path}/gantt_{fname_base}.png",
                title=f"Gantt Chart of {strategy} Scheduling",
                show=show_plot
            )

        return metrics




if __name__ == "__main__":


    run_simulation(
        arrival_rate=1,
        sim_time= 50,
        n_servers=3,
        strategy='FIFO',
        save_results=True,
        show_plot=True,
        service_dist='exponential',
        rate= rate
    )

    run_simulation(
        arrival_rate=1,
        sim_time= 50,
        n_servers=3,
        strategy='FIFO',
        save_results=True,
        show_plot=True,
        service_dist='exponential',
        rate= rate/2
    )

    run_simulation(
        arrival_rate=1,
        sim_time= 50,
        n_servers=3,
        strategy='FIFO',
        save_results=True,
        show_plot=True,
        service_dist='exponential',
        rate= rate*2
    )
