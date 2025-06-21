import simpy as sp
from process_handling import arrival_process, make_servers
from plots import plot_gantt_chart
import statistics
import pandas as pd
import os
import argparse



if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run a scheduling simulation with specified parameters.")
    parser.add_argument('--ar', type=float, default=0.4, help='Arrival rate of jobs (default: 0.4)')
    parser.add_argument('--sr', type=float, default=0.2, help='Service rate of jobs (default: 0.2)')
    parser.add_argument('--sim_time', type=float, default=50, help='Total simulation time (default: 50)')
    parser.add_argument('--n_proc', type=int, default=3, help='Number of servers (default: 3)')
    parser.add_argument('--strat', type=str, choices=['FIFO', 'LIFO', 'SJF'], default='FIFO',
                        help='Scheduling strategy to use (default: FIFO)')
    args = parser.parse_args()

    # Extract parameters from parsed arguments

    arrival_rate = parser.parse_args().ar
    service_rate = parser.parse_args().sr
    sim_time     = parser.parse_args().sim_time
    n_servers    = parser.parse_args().n_proc
    strategy     = parser.parse_args().strat


    intervals = []
    # Needed for metrics
    busy_times = [0.0]*n_servers    # total busy time
    jobs_done  = [0]*n_servers      # job counts

    env     = sp.Environment()
    servers = make_servers(env, n_servers, strategy)

    # start arrivals
    env.process(arrival_process(env = env,
                                arrival_rate = arrival_rate,
                                service_rate = service_rate,
                                servers = servers,
                                busy_times = busy_times,
                                jobs_done = jobs_done,
                                strategy = strategy,
                                intervals = intervals))

    env.run(until=sim_time)

    # ------ metrics computation ------
    utilizations = [bt/sim_time for bt in busy_times]
    mean_U       = statistics.mean(utilizations)
    std_U        = statistics.pstdev(utilizations)
    cv_U         = std_U/mean_U if mean_U else float('nan')
    total_jobs   = sum(jobs_done)
    throughput   = total_jobs/sim_time


    # Create a DataFrame for better visualization with all the metrics
    metrics = pd.DataFrame({
        'Strategy': [strategy],
        'Busy Times': [busy_times],
        'Jobs Done': [jobs_done],
        'Utilizations': [utilizations],
        'Mean Util': [mean_U],
        'Std Dev Util': [std_U],
        'CV Util': [cv_U],
        'Total Jobs': [total_jobs],
        'Throughput': [throughput]
    })


    # print(f"Strategy:               {strategy}")
    # print(f"Per-server busy times:  {busy_times}")
    # print(f"Per-server job counts:  {jobs_done}")
    # print(f"Utilizations:           {[f'{u:.3f}' for u in utilizations]}")
    # print(f"Mean util:              {mean_U:.3f}")
    # print(f"Std Dev util:           {std_U:.3f}")
    # print(f"CV util:                {cv_U:.3f}")
    # print(f"Total jobs processed:   {total_jobs}")
    # print(f"Throughput (jobs/time): {throughput:.3f}")

    # If the directory does not exist, create it
    if not os.path.exists(f"Results/{strategy}"):
        os.makedirs(f"Results/{strategy}")

    
    metrics.to_csv(f"Results/{strategy}/metrics_{strategy}_AR_{arrival_rate}_SR_{service_rate}.csv", index=False)
    plot_gantt_chart(intervals, n_servers, save_path=f"Results/{strategy}/gantt_{strategy}_AR_{arrival_rate}_SR_{service_rate}.png", title= f"Gantt Chart of {strategy} Scheduling", show=True)