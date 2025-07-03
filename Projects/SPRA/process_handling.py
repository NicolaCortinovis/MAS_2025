import random
import statistics
import simpy as sp



def make_servers(env, n_servers, strategy):
    """
    Create n_servers, using the correct Resource type for each strategy.
    
    Args:
        env: the simulation environment
        n_servers: number of servers to create
        strategy: scheduling discipline, one of:
            - 'FIFO'
            - 'LIFO'
            - 'SJF'
            - 'Preemptive LIFO'
    """
    if strategy == "FIFO":
        # plain first-in-first-out, non-preemptive
        return [sp.Resource(env, capacity=1) for _ in range(n_servers)]

    elif strategy == "LIFO":
        # non-preemptive last-in-first-out via priorities
        return [sp.PriorityResource(env, capacity=1) for _ in range(n_servers)]

    elif strategy == "SJF":
        # non-preemptive shortest-job-first via priorities
        return [sp.PriorityResource(env, capacity=1) for _ in range(n_servers)]

    elif strategy == "Preemptive LIFO":
        # preemptive LIFO: higher-priority means newer jobs can interrupt
        return [sp.PreemptiveResource(env, capacity=1) for _ in range(n_servers)]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def job(env, name, service_time, servers, busy_times, jobs_done,
        strategy, seq, intervals, sim_time, remaining_times, arrival_time, latencies):
    """
    A job process that requests a server, waits if necessary, runs for its service time,
    handles preemption if the strategy allows it, and records metrics for visualization.

    Args:
        env: SimPy simulation environment.
        name: Unique job name for logging.
        service_time: Total time this job needs to be processed.
        servers: List of SimPy server resources (one per processor).
        busy_times: List that tracks total busy time for each server.
        jobs_done: List that counts the number of completed jobs per server.
        strategy: Scheduling policy ('FIFO', 'LIFO', 'SJF', or 'Preemptive LIFO').
        seq: Sequence number for tie-breaking in LIFO strategies.
        intervals: List of tuples used to plot Gantt charts.
        sim_time: Total simulation time (used to clip visualization).
        remaining_times: List to track remaining service times for jobs in progress
        arrival_time: Time when the job arrived in the system (for latency calculation).
        latencies: List to track latencies of completed jobs (end time - arrival time).
        

    Example:
        - A job arrives and requests a server (according to the chosen strategy).
        - If the server is busy, it waits in the queue.
        - Once it gets access, it runs for its service time.
            - If the strategy is preemptive, it can be interrupted by higher-priority jobs.
            - If interrupted, it records how much time it ran before being preempted.
            - It then requeues itself to resume later.
        - The job's start time and end time are recorded for Gantt chart visualization.
        - The server's busy time and completed job count are updated accordingly.
    """

    # Choose the least-loaded server based on the current workload

    # Compute estimated workload on each server
    workloads = []
    for i, srv in enumerate(servers):
        # Work remaining in the queue (from service_time we attach to each request)
        queue_work = sum(getattr(req, 'service_time', 0.0) for req in srv.queue)
        
        # Work remaining in the current job
        running_work = remaining_times[i]
        workloads.append(queue_work + running_work)

    # Pick the server with the least total remaining work
    idx = workloads.index(min(workloads))
    server = servers[idx]

    # Debugging output
    # print(f"t={env.now:.2f}, workloads={workloads} → idx={idx} "
    #      f"with service_time={service_time:.2f}, seq={seq}, strategy={strategy}")

    # Create the correct type of request based on the strategy

    if strategy == "FIFO":
        # No priority handling, jobs are served in arrival order
        req = server.request()

    elif strategy == "LIFO":
        # Newer jobs have higher priority (smaller -seq value)
        req = server.request(priority=-seq)

    elif strategy == "Preemptive LIFO":
        # Same as LIFO but allows preemption of running jobs
        req = server.request(priority=-seq, preempt=True)

    else:
        # Shorter jobs have higher priority
        req = server.request(priority=service_time)

    req.service_time = service_time

    # Wait for and hold the server until the job is done or preempted 
    with req:
        yield req  # Wait for access to the resource

        # Record when the job actually starts
        start = env.now
        remaining = service_time
        remaining_times[idx] = remaining

        # For visualization: record the job bar in the Gantt chart
        end_for_plot = min(start + service_time, sim_time)  # Clip to simulation end
        intervals.append((idx, start, end_for_plot, name))  # (server_id, start, end, label)
        
        try:
            yield env.timeout(remaining)
            # Job completed
            busy_times[idx] += service_time
            jobs_done[idx] += 1
            remaining_times[idx] = 0.0  # Clear running work
            completion_time = env.now
            latencies.append(completion_time - arrival_time)
        except sp.Interrupt:
            worked = env.now - start
            busy_times[idx] += worked
            remaining -= worked
            remaining_times[idx] = remaining  # Update tracked remaining time
            env.process(job(env, name, remaining, servers,
                            busy_times, jobs_done, strategy, seq,
                            intervals, sim_time, remaining_times, arrival_time, latencies))  # Requeue job



def arrival_process(env, arrival_rate, service_dist,
                    servers, busy_times, jobs_done, strategy, intervals, sim_time,
                    remaining_times, latencies, arrival_times_by_job, **dist_params):    
    """
    Generates and launches jobs

    Args:
        env: SimPy simulation environment.
        arrival_rate: Average number of job arrivals per time unit (λ).
        service_dist: Distribution of service times ('exponential' or 'lognormal').
        servers: List of server resources available for job assignment.
        busy_times: List used to accumulate total busy time per server.
        jobs_done: List used to count the number of jobs completed by each server.
        strategy: Job scheduling strategy ('FIFO', 'LIFO', 'SJF', 'Preemptive LIFO').
        intervals: List of (server_id, start, end, job_name) for Gantt chart visualization.
        sim_time: Total simulation time (used for clipping visuals).
        remaining_times: List to track remaining service times for jobs in progress (used for SJF).
        latencies: List to track latencies of completed jobs (end time - arrival time).
        arrival_times_by_job: Dictionary to track arrival times of jobs by their unique IDs.
        dist_params: Parameters for the service distribution (rate for exponential, mu and sigma for lognormal).
    """

    i = 0  # Job sequence counter for unique job IDs and tie-breaking

    # Ensure the strategy provided is one of the supported types
    assert strategy in ["FIFO", "LIFO", "SJF", "Preemptive LIFO"], \
        f"Invalid strategy: {strategy}. Choose from 'FIFO', 'LIFO', 'Preemptive LIFO', 'SJF'."

    # Generate jobs (simulation duration is handled by the environment externally)
    while True:

        # Sample time until next job arrives 
        interarrival = random.expovariate(arrival_rate)  # Exponential interarrival (Poisson process)
        yield env.timeout(interarrival)                  # Wait for that time to pass in the simulation

        # Sample the service time required for the new job

        if service_dist == "exponential":
            rate = dist_params.get('rate', 1.0)        # Rate parameter for exponential distribution 
            service_time = random.expovariate(rate)     # Exponentially distributed job duration
        elif service_dist == "lognormal":
            mu = dist_params.get('mu', 0.0)        # Mean of the log-normal distribution
            sigma = dist_params.get('sigma', 1.0)  # Standard deviation of the log-normal distribution
            service_time = random.lognormvariate(mu, sigma)  # Log-normal job duration
        else:
            raise ValueError(f"Unknown service distribution: {service_dist}. "
                             "Use 'exponential' or 'lognormal'.")

        # Create and schedule a new job process in the environment
        arrival_time = env.now
        arrival_times_by_job[f"Job-{i}"] = arrival_time
        env.process(job(env,
                        f"Job-{i}",               # Unique job name
                        service_time,             # Time needed to complete this job
                        servers,                  # List of all server resources
                        busy_times,               # Metric: time each server was busy
                        jobs_done,                # Metric: count of completed jobs per server
                        strategy,                 # Scheduling policy in use
                        seq=i,                    # Sequence number (used for priority)
                        intervals=intervals,      # Gantt chart recording
                        sim_time=sim_time,        # Max simulation time for visualization
                        remaining_times=remaining_times,  # Remaining service times for jobs in progress
                        latencies=latencies,              # Latencies of completed jobs (end_time - arrival_time)
                        arrival_time=arrival_time))       # Arrival time of the job

        i += 1  # Increment the job ID for the next arriving job
