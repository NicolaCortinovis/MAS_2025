import random
import statistics
import simpy as sp



def make_servers(env, n_servers, strategy):
    """
    Create n_servers, using FIFO or PriorityResource based on strategy.
    FIFO: Resource(capacity=1)
    LIFO: PriorityResource(capacity=1), later use descending insertion order
    SJF : PriorityResource(capacity=1), priority = service_time
    Args:
        env: the simulation environment
        n_servers: number of servers to create
        strategy: scheduling discipline ('FIFO', 'LIFO', or 'SJF')
    """
    if strategy == "FIFO":
        return [sp.Resource(env, capacity=1) for _ in range(n_servers)]
    else:
        # PriorityResource supports a .request(priority=...) argument
        return [sp.PriorityResource(env, capacity=1) for _ in range(n_servers)]


def job(env, name, service_time, servers, busy_times, jobs_done, strategy, seq, intervals):
    """
    A job process that requests a resource, simulates service time,
    and logs metrics.
    Args:
        env: the simulation environment
        name: the name of the job (for logging)
        service_time: the time required to process this job (exponentially distributed)
        servers: a list of resources to be used by jobs
        busy_times: a list to record busy times for each resource
        jobs_done: a list to count jobs processed by each resource
        strategy: scheduling discipline ('FIFO', 'LIFO', or 'SJF')
        seq: sequence number of this job (for breaking ties in LIFO)
    """

    # --- pick the least-loaded server (queue+in-service) ---
    loads   = [len(p.queue) + p.count for p in servers]
    idx     = loads.index(min(loads))
    process = servers[idx]                # select the server

    # --- build the appropriate request for the chosen strategy ---
    if strategy == "FIFO":
        req = process.request()             # no priority, pure FIFO
    elif strategy == "LIFO":
        req = process.request(priority=-seq)  # newest jobs first
    else:
        req = process.request(priority=service_time)  # shortest jobs first
    
    with req:                               # request the resource
        yield req                           # wait until the resource is available

        start = env.now                     # log the start time of service
        yield env.timeout(service_time)     # simulate the service time
        end   = env.now                     # log the end time of service

        busy_times[idx] += end - start      # record the busy time for the server
        jobs_done[idx]  += 1                # increment the count of jobs done

        intervals.append((idx, start, end, name))


def arrival_process(env, arrival_rate, service_rate,
                    servers, busy_times, jobs_done, strategy, intervals):
    """
    Spawn new jobs at Poisson intervals, each with a pre-sampled service_time.
    Args:
        env: the simulation environment
        arrival_rate: the average rate of job arrivals (Poisson λ)
        service_rate: the average service rate (exponential μ)
        servers: a list of resources to be used by jobs
        busy_times: a list to record busy times for each resource
        jobs_done: a list to count jobs processed by each resource
        strategy: scheduling discipline ('FIFO', 'LIFO', or 'SJF')
    """
    i = 0

    assert strategy in ["FIFO", "LIFO", "SJF"], \
        f"Invalid strategy: {strategy}. Choose from 'FIFO', 'LIFO', or 'SJF'."
    

    while True:

        interarrival = random.expovariate(arrival_rate)  # exponential interarrival time
        yield env.timeout(interarrival)                  # wait for the next job to arrive
        service_time = random.expovariate(service_rate)  # pre-sample the service time

        env.process(job(env,
                        f"Job-{i}",
                        service_time,
                        servers,
                        busy_times,
                        jobs_done,
                        strategy,
                        seq=i,
                        intervals = intervals))                   # launch the job with its sequence number
        i += 1 
