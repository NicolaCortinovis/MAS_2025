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


def job(env, name, service_time, servers, busy_times, jobs_done, strategy, seq, intervals, sim_time):
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
    print(f"t={env.now:.2f}, loads={loads} → idx={idx} with service_time={service_time:.2f}, seq={seq}, strategy={strategy}")
    server = servers[idx]                # select the server

    # --- build the appropriate request for the chosen strategy ---
    if strategy == "FIFO":
        req = server.request()             # no priority, pure FIFO
    elif strategy == "LIFO":
        req = server.request(priority=-seq)  # newest jobs first
    elif strategy == "Preemptive LIFO":
        req = server.request(priority=-seq, preempt=True)
    else:
        req = server.request(priority=service_time)  # shortest jobs first
    
    with req:
        yield req
        start        = env.now
        remaining    = service_time

        # record the bar for the Gantt (clamped to sim_time)
        end_for_plot = min(start + service_time, sim_time)
        intervals.append((idx, start, end_for_plot, name))

        # now service in a loop, handling preemptions
        try:
            # attempt to finish the whole service time
            yield env.timeout(remaining)
            # if we get here, we finished
            busy_times[idx] += service_time
            jobs_done[idx]  += 1
        except sp.Interrupt as interrupt:
            # compute how long we actually ran
            worked = env.now - start
            busy_times[idx] += worked
            remaining -= worked
            # optionally: requeue the remainder
            env.process(job(env, name, remaining, servers, 
                             busy_times, jobs_done, strategy, seq, intervals, sim_time))


def arrival_process(env, arrival_rate, service_rate,
                    servers, busy_times, jobs_done, strategy, intervals, sim_time):
    """
    Spawn new jobs at Poisson intervals, each with a pre-sampled service_time.
    Args:
        env: the simulation environment
        arrival_rate: the average rate of job arrivals (Poisson λ)
        service_rate: the average service rate (exponential μ)
        servers: a list of resources to be used by jobs
        busy_times: a list to record busy times for each resource
        jobs_done: a list to count jobs processed by each resource
        strategy: scheduling discipline ('FIFO', 'LIFO', 'Preemptive LIFO' or 'SJF')
    """
    i = 0

    assert strategy in ["FIFO", "LIFO", "SJF", "Preemptive LIFO"], \
        f"Invalid strategy: {strategy}. Choose from 'FIFO', 'LIFO', 'Preemptive LIFO',  'SJF'."
    

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
                        intervals = intervals,
                        sim_time=sim_time))                   # launch the job with its sequence number
        i += 1 
