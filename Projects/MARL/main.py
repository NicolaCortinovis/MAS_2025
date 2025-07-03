# This is the script that runs the training and the evaluation of the agents in the football environment.
# We're going to train two BeliefAgents, one versus a random agent and one versus a BeliefAgent.
# After that each agent will be evaluated versus a random agent and a BeliefAgent.
# We're also going to evaluate random vs BeliefAgent 
# We're going to track the following metrics:
# - Games per 100k steps
# - Win percentage for agent A
# - Win percentage for agent B
# - Draw percentage
# - Match lengths
# The two agents then will be saved to disk and can be loaded later for further evaluation or deployment.


import pandas as pd
import pickle
import os
from tqdm import tqdm

from football_agents import train_for, evaluate, BeliefAgent, collect_episode_states
from football_env import SimpleFootballGame, NUM_STATES, ACT
from grid_vis import animate_episode


EXP_RATE = 0.2
GAMMA = 0.9 
ALPHA = 1
ALPHA_DECAY = 0.9999954 
TRAIN_STEPS = 1_000_000
TEST_STEPS = 100_000


if __name__ == "__main__":
    
    all_summaries = []
    env = SimpleFootballGame()
    total_runs = 5

    for run_idx in tqdm(range(total_runs), desc="Training and evaluating Belief Agent vs Random Agent", unit="run"):

        agent_A = BeliefAgent(n_states = NUM_STATES, actions = sorted(ACT), env = env, epsilon = EXP_RATE, gamma = GAMMA)

        train_for(
            env = env,
            agent_A = agent_A,
            agent_B = None,
            mode = 'random',
            total_steps = TRAIN_STEPS,
            alpha = ALPHA,
            alpha_decay = ALPHA_DECAY
        )

    
        finished, games_per_100k, win_A_pct, win_B_pct, draw_pct, match_lengths, match_length_given_outcome = evaluate(agent_A = agent_A,
                                                                                                                    agent_B =  None,
                                                                                                                    env = env,
                                                                                                                    n_steps= TEST_STEPS,
                                                                                                                    mode = "random",
                                                                                                                    gamma = GAMMA)
    
        # Build summary for this run
        summary = {
            'run': run_idx,
            'finished_games': finished,
            'games_per_100k_steps': games_per_100k,
            'win_A_pct': win_A_pct,
            'win_B_pct': win_B_pct,
            'draw_pct': draw_pct,
            'avg_match_length': sum(match_lengths) / len(match_lengths),
            'std_match_length': pd.Series(match_lengths).std()
        }
        all_summaries.append(summary)


        with open(f'Results/Agents/BR_agents/belief_agent_A_run{run_idx}.pkl', 'wb') as f:
            pickle.dump(agent_A, f)

    
    df_summary = pd.DataFrame(all_summaries)
    df_summary.to_csv('Results/BTRvsR_summary.csv', index=False)

    all_summaries = []
    env.reset()

    for run_idx in tqdm(range(total_runs), desc="Training and evaluating Belief Agent vs Belief Agent", unit="run"):

        agent_A = BeliefAgent(n_states = NUM_STATES, actions = sorted(ACT), env = env, epsilon = EXP_RATE, gamma = GAMMA)
        agent_B = BeliefAgent(n_states = NUM_STATES, actions = sorted(ACT), env = env, epsilon = EXP_RATE, gamma = GAMMA)

        train_for(
            env = env,
            agent_A = agent_A,
            agent_B = agent_B,
            mode = 'selfplay',
            total_steps = TRAIN_STEPS,
            alpha = ALPHA,
            alpha_decay = ALPHA_DECAY
        )

    
        finished, games_per_100k, win_A_pct, win_B_pct, draw_pct, match_lengths, match_length_given_outcome = evaluate(agent_A = agent_A,
                                                                                                                    agent_B =  agent_B,
                                                                                                                    env = env,
                                                                                                                    n_steps= TEST_STEPS,
                                                                                                                    mode = "selfplay",
                                                                                                                    gamma = GAMMA)
    
        # Build summary for this run
        summary = {
            'run': run_idx,
            'finished_games': finished,
            'games_per_100k_steps': games_per_100k,
            'win_A_pct': win_A_pct,
            'win_B_pct': win_B_pct,
            'draw_pct': draw_pct,
            'avg_match_length': sum(match_lengths) / len(match_lengths),
            'std_match_length': pd.Series(match_lengths).std()
        }
        all_summaries.append(summary)

        with open(f'Results/Agents/BBA_agents/belief_agent_A_run{run_idx}.pkl', 'wb') as f:
            pickle.dump(agent_A, f)

        with open(f'Results/Agents/BBA_agents/belief_agent_B_run{run_idx}.pkl', 'wb') as f:
            pickle.dump(agent_B, f)

    
    df_summary = pd.DataFrame(all_summaries)
    df_summary.to_csv('Results/BBAvsBBA_summary.csv', index=False)

    all_summaries = []
    env.reset()

    for run_idx in tqdm(range(total_runs), desc="Evaluating Belief Agent trained on Random vs Belief Agent trained on Belief Agent", unit="run"):

        with open(f'Results/Agents/BR_agents/belief_agent_A_run{run_idx}.pkl', 'rb') as f:
            agent_A = pickle.load(f)

        with open(f'Results/Agents/BBA_agents/belief_agent_B_run{run_idx}.pkl', 'rb') as f:
            agent_B = pickle.load(f)

        finished, games_per_100k, win_A_pct, win_B_pct, draw_pct, match_lengths, match_length_given_outcome = evaluate(agent_A = agent_A,
                                                                                                                    agent_B =  agent_B,
                                                                                                                    env = env,
                                                                                                                    n_steps= TEST_STEPS,
                                                                                                                    mode = "selfplay",
                                                                                                                    gamma = GAMMA)
    
        # Build summary for this run
        summary = {
            'run': run_idx,
            'finished_games': finished,
            'games_per_100k_steps': games_per_100k,
            'win_A_pct': win_A_pct,
            'win_B_pct': win_B_pct,
            'draw_pct': draw_pct,
            'avg_match_length': sum(match_lengths) / len(match_lengths),
            'std_match_length': pd.Series(match_lengths).std()
        }
        all_summaries.append(summary)


    df_summary = pd.DataFrame(all_summaries)
    df_summary.to_csv('Results/BTRvsBTR_summary.csv', index=False)

    all_summaries = []
    env.reset()

    for run_idx in tqdm(range(total_runs), desc="Evaluating Belief Agent trained on Belief Agent vs random agent", unit="run"):
        
        with open(f'Results/Agents/BBA_agents/belief_agent_A_run{run_idx}.pkl', 'rb') as f:
            agent_A = pickle.load(f)

        finished, games_per_100k, win_A_pct, win_B_pct, draw_pct, match_lengths, match_length_given_outcome = evaluate(agent_A = agent_A,
                                                                                                                    agent_B =  None,
                                                                                                                    env = env,
                                                                                                                    n_steps= TEST_STEPS,
                                                                                                                    mode = "random",
                                                                                                                    gamma = GAMMA)
    
        # Build summary for this run
        summary = {
            'run': run_idx,
            'finished_games': finished,
            'games_per_100k_steps': games_per_100k,
            'win_A_pct': win_A_pct,
            'win_B_pct': win_B_pct,
            'draw_pct': draw_pct,
            'avg_match_length': sum(match_lengths) / len(match_lengths),
            'std_match_length': pd.Series(match_lengths).std()
        }
        all_summaries.append(summary)

    df_summary = pd.DataFrame(all_summaries)
    df_summary.to_csv('Results/BBAvsRandom_summary.csv', index=False)


    all_summaries = []
    env.reset()

    for run_idx in tqdm(range(total_runs), desc="Evaluating random Agent vs Belief Agent trained on Belief Agent", unit="run"):

        with open(f'Results/Agents/BBA_agents/belief_agent_B_run{run_idx}.pkl', 'rb') as f:
            agent_B = pickle.load(f)

        finished, games_per_100k, win_A_pct, win_B_pct, draw_pct, match_lengths, match_length_given_outcome = evaluate(agent_A = None,
                                                                                                                    agent_B =  agent_B,
                                                                                                                    env = env,
                                                                                                                    n_steps= TEST_STEPS,
                                                                                                                    mode = "random vs selfplay",
                                                                                                                    gamma = GAMMA)
    
        # Build summary for this run
        summary = {
            'run': run_idx,
            'finished_games': finished,
            'games_per_100k_steps': games_per_100k,
            'win_A_pct': win_A_pct,
            'win_B_pct': win_B_pct,
            'draw_pct': draw_pct,
            'avg_match_length': sum(match_lengths) / len(match_lengths),
            'std_match_length': pd.Series(match_lengths).std()
        }
        all_summaries.append(summary)

    df_summary = pd.DataFrame(all_summaries)
    df_summary.to_csv('Results/RandomvsBBA_summary.csv', index=False)




