# Train, Test and plot/animate the agents in a multi-agent reinforcement learning environment.

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
    # Create the environment

    env = SimpleFootballGame()
    agent_A = BeliefAgent(n_states = NUM_STATES, actions = sorted(ACT), env = env, epsilon = EXP_RATE, gamma = GAMMA)

    # Train the agents
    train_for(
        env = env,
        agent_A = agent_A,
        agent_B = None,
        mode = 'random',
        total_steps = TRAIN_STEPS,
        alpha = ALPHA,
        alpha_decay = ALPHA_DECAY
    )

   
    # Evaluate the agents
    finished, games_per_100k, win_A_pct, win_B_pct, draw_pct, match_lengths, match_length_given_outcome = evaluate(agent_a = agent_A,
                                                                                                                   agent_b =  None,
                                                                                                                   env = env,
                                                                                                                   n_steps= TEST_STEPS,
                                                                                                                   mode = "random",
                                                                                                                   gamma = GAMMA)
    
    print(f"Games per 100k: {games_per_100k}")
    print(f"Win A %: {win_A_pct}")
    print(f"Win B %: {win_B_pct}")
    print(f"Draw %: {draw_pct}")

    
    # Play some episodes
    for i in range(5):
        print(f"Episode {i+1}:")
        episode = collect_episode_states(env, agent_a = agent_A, agent_b = None, mode = "random", gamma = GAMMA)
        animate_episode(episode)