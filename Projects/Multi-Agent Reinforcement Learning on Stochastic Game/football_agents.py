# This file contains the functions and class needed for the belief-based joing action learning
# algorithm

import numpy as np
from football_env import ACT, SimpleFootballGame, SimpleFootballGameClamp, NUM_STATES
import random as rand
from tqdm import tqdm


EXP_RATE = 0.2
GAMMA = 0.9 
ALPHA = 1
ALPHA_DECAY = 0.9999954 



class BeliefAgent:
    def __init__(self, n_states, actions, env, epsilon=0.2, gamma=0.9):
        """
        Initialize the belief-based joint action learning agent.
        :param n_states: Number of states in the environment.
        :param actions: List of possible actions.
        :param env: The environment instance (used for legal actions).
        :param epsilon: Exploration rate for ε-greedy action selection.
        :param gamma: Discount factor for future rewards.
        :param alpha0: Initial learning rate.
        :param alpha_decay: Decay factor for the learning rate.
        """
        self.n_states = n_states
        self.actions = actions
        self.env = env  
        self.epsilon = epsilon
        self.gamma = gamma
        self.n_actions = len(actions)

        # Q-values: shape [state, my_action, opp_action] (800 x 5 x 5)
        self.Q = np.zeros((n_states, self.n_actions, self.n_actions))

        # Note on self.Q
        # This way we have a 3D array where some entries will always be 0, specifically those
        # that correspond to illegal actions. E.g. top left corner (0,0) N is illegal so it
        # will never be picked and updated, but this is not a problem and doens't conflict with 
        # the convergence requirements as such actions are. Anyways its probably faster to work
        # like this instead of using a dict or similar structure. Better choice would be to create 
        # something "ad-hoc" but the gains are minimal and this is easier to read.

        # Initialize counts for each state-action pair, all to 1 as a  weak-informative prior
        # Handling of illegal actions is done inside  get_belief
        self.counts = np.ones((n_states, self.n_actions))


      

    def get_belief(self, state, legal_oppo_actions):
        """
        Compute the belief distribution over opponent actions in 'state',
        putting zero weight on illegal moves and renormalizing.
        """
       
    

        # Mask the counts for illegal actions
        c = self.counts[state].copy()
        mask = np.zeros_like(c)
        for a in legal_oppo_actions:
            mask[self.actions.index(a)] = 1
        c_masked = c * mask

        # Renormalize the counts to sum to 1
        total = c_masked.sum()

        try:
            if total == 0:
                raise ValueError(f"Total counts for state {state} is zero, cannot compute belief distribution.")
        except ZeroDivisionError:
            # If total is zero, it means no legal actions were available, return uniform belief
            belief = np.ones(len(self.actions)) / len(self.actions)
            return belief

        belief = c_masked / total

        return belief


    def select_action(self, state, legal_actions, legal_oppo_actions):
        """
        Select an action based on the eps-greedy policy using the belief distribution.
        :param state: Current state of the environment.
        :param legal_actions: List of legal actions available in the current state.
        :return: Selected action.
        """
        # eps–greedy
        if rand.random() < self.epsilon:
            return rand.choice(legal_actions)

        belief = self.get_belief(state, legal_oppo_actions)
        best_val, best_act = -float('inf'), None

        for a in legal_actions:
            i = self.actions.index(a)
            # Compute the value of the action under the belief distribution
            val = np.dot(self.Q[state, i, :], belief)
            if val > best_val:
                best_val, best_act = val, a

        return best_act



    def update(self, state, my_act, opp_act, legal_oppo_actions, reward, alpha, next_state, done):
        i = self.actions.index(my_act)
        j = self.actions.index(opp_act)
        # Q-learning update over joint action:
        # form max over my actions of expected Q in next state
        belief_next = self.get_belief(next_state, legal_oppo_actions)
        # value of next state under current policy
        expected = self.Q[next_state].dot(belief_next)  # shape (n_my_actions,)
        future_val = expected.max()                     # scalar
        target = reward + (0 if done else self.gamma * future_val)
        td = target - self.Q[state, i, j]
        self.Q[state, i, j] += alpha * td
        # update counts and decay alpha
        self.counts[state, j] += 1


def random_policy(legal_actions):
    """
    Select a random action from the legal actions available
    :param legal_actions: List of legal actions available in the current state (computed by the environment).
    :return: Randomly selected action from the legal actions.
    """
    if not legal_actions:
        raise ValueError("No legal actions available.") # Wont happen but still good to check
    return rand.choice(legal_actions)


def run_episode(env, agent_A, agent_B, alpha, max_steps=10000, mode="random"):
    """
    Run a single episode of the game.
    :param env: The environment instance.
    :param agent_A: The agent for player A.
    :param agent_B: The agent for player B (only used in self-play mode).
    :param max_steps: Maximum number of steps in the episode.
    :param mode: Mode of operation, either "random" or "selfplay".
    :return: Number of steps taken in the episode.
    """
    assert mode in ["random", "selfplay"]

    state = env.reset()
    done = False
    steps = 0
    

    while not done:
        # Legal actions for both players
        legal_A = env.legal_actions("A")
        legal_B = env.legal_actions("B")

        # Pick actions based on the mode
        if mode == "random":
            # Random policy for player B, agent A acts normally using its policy
            aA = agent_A.select_action(state, legal_A, legal_B)
            aB = random_policy(legal_B)
        else:  
            # Both agents select actions using their policies (both are BeliefAgents)
            aA = agent_A.select_action(state, legal_A, legal_B)
            aB = agent_B.select_action(state, legal_B, legal_A)

        # Update the environment with the selected actions
        next_state, rA, rB, done = env.step(aA, aB)


        next_legal_A = env.legal_actions("A")
        next_legal_B = env.legal_actions("B")


        # Update agents based on the actions taken and rewards received
        if mode == "random":
            agent_A.update(state, aA, aB, next_legal_B, rA, alpha, next_state, done)
        else:
            agent_A.update(state, aA, aB, next_legal_B, rA, alpha, next_state, done)
            agent_B.update(state, aB, aA, next_legal_A, rB, alpha, next_state, done)


        alpha *= ALPHA_DECAY  # Decay alpha after each step

        steps += 1
        state = next_state


    return steps, alpha



def train_for(env, agent_A, agent_B, mode, total_steps=1_000_000, alpha = ALPHA):
    """
    Train the agents for a specified number of steps.
    :param env: The environment instance.
    :param agent_A: The agent for player A.
    :param agent_B: The agent for player B (only used in self-play mode).
    :param mode: Mode of operation, either "random" or "selfplay".
    :param total_steps: Total number of steps to train the agents.
    """
    assert mode in ["random", "selfplay"]
    steps = 0
    episodes = 0
    while steps < total_steps:
        ep_len, alpha_new = run_episode(env, agent_A, agent_B, alpha=alpha, max_steps=10_000, mode=mode)
        alpha = alpha_new
        steps   += ep_len
        episodes+= 1
    print(f"Trained {mode:9s} in ~{steps} steps over {episodes} episodes")


def evaluate(agent_a, agent_b, env, n_steps=100_000, gamma=0.9, mode = "random"):
    """
    Play n_steps against random opponent, return (games_finished, games_per_100k, win_pct).
    :param agent: The agent to evaluate.
    :param env: The environment instance.
    :param n_steps: Total number of steps to evaluate the agent.
    :param gamma: Probability of not forcing a draw at each step.
    :return: Tuple containing the number of games finished, games per 100k steps,
    """
    steps = 0
    finished = 0
    wins_A = 0
    wins_B = 0
    draws = 0
    match_lengths = []
    match_length_given_outcome = {"A": [], "B": [], "draw": []}

     # Set epsilon to 0 for evaluation
    agent_a.epsilon = 0.0
    if agent_b is not None:
        agent_b.epsilon = 0.0

    assert mode in ["random", "selfplay"]

    while steps < n_steps:
        match_duration = 0
        state = env.reset()
        done = False
        # we'll count this game only if done==True before hitting a step cap
        while not done and steps < n_steps:
            legal_A = env.legal_actions("A")
            legal_B = env.legal_actions("B")
            


            if mode == "selfplay":
                # Both agents select actions using their policies
                aA = agent_a.select_action(state, legal_A, legal_B)
                aB = agent_b.select_action(state, legal_B, legal_A)
            else:
                # Random policy for player B, agent A acts normally using its policy
                aA = agent_a.select_action(state, legal_A, legal_B)
                aB = random_policy(legal_B)
            # Update the environment with the selected actions
            state, rA, rB, done = env.step(aA, aB)
            steps += 1
            match_duration += 1

            

            # Update the agent based on the actions taken and rewards received
            if not done and np.random.rand() > gamma:
                # force a draw
                done = True
                rA = rB = 0

        if done:
            finished += 1
            match_lengths.append(match_duration)
            if rA == 1:
                wins_A += 1
                match_length_given_outcome["A"].append(match_duration)  
            if rB == 1:
                wins_B += 1
                match_length_given_outcome["B"].append(match_duration)
            if rA == 0 and rB == 0:
                draws += 1
                match_length_given_outcome["draw"].append(match_duration)

    win_A_pct = 100.0 * wins_A / finished if finished > 0 else 0.0
    win_B_pct = 100.0 * wins_B / finished if finished > 0 else 0.0
    draw_pct = 100.0 * draws / finished if finished > 0 else 0.0
    games_per_100k = finished * (100_000 / steps)
    return finished, games_per_100k, win_A_pct, win_B_pct, draw_pct, match_lengths, match_length_given_outcome

if __name__ == "__main__":
    env = SimpleFootballGame()

    times = 10

    for i in tqdm(range(times)):

        # --- 1) Train A vs random B ---
        agent_A = BeliefAgent(n_states=NUM_STATES, actions=sorted(ACT), env=env)
        train_for(env, agent_A, agent_B = None, mode="random", total_steps=1_000_000)

        # Evaluate
        finished, games_per_100k, win_A_pct, win_B_pct, draw_pct, match_lengths, match_length_given_outcome = evaluate(agent_A, None, env, n_steps=100_000, mode="random")
        avg_length = np.mean(match_lengths) if match_lengths else 0
        print(f"vs Random: {win_A_pct:.1f}% wins, {win_B_pct:.1f}% losses, {draw_pct:.1f}% draws, {games_per_100k:.0f} games/100k steps, {avg_length:.1f} avg length per game")
        # Average match lengths given outcome
        for outcome, lengths in match_length_given_outcome.items():
            avg_length = np.mean(lengths) if lengths else 0
            print(f"  {outcome}: {avg_length:.1f}")

                # Write the results in a .txt file
        with open("results_random.txt", "a") as f:
            f.write(f"Run {i+1}:\n")
            f.write(f"vs Random: {win_A_pct:.1f}% wins, {win_B_pct:.1f}% losses, {draw_pct:.1f}% draws, {games_per_100k:.0f} games/100k steps, {avg_length:.1f} avg length per game\n")
            for outcome, lengths in match_length_given_outcome.items():
                avg_length = np.mean(lengths) if lengths else 0
                f.write(f"  {outcome}: {avg_length:.1f}\n")
            f.write("\n")
    
    for i in tqdm(range(times)):

        # # --- 2) Train A & B in self-play ---
        agent_A = BeliefAgent(n_states=NUM_STATES, actions=sorted(ACT), env=env)
        agent_B = BeliefAgent(n_states=NUM_STATES, actions=sorted(ACT), env=env)
        train_for(env, agent_A, agent_B, mode="selfplay", total_steps=1_000_000)

        # Evaluate
        finished, games_per_100k, win_A_pct, win_B_pct, draw_pct, match_lengths, match_length_given_outcome = evaluate(agent_A, agent_B, env, n_steps=100_000, mode = "selfplay")
        avg_length = np.mean(match_lengths) if match_lengths else 0
        print(f"vs Self-play: {win_A_pct:.1f}% wins, {win_B_pct:.1f}% losses, {draw_pct:.1f}% draws, {games_per_100k:.0f} games/100k steps, {avg_length:.1f} avg length per game")
        for outcome, lengths in match_length_given_outcome.items():
            avg_length = np.mean(lengths) if lengths else 0
            print(f"  {outcome}: {avg_length:.1f}")

        # Save results to a file
        with open("results_selfplay.txt", "a") as f:
            f.write(f"Run {i+1}:\n")
            f.write(f"vs Self-play: {win_A_pct:.1f}% wins, {win_B_pct:.1f}% losses, {draw_pct:.1f}% draws, {games_per_100k:.0f} games/100k steps, {avg_length:.1f} avg length per game\n")
            for outcome, lengths in match_length_given_outcome.items():
                avg_length = np.mean(lengths) if lengths else 0
                f.write(f"  {outcome}: {avg_length:.1f}\n")
            f.write("\n")

