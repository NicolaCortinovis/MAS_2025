# This file contains the functions and class needed for the belief-based joing action learning
# and its evaluation. We're going to see two types of agents
# 1) A belief-based joint action learning agent that learns from self-play and random opponents.
# 2) A random agent that acts randomly in the environment.

import numpy as np
from football_env import ACT, SimpleFootballGame, NUM_STATES
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
        # Using this approach we'll have a 3D array where some entries will always be 0, specifically those
        # that correspond to illegal actions. E.g. When an agent is in the top left corner (0,0) N is illegal so it
        # will never be picked and updated, but this is not a problem and doesn't conflict with
        # the convergence requirements as such actions are useless if not detremental in the grand context of things.
        # Its probably faster to work with a np  instead of using a dict or similar structure. 
        # A better choice would be to create something "ad-hoc" but I suspect the gains
        # to be marginal and the code would be more complex.

        # Initialize counts for each state-action pair, all to 1 as a weak-informative prior
        # Handling of illegal actions is done inside  get_belief
        self.counts = np.ones((n_states, self.n_actions))


      

    def get_belief(self, state, legal_oppo_actions):
        """
        Compute the belief distribution over opponent actions in 'state',
        putting zero weight on illegal moves and renormalizing.
        :param state: Current state of the environment.
        :param legal_oppo_actions: List of legal actions available for the opponent in the current
        :return: Belief distribution over opponent actions.
        """


        # Mask the counts for illegal actions
        c = self.counts[state].copy()
        mask = np.zeros_like(c)
        for a in legal_oppo_actions:
            mask[self.actions.index(a)] = 1
        c_masked = c * mask

        # Renormalize the counts to sum to 1
        total = c_masked.sum()

        assert total > 0, "Total counts must be greater than zero to avoid division by zero."
        
        # Compute the belief distribution
        belief = c_masked / total

        return belief


    def select_action(self, state, legal_actions, legal_oppo_actions):
        """
        Select an action based on the eps-greedy policy using the belief distribution.
        :param state: Current state of the environment.
        :param legal_actions: List of legal actions available in the current state.
        :param legal_oppo_actions: List of legal actions available for the opponent.
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
        """
        Update the agent's Q-values and counts based on the observed transition.
        :param state: Current state of the environment.
        :param my_act: Action taken by the agent.
        :param opp_act: Action taken by the opponent.
        :param legal_oppo_actions: List of legal actions available for the opponent in the next
        :param reward: Reward received for the action taken.
        :param alpha: Learning rate for the update.
        :param next_state: Next state of the environment after taking the action.
        :param done: Boolean indicating if the episode has ended.
        """
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
        # update counts of the belief distribution
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


def set_agent_to_greedy(agent):
    """
    Set the agent's exploration rate to 0, making it act greedily. Used for evaluation.
    :param agent: The agent instance to modify.
    """
    agent.epsilon = 0.0


def run_episode_train(env, agent_A, agent_B, alpha, alpha_decay, mode="random"):
    """
    Run a single episode of the game.
    :param env: The environment instance.
    :param agent_A: The agent for player A.
    :param agent_B: The agent for player B (only used in self-play mode).
    :param alpha: Learning rate for the agents.
    :param alpha_decay: Decay factor for the learning rate.
    :param mode: Mode of operation, either "random" or "selfplay".
    :return: Number of steps taken in the episode and the updated alpha value.
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


        alpha *= alpha_decay # Decay alpha after each step

        steps += 1
        state = next_state


    return steps, alpha



def train_for(env, agent_A, agent_B, mode, total_steps=1_000_000, alpha = ALPHA, alpha_decay=ALPHA_DECAY):
    """
    Train the agents for a specified number of steps.
    :param env: The environment instance.
    :param agent_A: The agent for player A.
    :param agent_B: The agent for player B (only used in self-play mode).
    :param mode: Mode of operation, either "random" or "selfplay".
    :param total_steps: Total number of steps to train the agents.
    :param alpha: Initial learning rate for the agents.
    :param alpha_decay: Decay factor for the learning rate.
    """
    assert mode in ["random", "selfplay"]
    steps = 0
    episodes = 0
    while steps < total_steps:
        ep_len, alpha_new = run_episode_train(env, agent_A, agent_B, alpha=alpha, alpha_decay = alpha_decay, mode=mode)
        alpha = alpha_new
        steps   += ep_len
        episodes+= 1


def run_episode_test(env, agent_A, agent_B, mode="random", gamma=0.9):
    """
    Run one full episode in test mode.
    :param env: The environment instance.
    :param agent_A: The agent for player A.
    :param agent_B: The agent for player B (only used in self-play mode).
    :param mode: Mode of operation, either "random" or "selfplay" or "random vs selfplay".
    :param gamma: Force draw parameter, 1 implies no forced draws.
    :return: Length of the match, reward for player A, and reward for player B.
    """
    state = env.reset()
    done = False
    match_length = 0

    assert mode in ["random", "selfplay", "random vs selfplay"], "Mode must be either 'random' or 'selfplay' or 'random vs selfplay'"

    while not done:
        legal_A = env.legal_actions("A")
        legal_B = env.legal_actions("B")

        if mode == "random":
            aA = agent_A.select_action(state, legal_A, legal_B)
            aB = random_policy(legal_B)
        elif mode == "selfplay":
            aA = agent_A.select_action(state, legal_A, legal_B)
            aB = agent_B.select_action(state, legal_B, legal_A)
        else:
            aA = random_policy(legal_A)
            aB = agent_B.select_action(state, legal_B, legal_A)

        state, rA, rB, done = env.step(aA, aB)
        match_length += 1

        # maybe force a draw
        if gamma != 1 and not done and np.random.rand() > gamma:
            done = True
            rA = rB = 0

    return match_length, rA, rB

def evaluate(agent_A, agent_B, env, n_steps=100000, gamma=0.9, mode="random"):
    """
    Play n_episodes, return aggregated stats.
    :param agent_A: The agent for player A.
    :param agent_B: The agent for player B (only used in self-play mode).
    :param env: The environment instance.
    :param n_episodes: Number of episodes to run for evaluation.
    :param gamma: Force draw parameter, 1 implies no forced draws.
    :param mode: Mode of operation, either "random" or "selfplay" or "random vs selfplay".
    :return: Tuple containing:
        - Total number of finished games
        - Average number of games per 100,000 steps
        - Win percentage for agent A
        - Win percentage for agent B
        - Draw percentage
        - List of lengths of each match
        - Dictionary with match lengths categorized by outcome (A win, B win, draw)
    """

    assert gamma >= 0 and gamma <= 1, "Gamma must be between 0 and 1"
    assert mode in ["random", "selfplay", "random vs selfplay"], "Mode must be either 'random' or 'selfplay' or 'random vs selfplay'"

    if mode == "random":
        set_agent_to_greedy(agent_A)
    elif mode == "selfplay":
        set_agent_to_greedy(agent_A)
        set_agent_to_greedy(agent_B)
    else:
        set_agent_to_greedy(agent_B)


    wins_A = wins_B = draws = 0                     # Initialize counters for wins and draws
    lengths = []                                    # List to store lengths of each match
    by_outcome = {"A": [], "B": [], "draw": []}     # Dictionary to categorize match lengths by outcome
    steps = 0                                       # Counter for total steps taken in the evaluation

    while steps < n_steps:
        length, rA, rB = run_episode_test(env, agent_A, agent_B,mode=mode, gamma=gamma)
        lengths.append(length)
        steps += length
        if rA == 1:
            wins_A += 1
            by_outcome["A"].append(length)
        elif rB == 1:
            wins_B += 1
            by_outcome["B"].append(length)
        else:
            draws += 1
            by_outcome["draw"].append(length)

    finished = wins_A + wins_B + draws
    win_A_pct = 100 * wins_A / finished
    win_B_pct = 100 * wins_B / finished
    draw_pct  = 100 * draws   / finished
    games_per_100k = finished * (100_000 / sum(lengths))

    return finished, games_per_100k, win_A_pct, win_B_pct, draw_pct, lengths, by_outcome


def collect_episode_states(env, agent_A, agent_B=None, mode="random", gamma=0.9):
    """
    Collect the states of an episode for visualization or analysis.
    :param env: The environment instance.
    :param agent_A: The agent for player A.
    :param agent_B: The agent for player B (only used in self-play mode).
    :param mode: Mode of operation, either "random" or "selfplay" or "random vs selfplay".
    :param gamma: Force draw parameter, 1 implies no forced draws.
    :return: List of states recorded during the episode.
    """
    trajectory = []
    state = env.reset()
    done = False
    
    assert mode in ["random", "selfplay", "random vs selfplay"], "Mode must be either 'random' or 'selfplay' or 'random vs selfplay'"
    assert gamma >= 0 and gamma <= 1, "Gamma must be between 0 and 1"

    if mode == "random":
        set_agent_to_greedy(agent_A)
    elif mode == "selfplay":
        set_agent_to_greedy(agent_A)
        set_agent_to_greedy(agent_B)
    else:
        set_agent_to_greedy(agent_B)

    while not done:
        # record current positions & who has the ball
        trajectory.append(env.get_state())  # returns ((x_b,y_b),(x_a,y_a),ownership)


        legal_A = env.legal_actions("A")
        legal_B = env.legal_actions("B")
        
        if mode == "random":
            aA = agent_A.select_action(state, legal_A, legal_B)
            aB = random_policy(legal_B)
        elif mode == "selfplay":
            aA = agent_A.select_action(state, legal_A, legal_B)
            aB = agent_B.select_action(state, legal_B, legal_A)
        else:
            aA = random_policy(legal_A)
            aB = agent_B.select_action(state, legal_B, legal_A)

        state, _, _, done = env.step(aA, aB)

        # Force a draw
        if gamma != 1 and not done and np.random.rand() > gamma:
            done = True


    return trajectory


