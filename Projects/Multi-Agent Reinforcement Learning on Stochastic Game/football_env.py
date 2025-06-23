# This file contains the class and functions for the football game environment described in the littman94 paper.

import numpy as np
import random as rand


ROW, COL = 4, 5                     # Size of the football field
N_CELLS = ROW * COL                 # Total number of cells in the field
ACT = {"N", "S", "E", "W", "H"}     # North, South, East, West, Halt (should be stand but for clearance halt)
A_STRT = (1,3)                      # Where the A agent starts
B_STRT = (2,1)                      # Where the B agent starts
A_WINS = [(1,0), (2,0)]             # Scoring positions for A
B_WINS = [(1,4), (2,4)]             # Scoring positions for B
DIFFS = {                           # Possible actions and their effects on the position  
    'N': (-1,  0),
    'S': ( 1,  0),
    'E': ( 0,  1),
    'W': ( 0, -1),
    'H': ( 0,  0)
}


class SimpleFootballGame:

    def __init__(self):
        self.step_count = 0


    def reset(self):
        """
        Reset the environment to its initial state.
        """
        self.A_pos = A_STRT
        self.B_pos = B_STRT
        self.step_count = 0
        self.ownership = rand.choice(["A", "B"])

        return self.state()


    def state(self):
        """
        Function to return the flattened state of the environment.
        Reasoning:
        Considering that each agent can be in (4 x 5) positions and the
        ownership can be either "A" or "B", the total number of states is:
        20 * 20 * 2 = 800 states. We flatten a state to its own unique index living
        inside the range [0, 799].
        """
        x_a, y_a = self.A_pos 
        x_b, y_b = self.B_pos

        index_a = x_a * COL + y_a  # Convert A's position 
        index_b = x_b * COL + y_b  # Convert B's position


        index = 2 * ((index_a * N_CELLS) + index_b) + (0 if self.ownership == "A" else 1)

        return index
    
    def get_state(self):
        """
        Get the current state of the environment and unpack it into a tuple.
        """
        
        index = self.state()
        ownership = "A" if index % 2 == 0 else "B"
        index //= 2

        x_b = index % COL
        index //= COL
        y_b = index % ROW
        index //= ROW

        x_a = index % COL
        y_a = index // COL

        return (x_a, y_a), (x_b, y_b), ownership
    
    def clamp(self, row, col):
        """Keep (row,col) on the board."""
        return (
            min(max(0, row), ROW-1),
            min(max(0, col), COL-1)
        )
    
    def legal_actions(self, player):
        """
        Get the legal actions for the given player.
        Args:
            player (str): 'A' or 'B' indicating which player's legal actions to return
        Returns:
            list: A list of legal actions for the player.
        """

        pos = self.A_pos if player=="A" else self.B_pos
        legal = []
        for act, (dr,dc) in DIFFS.items():
            new = self.clamp(pos[0] + dr, pos[1] + dc)
            # keep the halt action always; prune other moves
            # whose clamped new position == pos (i.e. they'd hit the wall)
            if act == 'H' or new != pos:
                legal.append(act)

        return legal
    
    def player_move(self, player, action):
        """ Move the player according to the action.
        Args:
            player (str): 'A' or 'B' indicating which player is moving.
            action (str): The action to take, one of 'N', 'S', 'E', 'W', 'H'.
        """
        
        # Effect of each action

        # Who is moving?
        if player == 'A':
            cur_pos = self.A_pos
            other_pos = self.B_pos
            pos_attr = 'A_pos'
            other = 'B'
        else:
            cur_pos = self.B_pos
            other_pos = self.A_pos
            pos_attr = 'B_pos'
            other = 'A'

        # Note that we never allow an action that would move off the board
        # so no need to clamp the position here.
        dr, dc = DIFFS.get(action, (0, 0))
        desired = (cur_pos[0] + dr, cur_pos[1] + dc)

        try:
            assert 0 <= desired[0] < ROW and 0 <= desired[1] < COL
        except AssertionError:
            raise ValueError(f"Invalid action {action} for player {player}. Desired position {desired} is out of bounds.")

        # If desired is occupied and it the action was not S, block & flip
        if desired == other_pos:
            self.ownership = other
        else: # Otherwise, move there and keep ownership
            setattr(self, pos_attr, desired)

        
    

    def step(self, action_a, action_b):
        """
        Take a step in the environment with the given actions for both players.
        Returns the new state, rewards for both players, and whether the game is done.
        Args:
            action_a (str): Action for player A.
            action_b (str): Action for player B.
        Returns:
            tuple: (new_state, reward_a, reward_b, done)
        """
        
        # Choose the first player to move randomly (uniform)
        player = rand.choice(["A", "B"])
        if player == "A":
            other_player = "B"
        else:
            other_player = "A"

        # Move the player then the other player
        self.player_move(player, action_a if player == "A" else action_b)
        self.player_move(other_player, action_b if player == "A" else action_a)

        # Increment step count
        self.step_count += 1

        # Did a player score?
        if self.A_pos in A_WINS and self.ownership == "A":
            return self.state(), 1, -1, True
        elif self.B_pos in B_WINS and self.ownership == "B":
            return self.state(), -1, 1, True
        else:
            return self.state(), 0, 0, False

    


if __name__ == "__main__":
    ############## DEBUGGER #################
    env = SimpleFootballGame()

    # Test state <-> get_state round-trip
    for _ in range(100):
        s0 = env.reset()
        posA, posB, owner = env.get_state()
        s1 = env.state()
        assert s0 == s1, f"Round-trip state failed: {s0} vs {s1}"
    print("State encoding/decoding consistent")

    # Test boundary moves never leave the board
    env.reset()
    env.A_pos = (0,0)
    env.B_pos = (3,4)
    for act in ['N','W']:
        try:
            env.player_move('A', act)
        except ValueError:
            pass  # If you kept the assert check, it'll raise; that's OK

        else:
            assert env.A_pos == (0,0), f"Boundary move {act} moved A off-board!"
    print("Boundary moves are blocked")

    # Test collision flips possession and blocks movement
    env.reset()
    env.A_pos, env.B_pos = (1,1), (2,1)
    env.ownership = 'A'
    env.player_move('A', 'S')   # A tries to move into B’s square
    assert env.A_pos == (1,1), "A should have been blocked"
    assert env.ownership == 'B', "Possession should have flipped to B"
    print("Collision logic correct")

    # Test goal scoring
    # Place A next to its goal, give A the ball, then step into the goal.
    env.reset()
    env.A_pos, env.B_pos = (1,1), (3,4)
    env.ownership = 'A'
    # we want A to move West into (1,0), which is in A_WINS
    # force move-order so A goes first:
    rand.seed(0)
    # 0. select random.choice(["A","B"]) → deterministic given the seed
    s, rA, rB, done = env.step('W', 'H')
    assert done, "Game should have terminated on goal"
    assert rA == +1 and rB == -1, f"Bad rewards: {rA},{rB}"
    print("Goal scoring works")

    # Test the face to face collision logic
    # Place A and B in adjacent squares, A has the ball
    env.reset()
    env.A_pos, env.B_pos = (1,1), (1,2)
    env.ownership = 'A'
    # A tries to move into B’s square, which should flip possession
    # and block A from moving
    env.player_move('A', 'E')  # A tries to move East into B
    assert env.A_pos == (1,1), "A should have been blocked from moving"
    assert env.ownership == 'B', "Possession should have flipped to B"
    # B tries to move into A’s square, which should flip possession
    # and block B from moving
    env.player_move('B', 'W')  # B tries to move West into A
    assert env.B_pos == (1,2), "B should have been blocked from moving"
    assert env.ownership == 'A', "Possession should have flipped back to A"
    print("Face-to-face collision logic works")

    #  Quick random rollout
    env.reset()
    for _ in range(50):
        # sample legal actions for both
        aA = rand.choice(env.legal_actions("A"))
        aB = rand.choice(env.legal_actions("B"))
        s, rA, rB, done = env.step(aA, aB)
        if done:
            env.reset()
    print("Random rollout (50 steps) completed without error")