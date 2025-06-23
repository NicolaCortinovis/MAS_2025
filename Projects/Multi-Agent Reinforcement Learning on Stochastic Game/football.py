# This file contains the class and functions for the football game environment described in the littman94 paper.

import numpy as np
import random as rand


ROW, COL = 4, 5                     # Size of the football field
ACT = {"N", "S", "E", "W", "H"}     # North, South, East, West, Halt (should be stand but for clearance halt)
A_STRT = (1,3)                      # Where the A agent starts
B_STRT = (2,1)                      # Where the B agent starts
A_WINS = [(1,0), (2,0)]             # Scoring positions for A
B_WINS = [(1,4), (2,4)]             # Scoring positions for B


class Football:

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


        index = 2*(index_a * (COL * ROW) + index_b) + (0 if self.ownership == "A" else 1)

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
        Return only those moves for `player` that do
        something (or are 'H'), i.e. that wouldn't
        be guaranteed no-ops by heading off the board.
        """

        pos = self.A_pos if player=="A" else self.B_pos
        legal = []
        for act, (dr,dc) in self.diffs.items():
            new = self.clamp(pos[0] + dr, pos[1] + dc)
            # keep the halt action always; prune other moves
            # whose clamped new position == pos (i.e. they'd hit the wall)
            if act == 'H' or new != pos:
                legal.append(act)

        # If player A is in A_WINS or player B is in B_WINS, then
        # they can act "illegaly" to score, so we allow all actions
        if player == 'A' and self.A_pos in A_WINS:
            legal = list(ACT)
        elif player == 'B' and self.B_pos in B_WINS:
            legal = list(ACT)

        return legal
    
    def player_move(self, player, action):
        
        # Effect of each action
        diffs = {
            'N': (-1,  0),
            'S': ( 1,  0),
            'E': ( 0,  1),
            'W': ( 0, -1),
            'H': ( 0,  0)
        }

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

        # Compute desired move, clamped to grid (Correct handling?)
        dr, dc = diffs.get(action, (0, 0))
        raw_row = cur_pos[0] + dr
        raw_col = cur_pos[1] + dc
        desired = (
            min(max(0, raw_row), self.n_rows - 1),
            min(max(0, raw_col), self.n_cols - 1)
        )

        # If desired is occupied and it the action was not S, block & flip
        if desired == other_pos and action != 'H':
            self.ownership = other
        else: # Otherwise, move there and keep ownership
            setattr(self, pos_attr, desired)

        
        # Did a player score?
        if self.A_pos in A_WINS and action == 'W':
            return "A wins"
        elif self.B_pos in B_WINS and action == 'E':
            return "B wins"
        else:
            return "No one wins"

    def step(self, action_a, action_b):
        
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


        

        return self.state()


if __name__ == "__main__":
    pass