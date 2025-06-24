# This file contains the class and functions for the simplified football game environment described in the littman94 paper.

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

NUM_STATES = 2 * (N_CELLS ** 2)   


class SimpleFootballGame:

    def __init__(self):
        """
        Initialize the football game environment.
        The environment consists of a 4x5 grid where two players (A and B) can move
        and score in their respective goal squares. The game starts with A and B
        positioned at their starting positions, and the ownership of the ball is
        randomly assigned to either player. The game allows for movement in the grid,
        with North, South, East, West, and Halt actions. The game logic includes
        scoring conditions, legal actions based on player positions, and  collision handling.
        """
        self.A_pos = A_STRT  # A's starting position
        self.B_pos = B_STRT  # B's starting position
        self.N_CELLS = N_CELLS  # Total number of cells in the field
        self.ACT = ACT  # Set of actions
        self.A_STRT = A_STRT  # A's starting position
        self.B_STRT = B_STRT  # B's starting position 
        self.A_WINS = A_WINS  # Winning positions for A
        self.B_WINS = B_WINS  # Winning positions for B
        self.DIFFS = DIFFS  # Action differences
        self.ownership = rand.choice(["A", "B"])  # Randomly assign ownership at the start

    def reset(self):
        """
        Reset the environment to its initial state.
        """
        self.A_pos = A_STRT
        self.B_pos = B_STRT
        self.ownership = rand.choice(["A", "B"])

        return self.state()


    def state(self):
        """
        Function to return the flattened state of the environment.
        A state is represented by player A's position, player B's position, and the ownership of the ball.
        Therefore, the state is an integer in the range [0, 799].
        """
        x_a, y_a = self.A_pos 
        x_b, y_b = self.B_pos

        index_a = x_a * COL + y_a  # Convert A's position 
        index_b = x_b * COL + y_b  # Convert B's position


        index = 2 * ((index_a * N_CELLS) + index_b) + (0 if self.ownership == "A" else 1)

        return index

    def get_state(self):
        """
        Decode the integer state back into ((x_b,y_b),(x_a,y_a), ownership). 
        Used for visualization and debugging.
        """
        idx = self.state()
        bit = idx % 2
        idx //= 2
        combined = idx  # 0..399
        idx_b = combined % N_CELLS  # 0..19
        idx_a = combined // N_CELLS
        x_b, y_b = divmod(idx_b, COL)
        x_a, y_a = divmod(idx_a, COL)
        owner = "A" if bit == 0 else "B"
        return (x_a, y_a), (x_b, y_b), owner
    
    def clamp(self, row, col):
        """
        Keep (row,col) on the board.
        :param row: Row index
        :param col: Column index
        :return: Clamped (row, col) tuple within the bounds of the grid
        """
        return (
            min(max(0, row), ROW-1),
            min(max(0, col), COL-1)
        )
    
    def legal_actions(self, player):
        """
        Get the legal actions for the given player, but
        always allow the scoring move in the goal squares.
        :param player: "A" or "B"
        :return: List of legal actions for the player
        """

        pos = self.A_pos if player == "A" else self.B_pos
        legal = []
        for act, (dr, dc) in DIFFS.items():
            new = self.clamp(pos[0] + dr, pos[1] + dc)
            # normally prune no‐ops (wall‐bounces)
            if act == 'H' or new != pos:
                legal.append(act)

        # **Special‐case**: if you're in your goal square, allow the goal‐direction move
        if player == "A" and pos in A_WINS and self.ownership == "A":
            if 'W' not in legal:
                legal.append('W')
        if player == "B" and pos in B_WINS and self.ownership == "B":
            if 'E' not in legal:
                legal.append('E')

        return legal
    
    def player_move(self, player, action):
        """
        Move player or score.  Returns (done, reward_A, reward_B).
        """
        # identify current pos, other pos, etc.
        if player=='A':
            cur_pos, other_pos = self.A_pos, self.B_pos
            pos_attr, opp       = 'A_pos', 'B'
            goal_cells, goal_mv = A_WINS, 'W'
        else:
            cur_pos, other_pos = self.B_pos, self.A_pos
            pos_attr, opp       = 'B_pos', 'A'
            goal_cells, goal_mv = B_WINS, 'E'

        dr, dc = DIFFS[action]
        desired = (cur_pos[0] + dr, cur_pos[1] + dc)

        # Did someone score?
        if self.ownership == player and cur_pos in goal_cells and action == goal_mv:
            if player=='A':
                return True, +1, -1  # A scores
            else:
                return True, -1, +1  # B scores

        if desired == other_pos and action!='H':
            # collision → flip possession
            self.ownership = opp
        else:
            # otherwise, move to desired position
            setattr(self, pos_attr, desired)

        # Game continues, no score
        return False, 0, 0

        
    def step(self, action_a, action_b):
        """
        Execute A’s and B’s moves in random order, checking for score after each.
        :param action_a: Action for player A
        :param action_b: Action for player B
        :return: (state, reward_A, reward_B, done)
        """
        # Randomly choose the order of actions and execute them
        for player, act in rand.sample([('A', action_a), ('B', action_b)], 2):
            done, rA, rB = self.player_move(player, act)

            if done:  # If someone scored, return the state and rewards
                # Reset positions and ownership for the next game
                self.reset()
                return self.state(), rA, rB, True

        # no goal this step
        return self.state(), 0, 0, False
    

class SimpleFootballGameClamp(SimpleFootballGame):
    """
    Variant where every of the 5 actions ('N','S','E','W','H') is always legal.
    Any move that would push you off the board is instead clamped to stay in the same
    boundary cell. Used for internal tests.
    """
    def legal_actions(self, player):
        # Everything is legal
        return list(DIFFS.keys())

    def player_move(self, player, action):
        """
        Exactly as in the parent, except we clamp *every* non-scoring move
        so that desired is always on-board.
        """

        if player == 'A':
            cur_pos, other_pos = self.A_pos, self.B_pos
            pos_attr, opp       = 'A_pos', 'B'
            goal_cells, goal_mv = A_WINS, 'W'
        else:
            cur_pos, other_pos = self.B_pos, self.A_pos
            pos_attr, opp       = 'B_pos', 'A'
            goal_cells, goal_mv = B_WINS, 'E'

        dr, dc = DIFFS[action]
        desired = (cur_pos[0] + dr, cur_pos[1] + dc)


        if self.ownership == player and cur_pos in goal_cells and action == goal_mv:
            return (True, +1, -1) if player=='A' else (True, -1, +1)


        new_pos = self.clamp(desired[0], desired[1])
        if new_pos == other_pos and action != 'H':
            self.ownership = opp
        else:
            setattr(self, pos_attr, new_pos)

        return False, 0, 0




if __name__ == "__main__":

    ####################################################################
    ############# Test the SimpleFootballGame Environment ##############
    ####################################################################

    env = SimpleFootballGame()


    # Test state encoding/decoding correctness
    for s0 in range(NUM_STATES):
            # manually set the internal state via decoding
            # decode ownership bit
            bit = s0 % 2
            combined = s0 // 2
            idx_a = combined // N_CELLS
            idx_b = combined % N_CELLS
            x_a, y_a = divmod(idx_a, COL)
            x_b, y_b = divmod(idx_b, COL)
            env.A_pos = (x_a, y_a)
            env.B_pos = (x_b, y_b)
            env.ownership = "A" if bit == 0 else "B"
            # re-encode via state()
            s1 = env.state()
            assert s0 == s1, f"Encoding mismatch: manual {s0} vs state() {s1} for A:{env.A_pos}, B:{env.B_pos}, owner:{env.ownership}"
    print("State encoding/decoding correctness verified for all states")

    # Test over 10k random resets that the ownership is randomly assigned
    ownerships = []
    for _ in range(10_000):
        env.reset()
        ownerships.append(env.ownership)
    ownership_counts = {k: ownerships.count(k) for k in set(ownerships)}
    assert len(ownership_counts) == 2, "Ownership should be either 'A' or 'B'"
    assert ownership_counts['A'] > 0 and ownership_counts['B'] > 0, "Both ownerships should be represented in the resets"
    print("Observed ownerships:", ownership_counts)

    # Test that legal actions are correctly generated
    env.reset()
    env.A_pos, env.B_pos = (1,1), (2,1)
    env.ownership = 'A'
    legal_A = env.legal_actions('A')
    legal_B = env.legal_actions('B')
    assert 'N' in legal_A and 'S' in legal_A and 'E' in legal_A and 'W' in legal_A and 'H' in legal_A, "A should have all actions available"
    assert 'N' in legal_B and 'S' in legal_B and 'E' in legal_B and 'W' in legal_B and 'H' in legal_B, "B should have all actions available"
    print("Legal actions generation works") 

    # Test that legal actions are pruned correctly
    env.reset()
    env.A_pos, env.B_pos = (0,0), (1,1)
    env.ownership = 'A'
    legal_A = env.legal_actions('A')
    legal_B = env.legal_actions('B')
    assert 'N' not in legal_A, "A should not be able to move North from (0,0)"
    assert 'W' not in legal_A, "A should not be able to move West from (0,0)"
    assert 'S' in legal_A and 'E' in legal_A, "A should be able to move South or East from (0,0)"
    assert 'N' in legal_B and 'S' in legal_B and 'E' in legal_B and 'W' in legal_B and 'H' in legal_B, "B should have all actions available from (1,1)"
    print("Legal actions pruning works")

    # Test that legal actions pruning works when in goal squares
    # A in its goal square, B in a normal square
    env.reset()
    env.A_pos, env.B_pos = (1,0), (2,1)
    env.ownership = 'A'
    legal_A = env.legal_actions('A') # A should be able to score
    legal_B = env.legal_actions('B')
    # A should have all actions available:
    assert 'N' in legal_A and 'S' in legal_A and 'E' in legal_A and 'W' in legal_A and 'H' in legal_A, "A should have all actions available from its goal square"

    # B in its goal square, A in a normal square
    env.reset()
    env.A_pos, env.B_pos = (2,1), (1,4)
    env.ownership = 'B'
    legal_A = env.legal_actions('A')
    legal_B = env.legal_actions('B') # B should be able to score
    # B should have all actions available:
    assert 'N' in legal_B and 'S' in legal_B and 'E' in legal_B and 'W' in legal_B and 'H' in legal_B, "B should have all actions available from its goal square"

    print("Legal actions pruning in goal squares works")

    # Test collision flips possession and blocks movement
    env.reset()
    env.A_pos, env.B_pos = (1,1), (2,1)
    env.ownership = 'A'
    env.player_move('A', 'S')   # A tries to move into B’s square
    assert env.A_pos == (1,1), "A should have been blocked"
    assert env.ownership == 'B', "Possession should have flipped to B"
    print("Collision logic correct")

    # Test scoring logic
    # A scores
    env.reset()
    env.A_pos, env.B_pos = (1,0), (2,1)
    A_move = 'W'  # A moves West to score
    B_move = 'H'  # B does nothing
    env.ownership = 'A'
    _ , rA, rB, done = env.step(A_move, B_move)  # A scores
    assert done, "A should have scored"
    assert rA == 1 and rB == -1, "Rewards should be +1 for A and -1 for B"
    assert env.A_pos == A_STRT and env.B_pos == B_STRT, "Positions should reset after scoring"
    assert env.ownership in ['A', 'B'], "Ownership should be randomly assigned after scoring"
    print("Scoring logic works for A")

    # B scores
    env.reset()
    env.B_pos, env.A_pos = (2,4), (1,1)
    env.ownership = 'B'
    A_move = 'H'  # A does nothing
    B_move = 'E'  # B moves East to score
    _, rA, rB, done = env.step(A_move, B_move)
    assert done, "B should have scored"
    assert rA == -1 and rB == 1, "Rewards should be -1 for A and +1 for B"
    assert env.A_pos == A_STRT and env.B_pos == B_STRT, "Positions should reset after scoring"
    assert env.ownership in ['A', 'B'], "Ownership should be randomly assigned after scoring"
    print("Scoring logic works for B")

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