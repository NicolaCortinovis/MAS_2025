import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np
import os
from football_env import SimpleFootballGame

def draw_static_field(ax, cols=5, rows=4):
    """
    Draws a static soccer field with goals and grid lines.
    :param ax: Matplotlib axis to draw on.
    :param cols: Number of columns in the grid.
    :param rows: Number of rows in the grid.
    """
    ax.set_xlim(-1.5, cols + 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    plt.gca().invert_yaxis()

    # Draw internal grid
    for x in range(cols):
        for y in range(rows):
            rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, edgecolor='black', facecolor='none')
            ax.add_patch(rect)

    # Draw goal nets
    for y in [1, 2]:
        # A's goal (left)
        ax.add_patch(plt.Rectangle((-1.5, y - 0.5), 1, 1, color='lightblue', alpha=0.3))
        ax.text(-1.0, y, "B's GOAL", va='center', ha='center', fontsize=8)

        # B's goal (right, flush with grid)
        ax.add_patch(plt.Rectangle((cols - 0.5, y - 0.5), 1, 1, color='gold', alpha=0.3))
        ax.text(cols, y, "A's GOAL", va='center', ha='center', fontsize=8)

    # Remove outer border (spines)
    for spine in ax.spines.values():
        spine.set_visible(False)



def animate_episode(trajectory, grid_size=(5, 4), interval=600, save_as=None):
    """
    Animates an episode of the soccer game.
    :param trajectory: List of states from the episode.
    :param grid_size: Tuple (cols, rows) for the grid size.
    :param interval: Delay between frames in milliseconds.
    :param save_as: If provided, saves the animation to this filename.
    """

    cols, rows = grid_size
    fig, ax = plt.subplots(figsize=(cols + 2, rows))
    draw_static_field(ax, cols, rows)

    agent_a = ax.plot([], [], 'o', color='#F03b20')[0]   
    agent_b = ax.plot([], [], 's', color='blue')[0]
    label_a = ax.text(0, 0, '', fontsize=10)
    label_b = ax.text(0, 0, '', fontsize=10)
    ball_circle = patches.Circle((0, 0), 0.4, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(ball_circle)
    ball_circle.set_visible(False)

    def init():
        agent_a.set_data([], [])
        agent_b.set_data([], [])
        ball_circle.set_visible(False)
        return agent_a, agent_b, label_a, label_b, ball_circle

    def update(frame):
        (ax_b_x, ax_b_y), (ax_a_x, ax_a_y), has_ball = trajectory[frame]
        agent_a.set_data([ax_a_x], [ax_a_y])
        agent_b.set_data([ax_b_x], [ax_b_y])
        label_a.set_position((ax_a_x + 0.1, ax_a_y + 0.1))
        label_b.set_position((ax_b_x + 0.1, ax_b_y + 0.1))
        label_a.set_text('A')
        label_b.set_text('B')
        ball_circle.set_center((ax_a_x, ax_a_y) if has_ball == 'A' else (ax_b_x, ax_b_y))
        ball_circle.set_visible(True)
        return agent_a, agent_b, label_a, label_b, ball_circle

    ani = animation.FuncAnimation(fig, update, frames=len(trajectory),
                                  init_func=init, blit=True, interval=interval, repeat=False)

    if save_as:
        os.makedirs("results", exist_ok=True)
        ani.save(f"results/{save_as}", writer="ffmpeg", fps=1)
        print(f"Saved animation to results/{save_as}")

    plt.title("Soccer Game")
    plt.show()

def plot_policy(policy_a=None, policy_b=None, mode="both", grid_size=(5, 4), title=None, save_as=None):
    """
    Plots the policies for agents A and B on a grid.
    :param policy_a: Dictionary mapping (row, col) to action for agent A.
    :param policy_b: Dictionary mapping (row, col) to action for agent B.
    :param mode: "A", "B", or "both" to indicate which policies to plot.
    :param grid_size: Tuple (cols, rows) for the grid size.
    :param title: Title for the plot.
    :param save_as: If provided, saves the plot to this filename.
    """
    cols, rows = grid_size
    fig, ax = plt.subplots(figsize=(cols + 2, rows))
    draw_static_field(ax, cols, rows)

    arrow_dx = {'N': 0, 'S': 0, 'E': 0.4, 'W': -0.4, 'H': 0}
    arrow_dy = {'N': -0.4, 'S': 0.4, 'E': 0, 'W': 0, 'H': 0}

    if mode in ["A", "both"] and policy_a:
        for (row, col), action in policy_a.items():
            dx = arrow_dx.get(action, 0)
            dy = arrow_dy.get(action, 0)
            ax.arrow(col, row, dx, dy,
                     head_width=0.15, head_length=0.1,
                     fc='red', ec='black', alpha=0.7)

    if mode in ["B", "both"] and policy_b:
        for (row, col), action in policy_b.items():
            dx = arrow_dx.get(action, 0)
            dy = arrow_dy.get(action, 0)
            ax.arrow(col + 0.1, row + 0.1, dx, dy,
                     head_width=0.15, head_length=0.1,
                     fc='blue', ec='black', alpha=0.7)

    if title:
        plt.title(title)

    if save_as:
        os.makedirs("results", exist_ok=True)
        plt.savefig(f"results/{save_as}")
        print(f"Saved static policy plot to results/{save_as}")

    plt.show()



def main():

    #######################################################################
        ########### Example of a simple football game ################
    #######################################################################

    env = SimpleFootballGame()

    # Define env characteristics
    env.n_rows = 4
    env.n_cols = 5
    env.diffs = {
        'N': (-1,  0),
        'S': ( 1,  0),
        'E': ( 0,  1),
        'W': ( 0, -1),
        'H': ( 0,  0)
    }

    trajectory = []
    env.reset()
    trajectory.append(env.get_state())

    for _ in range(15):
        a_action = np.random.choice(env.legal_actions("A"))
        b_action = np.random.choice(env.legal_actions("B"))
        env.step(a_action, b_action)
        trajectory.append(env.get_state())

    animate_episode(trajectory, save_as="example_game.mp4")

    # Random dummy policies for demo
    policy_a = {(r, c): np.random.choice(['N', 'S', 'E', 'W', 'H'])
                for r in range(env.n_rows) for c in range(env.n_cols)}
    policy_b = {(r, c): np.random.choice(['N', 'S', 'E', 'W', 'H'])
                for r in range(env.n_rows) for c in range(env.n_cols)}

    # Visualize separately
    plot_policy(policy_a=policy_a, mode="A", grid_size=(5, 4), title="Policy A")
    plot_policy(policy_b=policy_b, mode="B", grid_size=(5, 4), title="Policy B")

    # Visualize combined
    plot_policy(policy_a=policy_a, policy_b=policy_b, mode="both",
                grid_size=(5, 4), title="Combined Policy", save_as="combined_policy.png")



if __name__ == "__main__":
    
    # Plot the policies

    import pickle as pkl

    from football_env import SimpleFootballGame  # your env class
    from football_agents import set_agent_to_greedy

    env = SimpleFootballGame()

    # Load the trained agent
    with open("Results/Agents/BR_agents/belief_agent_A_run0.pkl", "rb") as f:
        agent = pkl.load(f)

    # Set the agent to greedy mode
    set_agent_to_greedy(agent)



    policy_a = {}
    rows, cols = 4, 5

    # Iterate over the grid positions where Agent A could be
    for row in range(rows):
        for col in range(cols):

            env.A_pos = (row, col)
            env.B_pos = (2, 1)
            env.ownership = "A"

            state = env.state()  # This should give you the state index (0–799)
            
            legal_A = env.legal_actions("A")
            legal_B = env.legal_actions("B")
        
            
            action = agent.select_action(state, legal_A, legal_B)
            policy_a[(row, col)] = action
            print(action)

    # Plot the extracted policy
    plot_policy(policy_a=policy_a, mode="A", title="Agent A Policy")
