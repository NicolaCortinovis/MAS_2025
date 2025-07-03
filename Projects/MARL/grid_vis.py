import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np
import os
from football_env import SimpleFootballGame

def draw_static_field(ax, rows=4, cols=5):
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
        ax.add_patch(plt.Rectangle((-1.5, y - 0.5), 1, 1, color='lightblue', alpha=0.3))
        ax.text(-1.0, y, "B's GOAL", va='center', ha='center', fontsize=8)

        ax.add_patch(plt.Rectangle((cols - 0.5, y - 0.5), 1, 1, color='gold', alpha=0.3))
        ax.text(cols, y, "A's GOAL", va='center', ha='center', fontsize=8)

    for spine in ax.spines.values():
        spine.set_visible(False)


def animate_episode(trajectory, grid_size=(4, 5), interval=600, save_as=None):
    rows, cols = grid_size
    fig, ax = plt.subplots(figsize=(cols + 2, rows))
    draw_static_field(ax, rows=rows, cols=cols)

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
        (a_row, a_col), (b_row, b_col), has_ball = trajectory[frame]
        agent_a.set_data([a_col], [a_row])
        agent_b.set_data([b_col], [b_row])
        label_a.set_position((a_col + 0.1, a_row + 0.1))
        label_b.set_position((b_col + 0.1, b_row + 0.1))
        label_a.set_text('A')
        label_b.set_text('B')
        ball_circle.set_center((a_col, a_row) if has_ball == 'A' else (b_col, b_row))
        ball_circle.set_visible(True)
        return agent_a, agent_b, label_a, label_b, ball_circle

    ani = animation.FuncAnimation(fig, update, frames=len(trajectory),
                                  init_func=init, blit=True, interval=interval, repeat=False)

    if save_as:
        os.makedirs("Results/Videos", exist_ok=True)
        ani.save(f"Results/Videos/{save_as}", writer="ffmpeg", fps=1)
        print(f"Saved animation to Results/Videos/{save_as}")

    plt.title("Soccer Game")
    plt.show()


def plot_policy(policy_a=None, policy_b=None, mode="both", grid_size=(4, 5),
                title=None, subtitle=None, save_as=None,
                fixed_position=None, ball_owner=None):

    rows, cols = grid_size
    fig, ax = plt.subplots(figsize=(cols + 2, rows))
    draw_static_field(ax, rows=rows, cols=cols)

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

    if subtitle:
        plt.title(subtitle, fontsize=10)
    if title:
        plt.suptitle(title, fontsize=14, fontstyle='italic')

    if fixed_position and mode in ["A", "B"]:
        row, col = fixed_position
        player_label = 'B' if mode == "A" else 'A'
        ax.text(col, row, player_label, fontsize=16, weight='bold',
                ha='center', va='center', color='black', zorder=10)

        if ball_owner == player_label:
            circle = patches.Circle((col, row), 0.4, fill=False, edgecolor='black', linewidth=2, zorder=9)
            ax.add_patch(circle)

    if save_as:
        os.makedirs("Results/Policies", exist_ok=True)
        plt.savefig(f"Results/Policies/{save_as}")
        print(f"Saved static policy plot to Results/Policies/{save_as}")

    plt.show()


def policy_for_starting_player(agent, env, starting_player, tracked_player, grid_size=(4, 5)):
    rows, cols = grid_size
    policy = {}

    if tracked_player == 'A':
        env.B_pos = (2, 1)  # B's starting position
        legal_B = env.legal_actions("B")
    else:
        env.A_pos = (1, 3)
        legal_A = env.legal_actions("A")

    env.ownership = starting_player

    for row in range(rows):
        for col in range(cols):
            if tracked_player == 'A':
                env.A_pos = (row, col)
                legal_A = env.legal_actions("A")
            else:
                env.B_pos = (row, col)
                legal_B = env.legal_actions("B")

            state = env.state()


            action = agent.select_action(state, legal_A, legal_B)
            policy[(row, col)] = action

    return policy


if __name__ == "__main__":
    
    # Plot the policies

    import pickle as pkl

    from football_env import SimpleFootballGame, A_STRT, B_STRT
    from football_agents import set_agent_to_greedy, collect_episode_states

    env = SimpleFootballGame()

    plotting_policies = False


    if plotting_policies == True:

        # Load the trained agent
        with open("Results/Agents/BR_agents/belief_agent_A_run0.pkl", "rb") as f:
            agent = pkl.load(f)

        # Set the agent to greedy mode
        set_agent_to_greedy(agent)


        # Policy plots of the starting states for 
        # BBA(r) vs RA
        # BBA(b) vs BBA(b)

        # Extract policies for agent A starting from its initial position
        policy_BBA_r_owner = policy_for_starting_player(agent, env, starting_player='A', tracked_player='A')
        policy_BBA_r_not_owner = policy_for_starting_player(agent, env, starting_player='B', tracked_player='A')

        plot_policy(policy_a = policy_BBA_r_owner, fixed_position = B_STRT, ball_owner= "A",mode="A", subtitle="BBA(r) vs RA", title = "A player π with ball ownership at game startt", save_as="Owner_BBA(R)vsRA_π.png")
        plot_policy(policy_a = policy_BBA_r_not_owner, fixed_position = B_STRT, ball_owner= "B", mode="A", subtitle="BBA(r) vs RA", title = "A player π without ball ownership at game start", save_as="BBA(R)vsOwnerRA_π.png")



        # Load the trained agent for BBA(b)
        with open("Results/Agents/BBA_agents/belief_agent_A_run0.pkl", "rb") as f:
            agent_a = pkl.load(f)

        with open("Results/Agents/BBA_agents/belief_agent_B_run0.pkl", "rb") as f:
            agent_b = pkl.load(f)

        # Set the agent to greedy mode
        set_agent_to_greedy(agent_a)
        set_agent_to_greedy(agent_b)

        # Extract policies for agent A starting from its initial position
        policy_BBA_b_owner_A_player = policy_for_starting_player(agent_a, env, starting_player='A', tracked_player='A')
        policy_BBA_b_not_owner_B_player = policy_for_starting_player(agent_b, env, starting_player='A', tracked_player='B')

        plot_policy(policy_a = policy_BBA_b_owner_A_player, mode="A", fixed_position = B_STRT, ball_owner= "A", subtitle="BBA(b) vs BBA(b)", title = "A player π with ball ownership at game start", save_as="Owner_BBA(B)vsBBA(B)_Aπ.png")
        plot_policy(policy_b = policy_BBA_b_not_owner_B_player, mode="B", fixed_position = A_STRT, ball_owner= "A", subtitle="BBA(b) vs BBA(b)", title = "B player π without ball ownership at game start", save_as="Owner_BBA(B)vsBBA(B)_Bπ.png")

    

    # Animate an episode of BBA(r) vs RA and one of BBA(b) vs BBA(b)

    # load the agents

    with open("Results/Agents/BR_agents/belief_agent_A_run0.pkl", "rb") as f:
        agent_BBA_r_A = pkl.load(f)

    with open("Results/Agents/BBA_agents/belief_agent_A_run0.pkl", "rb") as f:
        agent_BBA_b_A = pkl.load(f)
    
    with open("Results/Agents/BBA_agents/belief_agent_B_run0.pkl", "rb") as f:
        agent_BBA_b_B = pkl.load(f)

    set_agent_to_greedy(agent_BBA_r_A)
    set_agent_to_greedy(agent_BBA_b_A)
    set_agent_to_greedy(agent_BBA_b_B)

    # Animate an episode of BBA(r) vs RA
    env.reset()
    trajectory_BBA_r = collect_episode_states(env, agent_BBA_r_A, None, mode = 'random')
    animate_episode(trajectory_BBA_r, save_as="BBA(R)vsRA_episode.mp4")


    # Animate an episode of BBA(b) vs BBA(b)
    env.reset()
    trajectory_BBA_b = collect_episode_states(env, agent_BBA_b_A, agent_BBA_b_B, mode = 'selfplay')
    animate_episode(trajectory_BBA_b, save_as="BBA(B)vsBBA(B)_episode.mp4")
