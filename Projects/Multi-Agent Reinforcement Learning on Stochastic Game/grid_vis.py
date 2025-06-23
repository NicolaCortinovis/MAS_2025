import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np
import os
from football import Football  

def draw_static_field(ax, cols=5, rows=4):
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

    # Row labels
    for y in range(rows):
        ax.text(-2.0, y, str(y + 1), va='center', ha='center', fontsize=10)

    # Column labels
    for x in range(cols):
        ax.text(x, 3.8, str(x + 1), va='center', ha='center', fontsize=10)

    # Draw goal nets
    for y in [1, 2]:
        # A's goal (left)
        ax.add_patch(plt.Rectangle((-1.5, y - 0.5), 1, 1, color='gold', alpha=0.3))
        ax.text(-1.0, y, 'GOAL A', va='center', ha='center', fontsize=8)

        # B's goal (right, flush with grid)
        ax.add_patch(plt.Rectangle((cols - 0.5, y - 0.5), 1, 1, color='lightblue', alpha=0.3))
        ax.text(cols, y, 'GOAL B', va='center', ha='center', fontsize=8)

    # Remove outer border (spines)
    for spine in ax.spines.values():
        spine.set_visible(False)



def animate_episode(env_states, grid_size=(5, 4), interval=600, save_as=None):
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
        (ax_a_x, ax_a_y), (ax_b_x, ax_b_y), has_ball = env_states[frame]
        agent_a.set_data([ax_a_x], [ax_a_y])
        agent_b.set_data([ax_b_x], [ax_b_y])
        label_a.set_position((ax_a_x + 0.1, ax_a_y + 0.1))
        label_b.set_position((ax_b_x + 0.1, ax_b_y + 0.1))
        label_a.set_text('A')
        label_b.set_text('B')
        ball_circle.set_center((ax_a_x, ax_a_y) if has_ball == 'A' else (ax_b_x, ax_b_y))
        ball_circle.set_visible(True)
        return agent_a, agent_b, label_a, label_b, ball_circle

    ani = animation.FuncAnimation(fig, update, frames=len(env_states),
                                  init_func=init, blit=True, interval=interval, repeat=False)

    if save_as:
        os.makedirs("results", exist_ok=True)
        ani.save(f"results/{save_as}", writer="ffmpeg", fps=1)
        print(f"Saved animation to results/{save_as}")

    plt.title("Soccer Game")
    plt.show()

def main():
    env = Football()

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

if __name__ == "__main__":
    main()
