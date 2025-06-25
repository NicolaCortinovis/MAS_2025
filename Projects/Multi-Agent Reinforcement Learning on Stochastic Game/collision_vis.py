import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np
import os
from football_env import SimpleFootballGame

def draw_static_field(ax, cols=5, rows=4):
    ax.set_xlim(-1.5, cols + 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    plt.gca().invert_yaxis()

    for x in range(cols):
        for y in range(rows):
            rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, edgecolor='black', facecolor='none')
            ax.add_patch(rect)

    for y in [1, 2]:
        ax.add_patch(plt.Rectangle((-1.5, y - 0.5), 1, 1, color='lightblue', alpha=0.3))
        ax.text(-1.0, y, "B's GOAL", va='center', ha='center', fontsize=8)
        ax.add_patch(plt.Rectangle((cols - 0.5, y - 0.5), 1, 1, color='gold', alpha=0.3))
        ax.text(cols, y, "A's GOAL", va='center', ha='center', fontsize=8)

    for spine in ax.spines.values():
        spine.set_visible(False)


def animate_episode(trajectory, grid_size=(5, 4), interval=600, save_as=None,
                    show_moves=False, moving_sequence=None, at_same_time=True):
    cols, rows = grid_size
    fig, ax = plt.subplots(figsize=(cols + 2, rows))
    draw_static_field(ax, cols, rows)

    agent_a = ax.plot([], [], 'o', color='#F03b20')[0]
    agent_b = ax.plot([], [], 's', color='blue')[0]
    label_a = ax.text(0, 0, '', fontsize=12)
    label_b = ax.text(0, 0, '', fontsize=12)
    move_a_label = ax.text(0, 0, '', fontsize=16, color='#F03b20', fontweight='normal')
    move_b_label = ax.text(0, 0, '', fontsize=16, color='blue', fontweight='normal')

    ball_circle = patches.Circle((0, 0), 0.4, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(ball_circle)
    ball_circle.set_visible(False)

    # Previous move tracking
    prev_a_action = ''
    prev_b_action = ''
    prev_a_pos = (0, 0)
    prev_b_pos = (0, 0)

    def init():
        agent_a.set_data([], [])
        agent_b.set_data([], [])
        ball_circle.set_visible(False)
        move_a_label.set_text('')
        move_b_label.set_text('')
        return agent_a, agent_b, label_a, label_b, ball_circle, move_a_label, move_b_label

    def get_arrow_from_action(action):
        return {'N': '↑', 'S': '↓', 'E': '→', 'W': '←', 'H': 'H'}.get(action, '')

    def update(frame):
        nonlocal prev_a_action, prev_b_action, prev_a_pos, prev_b_pos

        (a_y, a_x), (b_y, b_x), has_ball, a_action, b_action = trajectory[frame]

        # Agent positions
        agent_a.set_data([a_x], [a_y])
        agent_b.set_data([b_x], [b_y])
        label_a.set_position((a_x + 0.1, a_y + 0.1))
        label_b.set_position((b_x + 0.1, b_y + 0.1))
        label_a.set_text('A')
        label_b.set_text('B')

        # Ball position
        ball_circle.set_center((a_x, a_y) if has_ball == 'A' else (b_x, b_y))
        ball_circle.set_visible(True)

        # Move arrows
        if show_moves:
            current_a = get_arrow_from_action(a_action) if a_action else ''
            current_b = get_arrow_from_action(b_action) if b_action else ''

            if at_same_time:
                if a_action == 'H' and b_action == 'H' and frame > 0:
                    # Retain previous arrows and positions
                    move_a_label.set_text(get_arrow_from_action(prev_a_action))
                    move_b_label.set_text(get_arrow_from_action(prev_b_action))
                    move_a_label.set_position((prev_a_pos[0], prev_a_pos[1] - 1))
                    move_b_label.set_position((prev_b_pos[0], prev_b_pos[1] - 1))
                    move_a_label.set_fontweight('normal')
                    move_b_label.set_fontweight('normal')
                else:
                    # Show both arrows normally
                    move_a_label.set_text(current_a)
                    move_b_label.set_text(current_b)
                    move_a_label.set_position((a_x, a_y - 1))
                    move_b_label.set_position((b_x, b_y - 1))
                    move_a_label.set_fontweight('normal')
                    move_b_label.set_fontweight('normal')
                    prev_a_action = a_action
                    prev_b_action = b_action
                    prev_a_pos = (a_x, a_y)
                    prev_b_pos = (b_x, b_y)
            else:
                # Sequential mode: show only intended move in bold
                move_a_label.set_text(current_a if a_action != 'H' else '')
                move_b_label.set_text(current_b if b_action != 'H' else '')
                move_a_label.set_position((a_x, a_y - 1))
                move_b_label.set_position((b_x, b_y - 1))
                move_a_label.set_fontweight('bold' if moving_sequence and moving_sequence[frame] == 'A' else 'normal')
                move_b_label.set_fontweight('bold' if moving_sequence and moving_sequence[frame] == 'B' else 'normal')
        else:
            move_a_label.set_text('')
            move_b_label.set_text('')

        return agent_a, agent_b, label_a, label_b, ball_circle, move_a_label, move_b_label

    ani = animation.FuncAnimation(fig, update, frames=len(trajectory),
                                  init_func=init, blit=True, interval=interval, repeat=False)

    if save_as:
        os.makedirs("results/collisions", exist_ok=True)
        ani.save(f"results/collisions/{save_as}", writer="ffmpeg", fps=1)
        print(f"Saved animation to results/collisions/{save_as}")


    plt.title("Soccer Game")
    plt.show()



def create_collision_scenario(initial_a, initial_b, action_a, action_b, ownership, env):
    env.reset()
    env.A_pos = initial_a
    env.B_pos = initial_b
    env.ownership = ownership
    trajectory = []
    trajectory.append((env.A_pos, env.B_pos, env.ownership, None, None))
    env.step(action_a, action_b)
    trajectory.append((env.A_pos, env.B_pos, env.ownership, action_a, action_b))
    env.step('H', 'H')
    trajectory.append((env.A_pos, env.B_pos, env.ownership, 'H', 'H'))
    return trajectory

def create_sequential_collision_scenario(initial_a, initial_b, action_a, action_b, ownership, env, first_to_move='B'):
    env.reset()
    env.A_pos = initial_a
    env.B_pos = initial_b
    env.ownership = ownership
    trajectory = []
    trajectory.append((env.A_pos, env.B_pos, env.ownership, None, None))
    if first_to_move == 'A':
        env.step(action_a, 'H')
        trajectory.append((env.A_pos, env.B_pos, env.ownership, action_a, 'H'))
        env.step('H', 'H')
        trajectory.append((env.A_pos, env.B_pos, env.ownership, 'H', 'H'))
        env.step('H', action_b)
        trajectory.append((env.A_pos, env.B_pos, env.ownership, 'H', action_b))
    elif first_to_move == 'B':
        env.step('H', action_b)
        trajectory.append((env.A_pos, env.B_pos, env.ownership, 'H', action_b))
        env.step('H', 'H')
        trajectory.append((env.A_pos, env.B_pos, env.ownership, 'H', 'H'))
        env.step(action_a, 'H')
        trajectory.append((env.A_pos, env.B_pos, env.ownership, action_a, 'H'))
    env.step('H', 'H')
    trajectory.append((env.A_pos, env.B_pos, env.ownership, 'H', 'H'))
    return trajectory

if __name__ == "__main__":
    env = SimpleFootballGame()
    env.n_rows = 4
    env.n_cols = 5
    env.diffs = {
        'N': (-1, 0),
        'S': (1, 0),
        'E': (0, 1),
        'W': (0, -1),
        'H': (0, 0)
    }

    traj1 = create_collision_scenario(
        initial_a=(2, 2), initial_b=(2, 1), action_a='W', action_b='H', ownership='A', env=env
    )
    animate_episode(traj1, save_as="collision_case1.mp4", show_moves=True)

    traj_b_first = create_sequential_collision_scenario(
        initial_a=(2, 2), initial_b=(2, 1), action_a='W', action_b='E', ownership='A', env=env, first_to_move='B'
    )
    animate_episode(traj_b_first, save_as="collision_b_first.mp4", show_moves=True,
                    moving_sequence=[None, 'B', None, 'A', None], at_same_time=False)

    traj_a_first = create_sequential_collision_scenario(
        initial_a=(2, 2), initial_b=(2, 1), action_a='W', action_b='E', ownership='A', env=env, first_to_move='A'
    )
    animate_episode(traj_a_first, save_as="collision_a_first.mp4", show_moves=True,
                    moving_sequence=[None, 'A', None, 'B', None], at_same_time=False)
