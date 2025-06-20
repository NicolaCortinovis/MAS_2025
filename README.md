# MAS_2025
This repository contains the code for the projects of the Multi Agent Systems course teached by Professors Padoan Tommaso and Petrov Tatjana at the Artificial Intelligence &amp; Data Science Master @ University of Trieste

## 6. Scheduling Problem with Random Arrivals

**Setting**: Scheduling problems are central to computing systems, where jobs (or tasks) must be assigned
to limited resources over time. The goal is often to optimize metrics such as average waiting time,
latency[^latency], or system throughput. The complexity of the scheduling problem varies depending on system
assumptions, such as the number of servers, whether preemption is allowed, and whether job durations
and arrival times are known in advance.
**Objectives**:
 - Implement a simulation of a task processing system where tasks arrive according to a Poisson
process and are distributed to three processors.
- Assume each task has a service time drawn from an exponential or another specified distribution
- Compare at least two scheduling strategies — such as **First-In-First-Out (FIFO)** and **Shortest
Job First (SJF)** — in terms of throughput, average waiting time, and load balance.
- Evaluate the strategies under different load conditions (e.g., low vs. high arrival rate)
- (optional) Model the scheduling problem as an MDP

[^latency]:  **Latency** refers to the total time a job spends in the system from its arrival to its completion. It includes both the
waiting time and the service time. Minimizing latency is crucial in systems where responsiveness is key, such as in real-time
processing or user-facing services.

## Results

---

## 2. Multi-Agent Reinforcement Learning on Stochastic Game
#### Setting: Simplified Football Game
A simplified version of football introduced by Littman[^Lit94]. The game is played on a grid 4 × 5 as
depicted below. Two players, A and B, occupy distinct squares of the grid and can choose
one of 5 actions on each turn: N , S, E, W , or stand , corresponding to wanting to move
one square towards the specified direction (North, South, East, West) or standing still in the
current position. Once both players have selected their actions, the two moves are executed in
random order (uniformly). Initially players are positioned as shown in the figure. The circle
in the figures represents the ball possession. Initially possession of the ball goes to one or the
other player at random (uniformly). When a player with the ball steps into its opponent’s
goal area (left for A, right for B), that player wins the game and receives +1 reward (while its
opponent receives −1). Then, the game is reset to the initial configuration, and ball possession
is reassigned randomly to one of the players.
When a player executes an action that would take it to the square occupied by the other
player, possession of the ball goes to the latter player and the move does not take place. A
good defensive maneuver, then, is to stand where the player with the ball wants to go. 
For example, if the two players are one in front of the other and they both choose to go towards
the other, none of them will actually move and ball possession will shift to (or stay with) the
one randomly extracted to move first, since it will stop the other’s movement during the second
action, taking (or keeping) the ball.

<p align="center">
  <img src="https://github.com/user-attachments/assets/4397b224-765c-4648-92f1-37c063f9aa6a" 
       alt="MAS Football" 
       width="400"/>
</p>

#### Objectives
 - Implement the belief-based joint action learning algorithm (explained in class), with be-
havioural strategies. As belief function you can use the one mentioned in class (frequency
of the opponent’s actions in each state) and/or experiment your own.
 - Use the implemented algorithm to learn behavioural strategies for the two-player zero-sum
stochastic game representing the simplified version of football by [Lit94]. As in the paper,
the learning must be performed against a random opponent, and against another learner
of identical design, for 106 steps and parameters:

$$
\begin{aligned}
 \epsilon & = 0.2\\
 \alpha_0 & = 1.0 \\
 \alpha_{t+1} & = \alpha_t \cdot 10^{\log_{10}(0.01)/10^6} \approx \alpha_t \cdot 0.9999954\\
 \gamma & = 0.9
\end{aligned}
$$

 -  Test, for 105 steps, the obtained strategies against a random opponent and against each
other, counting the total number of games actually finished and the % of games won.
To emulate the discount factor, at every step the game can terminate immediately with
probability 1 − γ = 0.1, resulting in a draw (0 reward to both players) and resetting the
game. Then discuss your findings, and try to compare them with those in Littman[^Lit94], reported
below, even though the latter were also tested against Q-learned best response strategies.

<div align="center">

| Method                             | % won vs. random | Games vs. random | % won vs. Q-best | Games vs. Q-best |
|------------------------------------|-----------------:|-----------------:|-----------------:|-----------------:|
| minimax-Q (random learning)        |            99.3% |            6500  |            35.0% |            4300  |
| minimax-Q (identical learning)     |            99.3% |            7200  |            37.5% |            4400  |
| independent-Q (random learning)    |            99.4% |           11300  |             0.0% |            5500  |
| independent-Q (identical learning) |            99.5% |            8600  |             0.0% |            1200  |

</div>

---

Projects were chosen from...


[^Lit94]: [M. L. Littman (1994) Markov games as a framework for multi-agent reinforcement learning. Proceedings of ICML’94](Multi-Agent Reinforcement Learning on Stochastic Game/littman94markov.pdf)
