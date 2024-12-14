import os
import copy
import math
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm


import simulate


## Config ## 
GAME = "minigo"
N_ITERS = 5 # Number of iterations to perform
K = 32 # Initial K-factor for updating Elo ratings
DECAY_RATE = 0.2
n_agents = 5


## Automated ##
game_path = f"./games/{GAME}.py"
checkpoints = [3200 * i for i in range(1, 5+1)]
weight_paths = [f"./weights/{GAME}/{x}.checkpoint" for x in checkpoints]
print("Using Paths:", weight_paths)


def calculate_expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_ratings(rating_a, rating_b, score_a, k):
    expected_a = calculate_expected_score(rating_a, rating_b)
    expected_b = 1 - expected_a
    new_rating_a = rating_a + k * (score_a - expected_a)
    new_rating_b = rating_b + k * ((1 - score_a) - expected_b)
    return new_rating_a, new_rating_b


def simulate_game(a_path, b_path):
    return simulate.simulate_game(GAME, a_path, b_path)


def plot_rating_history(history):
    plt.figure(figsize=(12, 6))
    for i in range(N_ITERS - 1):
        ratings = history[i]
        plt.plot(
            range(len(ratings)), 
            ratings, 
            color='lightblue', 
            marker='o', 
        )
    final_ratings = history[N_ITERS - 1]
    plt.plot(
        range(len(final_ratings)), 
        final_ratings, 
        color='darkblue', 
        marker='o', 
        markersize=10,
    )
    # Customize the plot
    plt.title('{GAME} Agent Elo Ratings Over Iterations', fontsize=15)
    plt.xlabel('Iterations', fontsize=12)
    # Set custom x-axis
    plt.xticks(range(len(checkpoints)), checkpoints, rotation=45)
    plt.ylabel('Elo Rating', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    return;


def main():
    
    # Ratings
    ratings = [1200 for _ in range(n_agents)]

    history = [copy.copy(ratings)]

    for i in range(N_ITERS):
        kn = K * (math.e ** (-1 * DECAY_RATE * i))
        # Pit all agents against each other
        for p1 in range(n_agents):
            for p2 in range(p1+1, n_agents):
                score = simulate_game(weight_paths[p1], weight_paths[p2])
                r1, r2 = update_ratings(ratings[p1], ratings[p2], score, kn)
                ratings[p1] = r1
                ratings[p2] = r2
        history.append(copy.copy(ratings))
        print(f"Ratings for iteration {i}:", ratings)

    print("Final ratings: ", ratings)
    plot_rating_history(history)


if __name__ == "__main__":
    main()
