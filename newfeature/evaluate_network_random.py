from game_nogroup import State
from game import random_action
from pv_mcts_3his_fast import pv_mcts_action
import torch
import time
import multiprocessing
import csv
import matplotlib.pyplot as plt
import os

EN_GAME_COUNT = 15
EN_TEMPERATURE = 0

def calculate_points(state):    
    return state.finish()

def play(next_actions):
    state = State()
    while not state.is_done():
        current_player = state.get_player_order()
        action = next_actions[current_player](state)
        state = state.next(action)
    return calculate_points(state)

def evaluate_network(model_path, index, wins_latest, wins_random1, wins_random2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_latest = torch.jit.load(model_path).to(device)
    next_action_latest = pv_mcts_action(model_latest, EN_TEMPERATURE)

    win_count_latest = 0
    win_count_random1 = 0
    win_count_random2 = 0

    for i in range(EN_GAME_COUNT):
        if i % 3 == 0:
            results = play([next_action_latest, random_action, random_action])
        elif i % 3 == 1:
            results = play([random_action, next_action_latest, random_action])
        else:
            results = play([random_action, random_action, next_action_latest])

        if results[0] == 1:
            win_count_latest += 1
        elif results[1] == 1:
            win_count_random1 += 1
        elif results[2] == 1:
            win_count_random2 += 1

        print(f'\rEvaluate {i+1}/{EN_GAME_COUNT}', end='')

    wins_latest[index] = win_count_latest
    wins_random1[index] = win_count_random1
    wins_random2[index] = win_count_random2
    del model_latest
    torch.cuda.empty_cache()

def evaluate_model(model_path, num_processes):
    print(f"\nEvaluating model: {model_path}")
    manager = multiprocessing.Manager()
    wins_latest = manager.dict()
    wins_random1 = manager.dict()
    wins_random2 = manager.dict()
    jobs = []
    start = time.time()

    for i in range(1, num_processes + 1):
        p = multiprocessing.Process(target=evaluate_network, args=(model_path, i, wins_latest, wins_random1, wins_random2))
        jobs.append(p)
        p.start()
        print(f"Process{i} start!")

    for proc in jobs:
        proc.join()

    end = time.time()
    total_games = EN_GAME_COUNT * num_processes
    total_wins_latest = sum(wins_latest.values())
    win_rate = total_wins_latest / total_games

    print(f"\nTotal Games: {total_games}")
    print(f"Latest Wins: {total_wins_latest}, Win Rate: {win_rate:.2%}")
    print(f"Evaluation time: {end - start:.2f} seconds")

    return model_path, total_wins_latest, total_games, win_rate

def save_results_to_csv(results, filename='evaluation_results_1600.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model', 'Latest Wins', 'Total Games', 'Win Rate'])
        for row in results:
            writer.writerow(row)

def plot_results(results, output_path='win_rate_plot_1600.png'):
    x = list(range(1, len(results) + 1))  # 1, 2, 3, ...
    win_rates = [r[3] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(x, win_rates, marker='.', linestyle='-', color='royalblue')
    plt.ylim(0, 1)
    plt.xticks(x)  # X 軸為整數 1~N
    plt.xlabel('Iteration')
    plt.ylabel('Win rate')
    plt.title('RL vs Random')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Win rate plot saved to {output_path}")

if __name__ == '__main__':
    num_processes = 8
    model_paths = [
        '../model/250326/22layers/best_val.pt',
        '../model/250419/22layers/best_val.pt',
        '../model/250430/22layers/best_val.pt',
        '../model/250505/22layers/best_val.pt',
        '../model/250505/iter2/best_val.pt',
        '../model/250521/best_val.pt'
    ]

    all_results = []
    for path in model_paths:
        result = evaluate_model(path, num_processes)
        all_results.append(result)

    save_results_to_csv(all_results)
    plot_results(all_results)
