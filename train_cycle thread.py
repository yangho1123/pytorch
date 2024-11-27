from self_play import self_play
from train_network import train_network
from evaluate_network import evaluate_network
from train_network_flag import train_network as train_no_his
import time
import os
import threading

#evaluate_best_player()
#dual_network()

def self_play_wrapper(i):
    print('SelfPlay', i, '=================')
    start_time = time.time()
    self_play()  
    print("self_play() time: {:.2f} seconds".format(time.time() - start_time))

def train_and_evaluate(file_path):
    if os.path.exists(file_path):  # 如果file存在就不跑歷史資料
        start_time = time.time()
        train_no_his()
        print("train_no_his() time: {:.2f} seconds".format(time.time() - start_time))
        start_time = time.time()
        update_best_player = evaluate_network()  # 這個函式會決定要不要保留file.txt
        print("evaluate_network() time: {:.2f} seconds".format(time.time() - start_time))
    else:  # 如果file不在就跑歷史資料
        start_time = time.time()
        train_network()
        print("train_network() time: {:.2f} seconds".format(time.time() - start_time))
        start_time = time.time()
        update_best_player = evaluate_network()  # 這個函式會決定要不要保留file.txt
        print("evaluate_network() time: {:.2f} seconds".format(time.time() - start_time))

# 直接讀這次產生的多筆資料
def train():
    start_time = time.time()
    train_network()
    print("train_network() time: {:.2f} seconds".format(time.time() - start_time))
    start_time = time.time()
    update_best_player = evaluate_network()  # 這個函式會決定要不要保留file.txt
    print("evaluate_network() time: {:.2f} seconds".format(time.time() - start_time))

if __name__ == '__main__':
    loop_start_time = time.time()  # 迴圈開始的時間
    for i in range(1):
        threads = []
        for i in range(1, 4):
            thread = threading.Thread(target=self_play_wrapper, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
        print("所有執行緒都已完成")
        # 多執行緒 self_play 結束後，執行訓練和評估
        file_path = r'C:\Users\user\AlphaZero_othello\torchdata\file.txt'
        train()

    loop_end_time = time.time()  # 迴圈結束的時間
    
    print("Total loop time: {:.2f} seconds".format(loop_end_time - loop_start_time))