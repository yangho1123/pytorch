from self_play_3his_1value_test import self_play
from train_network_3his import train_network
from evaluate_network_3his import evaluate_network_multiprocess, update_best_player, EN_GAME_COUNT
from load_data import load_data, prepare_data, write_prepared_data
import time
import multiprocessing

train_epochs = 1

def self_play_wrapper(i):
    print('Train cycle', i, '=================')
    start_time = time.time()
    self_play()  
    print("self_play() time: {:.2f} seconds".format(time.time() - start_time))

def prepare_and_write():
    history = load_data(4)   #載入歷史資料(筆數)
    prepared_history = prepare_data(history)    # 重塑歷史資料
    write_prepared_data(prepared_history)   # 寫入重塑好的歷史資料檔案

def train():
    start_time = time.time()
    train_network()
    print("train_network() time: {:.2f} seconds".format(time.time() - start_time))

if __name__ == '__main__':
    loop_start_time = time.time()  # 迴圈開始的時間
    loop_times = []
    update_times = 0
    for i in range(train_epochs):
        epoch_start_time = time.time()
        print("=================")
        print("train cycle: ", i)
        print("=================")
        # num_processes = 5  # 決定你想要運行多少個進程
        # processes = []
        # for j in range(num_processes):
        #     p = multiprocessing.Process(target=self_play_wrapper, args=(j,))
        #     p.start()
        #     processes.append(p)

        # for p in processes:
        #     p.join()
        prepare_and_write()
        # 多進程 self_play 結束後，執行訓練        
        train()
        # 多進程執行evaluate
        evaluate_processes = 4
        _, _, _, _, update = evaluate_network_multiprocess(evaluate_processes)
        epoch_end_time = time.time()  # 當前循環的結束時間
        epoch_time = epoch_end_time - epoch_start_time
        loop_times.append(epoch_time)
        update_times = update_times + update    # 累計更新次數
        # update_best_player()
        print(f"this cycle {i} time: {epoch_time:.2f} seconds")

    loop_end_time = time.time()  # 迴圈結束的時間
    total_time = loop_end_time - loop_start_time
    average_time = sum(loop_times) / len(loop_times)
    print("=================")
    print(f"total time: {total_time:.2f} seconds")
    print(f"average time: {average_time:.2f} seconds ( {train_epochs} cycle)")
    print("update times: ", update_times)
    print("=================")