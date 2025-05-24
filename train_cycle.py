from self_play import self_play
from train_network import train_network
from evaluate_network import evaluate_network, update_best_player, EN_GAME_COUNT
import time
import multiprocessing

train_epochs = 1

def self_play_wrapper(i):
    print('Train cycle', i, '=================')
    start_time = time.time()
    self_play()  
    print("self_play() time: {:.2f} seconds".format(time.time() - start_time))

def train():
    start_time = time.time()
    train_network()
    print("train_network() time: {:.2f} seconds".format(time.time() - start_time))

if __name__ == '__main__':
    loop_start_time = time.time()  # 迴圈開始的時間
    for i in range(train_epochs):
        num_processes = 5  # 決定你想要運行多少個進程
        processes = []
        for i in range(num_processes):
            p = multiprocessing.Process(target=self_play_wrapper, args=(i,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # 多進程 self_play 結束後，執行訓練        
        # train()
        # # 多進程執行evaluate
        # manager = multiprocessing.Manager()
        # points_latest = manager.dict()
        # points_best = manager.dict()
        # no1_latest = manager.dict()
        # no1_best = manager.dict()
        # jobs = []
        # start = time.time()
        # for i in range(1, 5):
        #     p = multiprocessing.Process(target = evaluate_network, args=(i, points_latest, points_best, no1_latest, no1_best))
        #     jobs.append(p)
        #     p.start()
        #     print(f"Process{i} start!")
        # for proc in jobs:
        #     proc.join()
        # end = time.time()
        # total_game = EN_GAME_COUNT * 4
        # total_point0 = 0
        # total_point1 = 0
        # total_no1_0 = 0
        # total_no1_1 = 0
        # for i in range(1,5):
        #     total_point0 += points_latest.get(i)
        #     total_point1 += points_best.get(i)
        #     total_no1_0 += no1_latest.get(i)
        #     total_no1_1 += no1_best.get(i)
        # print(f"score: latest({total_point0}), no.1({total_no1_0}) vs best({total_point1}), no.1({total_no1_1})")
        # if total_point0 > total_point1:
        #     update_best_player()
        # print("evaluate time: %f sec" % (end - start))

    loop_end_time = time.time()  # 迴圈結束的時間
    print("Total loop time: {:.2f} seconds".format(loop_end_time - loop_start_time))