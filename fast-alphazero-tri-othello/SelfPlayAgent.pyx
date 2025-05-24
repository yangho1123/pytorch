# cython: language_level=3
import numpy as np
import torch
import torch.multiprocessing as mp
import time

from MCTS import MCTS

class SelfPlayAgent(mp.Process):
    """
    自我對弈代理類，使用多進程並行生成訓練數據。
    """

    def __init__(self, id, game, ready_queue, batch_ready, batch_tensor, policy_tensor, value_tensor, output_queue,
                result_queue, complete_count, games_played, args):
        """
        初始化自我對弈代理。
        
        參數:
            id: 代理ID
            game: 遊戲類別實例
            ready_queue: 準備好的批次隊列
            batch_ready: 批次準備就緒的事件
            batch_tensor: 用於神經網絡輸入的張量
            policy_tensor: 用於存儲策略輸出的張量
            value_tensor: 用於存儲價值輸出的張量
            output_queue: 輸出訓練樣本的隊列
            result_queue: 遊戲結果的隊列
            complete_count: 完成的進程計數
            games_played: 已完成的遊戲數量
            args: 配置參數
        """
        super().__init__()
        self.id = id
        self.game = game
        self.ready_queue = ready_queue
        self.batch_ready = batch_ready
        self.batch_tensor = batch_tensor
        self.batch_size = self.batch_tensor.shape[0]
        self.policy_tensor = policy_tensor
        self.value_tensor = value_tensor
        self.output_queue = output_queue
        self.result_queue = result_queue
        self.games = []  # 當前進行中的遊戲
        self.canonical = []  # 標準形式的遊戲狀態
        self.histories = []  # 遊戲歷史
        self.player = []  # 當前玩家
        self.turn = []  # 當前回合
        self.mcts = []  # MCTS搜索樹
        self.games_played = games_played
        self.complete_count = complete_count
        self.args = args
        self.valid = torch.zeros_like(self.policy_tensor)
        self.fast = False  # 是否進行快速模擬
        
        # 初始化所有並行遊戲
        for _ in range(self.batch_size):
            self.games.append(self.game.getInitBoard())
            self.histories.append([])
            self.player.append(1)
            self.turn.append(1)
            self.mcts.append(MCTS(self.game, None, self.args))
            self.canonical.append(None)

    def run(self):
        """
        進程的主要執行函數。
        """
        np.random.seed()
        #print(f"工作進程 {self.id} 開始執行")
        
        while self.games_played.value < self.args.gamesPerIteration:
            # 生成標準形式的遊戲狀態
            self.generateCanonical()
            
            # 決定是否使用快速模擬
            self.fast = np.random.random_sample() < self.args.probFastSim
            
            # 執行MCTS搜索
            if self.fast:
                for i in range(self.args.numFastSims):
                    self.generateBatch()
                    self.processBatch()
            else:
                for i in range(self.args.numMCTSSims):
                    self.generateBatch()
                    self.processBatch()
            
            # 執行一步移動
            self.playMoves()
            
            # 每完成一定数量的游戏后清理MCTS树
            if self.games_played.value % 10 == 0:
                for i in range(self.batch_size):
                    self.mcts[i] = MCTS(self.game, None, self.args)
            
            # 打印当前进度
            #print(f"工作進程 {self.id} - 當前已完成遊戲數: {self.games_played.value}/{self.args.gamesPerIteration}")
        
        print(f"工作進程 {self.id} 完成所有遊戲，共完成 {self.games_played.value} 場遊戲")
        
        # 更新完成計數
        with self.complete_count.get_lock():
            self.complete_count.value += 1
        
        # 關閉輸出隊列
        self.output_queue.close()
        self.output_queue.join_thread()

    def generateBatch(self):
        """
        為批處理生成輸入。
        找到需要評估的葉節點，將其添加到批次中。
        """
        for i in range(self.batch_size):
            board = self.mcts[i].findLeafToProcess(self.canonical[i], True)
            if board is not None:
                self.batch_tensor[i] = self.prepare_input(board)
        self.ready_queue.put(self.id)

    def processBatch(self):
        """
        處理神經網絡的預測結果，更新MCTS樹。
        """
        self.batch_ready.wait()
        self.batch_ready.clear()
        # 將預測結果應用於MCTS樹
        for i in range(self.batch_size):
            policy = self.policy_tensor[i].data.numpy()
            #self.mcts[i].processResults(
            #    self.policy_tensor[i].data.numpy(), self.value_tensor[i][0])
            self.mcts[i].processResults(policy, self.value_tensor[i][0])

    def playMoves(self):    #這個方法可能有問題
        """
        根據MCTS搜索結果選擇動作，並更新遊戲狀態。
        """
        for i in range(self.batch_size):
            # 根據溫度參數確定是否使用確定性策略
            temp = int(self.turn[i] < self.args.tempThreshold)
            # 根據MCTS訪問次數選擇動作
            policy = self.mcts[i].getExpertProb(
                self.canonical[i], temp, not self.fast)

            action = np.random.choice(len(policy), p=policy)
            
            # 如果不是快速模擬，記錄遊戲歷史（用於訓練）
            if not self.fast:
                self.histories[i].append((self.canonical[i], self.mcts[i].getExpertProb(self.canonical[i], prune=True),
                                         self.mcts[i].getExpertValue(self.canonical[i]), self.player[i]))
            
            # 執行選擇的動作，獲取新的遊戲狀態
            self.games[i], self.player[i] = self.game.getNextState(self.games[i], self.player[i], action)
            self.turn[i] += 1
            
            winner = self.game.getGameEnded(self.games[i], 1)
            #需檢查是否能進入這個if
            if winner != 0:
                self.result_queue.put(winner)
                lock = self.games_played.get_lock()
                lock.acquire()
                if self.games_played.value < self.args.gamesPerIteration:
                    self.games_played.value += 1
                    lock.release()
                    for hist in self.histories[i]:
                        if self.args.symmetricSamples:
                            sym = self.game.getSymmetries(hist[0], hist[1])
                            for b, p in sym:
                                self.output_queue.put((b, p,
                                                       winner *
                                                       hist[3] *
                                                       (1 - self.args.expertValueWeight.current)
                                                       + self.args.expertValueWeight.current * hist[2]))
                        else:
                            self.output_queue.put((hist[0], hist[1],
                                                   winner *
                                                   hist[3] *
                                                   (1 - self.args.expertValueWeight.current)
                                                   + self.args.expertValueWeight.current * hist[2]))
                    self.games[i] = self.game.getInitBoard()
                    self.histories[i] = []
                    self.player[i] = 1
                    self.turn[i] = 1
                    self.mcts[i] = MCTS(self.game, None, self.args)
                else:
                    lock.release()

    def generateCanonical(self):
        """
        為所有當前遊戲生成標準形式的遊戲狀態。
        """
        for i in range(self.batch_size):
            self.canonical[i] = self.game.getCanonicalForm(
                self.games[i], self.player[i])
                
    def prepare_input(self, board):
        """
        將遊戲狀態轉換為神經網絡輸入格式。
        
        轉換為固定的紅-綠-藍通道順序，以便神經網絡始終接收一致的輸入格式。
        """
        try:
            if board is None:
                print("錯誤: prepare_input 收到 None 棋盤")
                # 返回全零張量
                return torch.zeros(4, 11, 11)
            
            if not isinstance(board, np.ndarray):
                print(f"警告: 輸入棋盤類型不是numpy數組，而是 {type(board)}")
                # 嘗試轉換
                board = np.array(board)
            
            # 檢查形狀
            if board.shape[0] < 4:
                print(f"錯誤: 棋盤形狀不符合預期，實際形狀: {board.shape}")
                # 返回全零張量
                return torch.zeros(4, 11, 11)
            
            # 獲取當前玩家順序
            player_order = int(board[3, 0, 0] * 2 + 0.5)  # 0:紅, 1:綠, 2:藍
            
            # 創建標準化的輸入張量（4通道）
            standard_board = np.zeros((4, 11, 11), dtype=np.float32)
            
            # 根據當前玩家順序重新排列通道0-2
            if player_order == 0:  # 紅方回合
                # 標準順序：[紅,綠,藍]，當前順序：[紅,綠,藍]
                standard_board[0:3, :, :] = board[0:3, :, :]
            elif player_order == 1:  # 綠方回合
                # 標準順序：[紅,綠,藍]，當前順序：[綠,藍,紅]
                standard_board[0, :, :] = board[2, :, :]  # 紅方 (位於通道2)
                standard_board[1, :, :] = board[0, :, :]  # 綠方 (位於通道0)
                standard_board[2, :, :] = board[1, :, :]  # 藍方 (位於通道1)
            else:  # 藍方回合
                # 標準順序：[紅,綠,藍]，當前順序：[藍,紅,綠]
                standard_board[0, :, :] = board[1, :, :]  # 紅方 (位於通道1)
                standard_board[1, :, :] = board[2, :, :]  # 綠方 (位於通道2)
                standard_board[2, :, :] = board[0, :, :]  # 藍方 (位於通道0)
            
            # 保留玩家順序標記（通道3）
            standard_board[3, :, :] = board[3, :, :]
            
            return torch.FloatTensor(standard_board)
        except Exception as e:
            print(f"prepare_input 錯誤: {e}")
            # 返回全零張量
            return torch.zeros(4, 11, 11) 