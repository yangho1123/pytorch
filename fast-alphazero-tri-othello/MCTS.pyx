# cython: language_level=3
# cython: linetrace=True
# cython: profile=True
# cython: binding=True

import math
import numpy as np
cimport numpy as np

EPS = 1e-8

cdef class Node:
    """
    蒙特卡洛樹搜索的節點類，跟踪訪問次數和Q值
    """
    
    cdef public:
        object game_state  # 遊戲狀態
        int player  # 玩家編號
        list children  # 子節點列表
        dict valid_moves  # 有效移動字典
        dict num_visits  # 訪問次數
        dict q_values  # Q值
        dict p_values  # 先驗概率
        int is_expanded  # 是否已展開
        int is_terminal  # 是否為終端節點
        float value  # 節點價值

    def __init__(self, game_state, player):
        self.game_state = game_state
        self.player = player
        self.children = []
        self.valid_moves = {}
        self.num_visits = {}
        self.q_values = {}
        self.p_values = {}
        self.is_expanded = 0
        self.is_terminal = 0
        self.value = 0.0

class MCTS:
    """
    使用神經網絡的蒙特卡洛樹搜索算法
    """

    def __init__(self, game, net, args):
        self.game = game
        self.net = net
        self.args = args
        self.nodes = {}  # 緩存已訪問的狀態
        self.Qsa = {}    # 存儲Q值 (s,a)
        self.Nsa = {}    # 存儲訪問次數 (s,a)
        self.Ns = {}     # 存儲節點訪問次數 s
        self.Ps = {}     # 存儲策略 s
        self.Es = {}     # 存儲遊戲結束標誌 s
        self.Vs = {}     # 存儲有效移動 s

    def getActionProb(self, canonical_board, temp=1):
        """
        從當前棋盤狀態為當前玩家生成移動概率。
        
        參數:
            canonical_board: 標準形式的遊戲狀態
            temp: 溫度參數，控制探索/利用的權衡。temp=1 (探索) 和 temp=0 (利用)
            
        返回:
            probs: 移動概率的列表
        """
        for i in range(self.args.numMCTSSims):
            self._search(canonical_board)

        s = self._stringRepresentation(canonical_board)
        counts = [self.Nsa.get((s, a), 0) for a in range(self.game.getActionSize())]

        if temp == 0:
            # 選擇訪問次數最多的動作
            bestA = np.argmax(counts)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        # 應用溫度
        counts = [x ** (1. / temp) for x in counts]
        # 歸一化
        total_counts = float(sum(counts))
        if total_counts == 0:
            # 如果所有動作都沒有被訪問過，返回均勻分布
            return [1/len(counts) for _ in counts]
        probs = [x / total_counts for x in counts]
        return probs

    def findLeafToProcess(self, canonical_board, update_nodes=True):
        """
        找到需要處理的葉節點。
        
        參數:
            canonical_board: 標準形式的遊戲狀態
            update_nodes: 是否更新節點信息
            
        返回:
            leaf: 葉節點
        """
        s = self._stringRepresentation(canonical_board)
        
        #print(f"嘗試尋找葉節點，棋盤狀態: {s[:20]}...")

        if s not in self.nodes:
            # 新節點，創建一個根節點
            #print(f"創建新的根節點: {s[:20]}...")
            self.nodes[s] = Node(canonical_board, 1)
            # 新增：直接返回這個新節點，因為它肯定是葉節點
            if update_nodes:
                self.nodes[s].is_expanded = 1
            return canonical_board

        # 如果節點已經被標記為終端節點，直接返回
        if self.nodes[s].is_terminal:
            #print(f"節點已是終端節點: {s[:20]}...")
            return None

        # 如果節點尚未展開，返回該節點
        if not self.nodes[s].is_expanded:
           # print(f"找到未展開的節點: {s[:20]}...")
            if update_nodes:
                self.nodes[s].is_expanded = 1
            return canonical_board

        # 尋找最佳子節點前，確保該節點有合法移動和有效的策略值
        current_node = self.nodes[s]
        if not current_node.valid_moves:
            #print(f"節點沒有有效移動記錄，獲取有效移動: {s[:20]}...")
            valids = self.game.getValidMoves(current_node.game_state, current_node.player)
            valid_count = sum(valids)
            
            if valid_count == 0:
                #print(f"該節點沒有合法移動，標記為終端節點: {s[:20]}...")
                current_node.is_terminal = 1
                return None
            
            # 為節點設置合法移動
            for a in range(len(valids)):
                if valids[a]:
                    current_node.valid_moves[a] = 1
                    # 初始化該移動的先驗概率為均勻值
                    current_node.p_values[a] = 1.0 / valid_count
        
        # 檢查節點是否有有效移動
        if not current_node.valid_moves:
            print(f"節點設置有效移動後仍為空，異常情況: {s[:20]}...")
            return None

        # 選擇最佳子節點，根據UCB公式
        best_ucb = -float('inf')
        best_action = -1

        for a in range(self.game.getActionSize()):
            if a in current_node.valid_moves:
                # 計算UCB值
                if (s, a) in self.Nsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * current_node.p_values[a] * \
                        math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * current_node.p_values[a] * math.sqrt(self.Ns[s] + 1e-8)

                if u > best_ucb:
                    best_ucb = u
                    best_action = a

        # 如果沒有找到最佳行動（不應該發生，但為安全起見）
        if best_action == -1:
            print(f"無法找到最佳行動，異常情況: {s[:20]}...")
            valid_actions = [a for a in current_node.valid_moves]
            if valid_actions:
                best_action = valid_actions[0]
            else:
                current_node.is_terminal = 1
                return None

        #print(f"選擇行動 {best_action} 進行樹搜索")
        
        # 獲取下一個狀態
        next_s, next_player = self.game.getNextState(current_node.game_state, current_node.player, best_action)
        next_canon = self.game.getCanonicalForm(next_s, next_player)
        next_s_rep = self._stringRepresentation(next_canon)

        # 檢查是否已經訪問過該狀態
        if next_s_rep not in self.nodes:
            # print(f"找到新的葉節點: {next_s_rep[:20]}...")
            self.nodes[next_s_rep] = Node(next_canon, next_player)
            return next_canon
        elif not self.nodes[next_s_rep].is_expanded:
            # print(f"找到已存在但未展開的葉節點: {next_s_rep[:20]}...")
            if update_nodes:
                self.nodes[next_s_rep].is_expanded = 1
            return next_canon

        # 如果下一個狀態已經展開，遞歸尋找葉節點
        return self.findLeafToProcess(next_canon, update_nodes)

    def processResults(self, pi, v):
        """
        使用神經網絡的輸出更新MCTS樹。
        
        參數:
            pi: 策略輸出（動作概率）
            v: 價值輸出（期望獲勝概率）
        """
        if not self.nodes:
            return
            
        # 更新最後處理的節點
        for s, node in list(self.nodes.items()):
            if node.is_expanded and not node.valid_moves:
                # 獲取有效移動
                valids = self.game.getValidMoves(node.game_state, node.player)
                # 策略總和印出0
                #print(f"有效移動數量: {sum(valids)}")
                #print(f"策略形狀: {pi.shape}")
                #print(f"策略總和: {sum(pi)}")
            
                # 如果沒有有效移動，標記為終端節點
                if sum(valids) == 0:
                    node.is_terminal = 1
                    continue
                    
                # 更新策略（先驗概率）
                ps = pi
                
                # 遮罩無效移動
                ps = ps * valids
                sum_ps = sum(ps)
                if sum_ps > 0:
                    ps /= sum_ps  # 重新歸一化
                else:
                    # 若所有移動都無效（不應該發生），使用均勻分布
                    print("All valid moves were masked, using uniform distribution")
                    # 修正: 使用 NumPy 數組進行除法
                    ps = np.array(valids, dtype=np.float32)
                    total = sum(ps)
                    if total > 0:  # 確保總和不為零
                        ps = ps / total
                
                # 存儲策略和節點訪問計數
                self.Ps[s] = ps
                self.Ns[s] = 0
                
                # 設置節點的策略值和有效移動
                for a in range(len(ps)):
                    if valids[a]:
                        node.valid_moves[a] = 1
                        node.p_values[a] = ps[a]
                        
                node.value = v

    def _search(self, canonical_board):
        """
        執行MCTS搜索以更新節點統計信息。
        
        參數:
            canonical_board: 標準形式的遊戲狀態
            
        返回:
            v: 該節點的價值
        """
        s = self._stringRepresentation(canonical_board)
        
        # 如果節點未評估或未展開，使用神經網絡進行評估
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonical_board, 1)
        if self.Es[s] != 0:
            # 終局，返回獲勝結果
            return -self.Es[s]
            
        if s not in self.Ps:
            # 葉節點，使用神經網絡預測
            self.Ps[s], v = self.net.predict(canonical_board)
            # 確保 self.Vs[s] 是 NumPy 數組
            valids = self.game.getValidMoves(canonical_board, 1)
            self.Vs[s] = np.array(valids, dtype=np.float32)
            
            # 遮罩無效移動
            self.Ps[s] = self.Ps[s] * self.Vs[s]
            sum_Ps = np.sum(self.Ps[s])
            if sum_Ps > 0:
                self.Ps[s] /= sum_Ps  # 重新歸一化
            else:
                # 所有移動都不合法，使用均勻分布
                print("所有移動都被遮罩，使用均勻分布")
                total = np.sum(self.Vs[s])
                if total > 0:  # 確保總和不為零
                    self.Ps[s] = self.Vs[s] / total
                else:
                    # 如果沒有合法移動，將第一個位置設為1
                    self.Ps[s] = np.zeros_like(self.Vs[s])
                    if len(self.Ps[s]) > 0:
                        self.Ps[s][0] = 1.0
            
            self.Ns[s] = 0
            return -v
            
        # 根據UCB選擇最佳動作
        best_a = self._select_action(s)
        
        # 檢查選擇的動作是否有效
        if best_a == -1:
            print(f"警告：在_search中無法選擇有效動作，狀態: {s[:20]}...")
            # 如果無法選擇有效動作，返回0
            return 0
        
        # 執行動作
        next_s, next_player = self.game.getNextState(canonical_board, 1, best_a)
        next_canonical = self.game.getCanonicalForm(next_s, next_player)
        
        # 遞歸搜索
        v = self._search(next_canonical)
        
        # 更新統計信息
        if (s, best_a) in self.Qsa:
            self.Qsa[(s, best_a)] = (self.Nsa[(s, best_a)] * self.Qsa[(s, best_a)] + v) / (self.Nsa[(s, best_a)] + 1)
            self.Nsa[(s, best_a)] += 1
        else:
            self.Qsa[(s, best_a)] = v
            self.Nsa[(s, best_a)] = 1
            
        self.Ns[s] += 1
        return -v  # 返回負值，因為我們需要換位視角
        
    def _select_action(self, s):
        """
        根據UCB公式選擇最佳動作。
        
        參數:
            s: 狀態表示
            
        返回:
            a: 選擇的動作
        """
        best_ucb = -float('inf')
        best_a = -1
        
        # 檢查是否有有效移動
        if s not in self.Vs or np.sum(self.Vs[s]) == 0:
            print(f"警告：狀態 {s[:20]} 沒有有效移動")
            return -1
        
        # 遍歷所有有效動作
        for a in range(len(self.Ps[s])):
            if self.Vs[s][a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * \
                        math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)
                
                if u > best_ucb:
                    best_ucb = u
                    best_a = a
        
        # 如果沒有找到最佳動作，隨機選擇一個有效動作
        if best_a == -1:
            valid_moves = [i for i, x in enumerate(self.Vs[s]) if x > 0]
            if valid_moves:
                best_a = np.random.choice(valid_moves)
        
        return best_a
        
    def getExpertProb(self, canonical_board, temp=0, prune=False):
        """
        返回根據MCTS訪問次數的動作概率。
        
        參數:
            canonical_board: 標準形式的遊戲狀態
            temp: 溫度參數
            prune: 是否修剪訪問次數少的動作
            
        返回:
            probs: 移動概率的列表
        """
        s = self._stringRepresentation(canonical_board)
        
        if s not in self.nodes:
            # 如果節點不在樹中，返回均勻分布
            valids = self.game.getValidMoves(canonical_board, 1)
            total = sum(valids)
            if total <= 0:  # 避免零除錯誤
                # 如果沒有有效移動，創建一個單位向量（第一個位置為1）
                probs = [0] * len(valids)
                if len(probs) > 0:
                    probs[0] = 1.0
                return probs
            # 創建均勻分佈，確保總和為1
            probs = [float(x) / total for x in valids]
            # 額外確保總和為1（防止浮點數誤差）
            total_prob = sum(probs)
            if total_prob > 0 and abs(total_prob - 1.0) > 1e-10:
                probs = [p / total_prob for p in probs]
            return probs
            
        counts = np.zeros(self.game.getActionSize())
        node = self.nodes[s]
        
        # 完全重新獲取有效移動
        if not node.valid_moves:
            # 如果節點沒有有效移動（剛被創建），返回均勻分布
            valids = self.game.getValidMoves(canonical_board, 1)
            total = sum(valids)
            if total <= 0:  # 避免零除錯誤
                # 如果沒有有效移動，創建一個單位向量
                probs = [0] * len(valids)
                if len(probs) > 0:
                    probs[0] = 1.0
                return probs
            # 創建均勻分佈
            probs = [float(x) / total for x in valids]
            # 確保總和為1
            total_prob = sum(probs)
            if total_prob > 0 and abs(total_prob - 1.0) > 1e-10:
                probs = [p / total_prob for p in probs]
            return probs
            
        # 直接使用getValidMoves獲取有效移動，而不依賴node.valid_moves
        valids = self.game.getValidMoves(canonical_board, 1)
            
        # 收集所有動作的訪問次數
        for a in range(self.game.getActionSize()):
            # 使用valids數組而不是node.valid_moves字典
            if valids[a] and (s, a) in self.Nsa:
                counts[a] = self.Nsa[(s, a)]
                
        if temp == 0:
            # 選擇訪問次數最多的動作
            bestA = np.argmax(counts)
            probs = [0] * len(counts)
            probs[bestA] = 1.0
            return probs
            
        # 使用溫度參數來調整概率分布
        counts = np.array([x ** (1. / temp) if x > 0 else 0 for x in counts])
        total = np.sum(counts)
        
        # 計算概率
        if total <= 0:  # 避免零除錯誤
            # 返回基於有效移動的均勻分佈
            total = np.sum(valids)  # 使用valids數組
            if total <= 0:
                # 如果沒有有效移動，創建一個單位向量
                probs = [0] * len(valids)
                if len(probs) > 0:
                    probs[0] = 1.0
                return probs
            probs = np.array(valids, dtype=np.float32) / total
            return probs.tolist()
            
        probs = counts / total
        
        if prune:
            # 只保留前N個訪問次數最高的動作
            sorted_indices = np.argsort(probs)[::-1]
            threshold_idx = min(3, len(sorted_indices) - 1)
            threshold = probs[sorted_indices[threshold_idx]]
            
            # 應用閾值，保留高概率動作
            mask = probs >= threshold
            probs = probs * mask
            
            # 重新歸一化
            total = np.sum(probs)
            if total > 0:
                probs = probs / total
            else:
                # 如果所有概率都被修剪掉了，返回最高的一個
                probs = np.zeros_like(probs)
                if len(sorted_indices) > 0:
                    probs[sorted_indices[0]] = 1.0
        
        # 最後確保總和為1（處理浮點數誤差）
        total = np.sum(probs)
        if abs(total - 1.0) > 1e-10 and total > 0:
            probs = probs / total
            
        return probs.tolist()
        
    def getExpertValue(self, canonical_board):
        """
        返回節點的專家價值（平均Q值）。
        
        參數:
            canonical_board: 標準形式的遊戲狀態
            
        返回:
            value: 節點價值
        """
        s = self._stringRepresentation(canonical_board)
        
        if s not in self.nodes:
            return 0
            
        node = self.nodes[s]
        
        if node.is_terminal:
            # 終端節點，返回遊戲結果
            return self.game.getGameEnded(canonical_board, 1)
            
        # 返回節點的平均Q值
        return node.value

    def _stringRepresentation(self, canonical_board):
        """
        獲取棋盤狀態的字符串表示。
        
        參數:
            canonical_board: 標準形式的遊戲狀態（numpy 數組格式）
            
        返回:
            string: 狀態的字符串表示
        """
        # 將數組轉換為字符串
        board_str = ""
        
        # 遍歷11x11棋盤
        for i in range(11):
            for j in range(11):
                # 檢查三個玩家通道，決定該位置的值
                if canonical_board[0, i, j] == 1:
                    board_str += "1"  # 當前玩家
                elif canonical_board[1, i, j] == 1:
                    board_str += "2"  # 下一個玩家
                elif canonical_board[2, i, j] == 1:
                    board_str += "3"  # 上一個玩家
                else:
                    board_str += "0"  # 空格或邊界
                    
        # 添加玩家順序和棄權計數作為區分
        player_order = canonical_board[3, 0, 0]
        pass_count = canonical_board[4, 0, 0]
        board_str += f"_{player_order}_{pass_count}"
                    
        return board_str 