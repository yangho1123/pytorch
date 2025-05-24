import numpy as np

"""
遊戲類別，提供AlphaZero所需的遊戲規則和狀態操作。
這個類別是基於game_nogroup.py中的State類實現的接口，
用於與fast-alphazero框架兼容。
"""

class Game():
    """
    三國翻轉棋遊戲類別。
    """
    
    def __init__(self):
        # 遊戲參數
        self.directions = [-11, -10, -1, 1, 10, 11]  # 六個方向
        # 棋盤邊界 - 六邊形棋盤
        self.board_size = (11, 11)
        self.action_size = 122  # 121個位置 + 1個棄權

    def getInitBoard(self):
        """
        返回遊戲初始狀態。
        返回形狀為 (5, 11, 11) 的 numpy 數組
        """
        # 初始化棋盤，5個通道
        board = np.zeros((5, 11, 11), dtype=np.float32)
        
        # 通道0-2存儲三方棋子
        # 原始一維棋盤轉為二維
        for i in range(11):
            for j in range(11):
                # 設置棋盤邊界
                if (i + j <= 4) or (i + j >= 16):
                    board[0:3, i, j] = -1  # 所有玩家通道的邊界都設為-1
        
        # 設置初始棋子
        # 紅子（通道0）
        board[0, 4, 5] = 1  # 對應 49
        board[0, 6, 5] = 1  # 對應 71
        
        # 綠子（通道1）
        board[1, 4, 6] = 1  # 對應 50
        board[1, 5, 5] = 1  # 對應 60
        board[1, 6, 4] = 1  # 對應 70
        
        # 藍子（通道2）
        board[2, 5, 4] = 1  # 對應 59
        board[2, 5, 6] = 1  # 對應 61
        
        # 通道3：玩家順序標記，初始為0（紅方）
        board[3, :, :] = 0.0
        
        # 通道4：棄權計數，初始為0
        board[4, :, :] = 0.0
        
        return board
    
    def getBoardSize(self):
        """
        返回棋盤尺寸。
        """
        return self.board_size
    
    def getActionSize(self):
        """
        返回動作空間大小。
        121個位置 + 1個棄權動作
        """
        return self.action_size
    
    def getNextState(self, board, player, action):
        """
        給定當前狀態、玩家和動作，返回下一個狀態。
        
        參數:
            board: 當前遊戲狀態 (5, 11, 11) numpy 數組
            player: 當前玩家 (1 代表當前玩家)
            action: 動作 (0-120為棋盤位置，121為棄權)
            
        返回:
            next_board: 下一個遊戲狀態
            next_player: 下一個玩家 (總是1，因為AlphaZero框架使用標準化視角)
        """
        next_board = board.copy()
        player_order = int(board[3, 0, 0] * 2 + 0.5)  # 0:紅, 1:綠, 2:藍
        
        if action == 121:
            next_board[4, :, :] += 1
        else:
            next_board[4, :, :] = 0
            y, x = action // 11, action % 11
            
            # 檢查位置是否合法
            if next_board[0, y, x] != 0 or next_board[1, y, x] != 0 or next_board[2, y, x] != 0:
                return next_board, 1
                
            # 在當前位置落子（通道0始終是當前玩家）
            next_board[0, y, x] = 1
            
            # 翻轉棋子
            for dx, dy in [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]:
                self._flip_direction(next_board, x, y, dx, dy)
        
        # 更新玩家順序
        new_player_order = (player_order + 1) % 3
        player_value = new_player_order / 2  # 0.0=紅, 0.5=綠, 1.0=藍
        next_board[3, :, :] = player_value
        
        # 交換通道順序，使通道0始終代表當前玩家
        temp = next_board[0].copy()
        next_board[0] = next_board[1]
        next_board[1] = next_board[2]
        next_board[2] = temp
        
        return next_board, 1
    
    def getValidMoves(self, board, player):
        """
        返回合法動作的掩碼向量。
        參數:
            board: 當前遊戲狀態
            player: 當前玩家 (1 代表當前玩家)
            
        返回:
            valid_moves: 一個長度為122的二進制向量，1表示合法動作
        """
        valid_moves = [0] * self.action_size
        
        # 檢查遊戲是否結束
        if self.getGameEnded(board, player) != 0:
            return valid_moves
            
        actions = self._get_legal_actions(board)
        # 如果沒有合法落子點，允許棄權
        if not actions:
            valid_moves[121] = 1
        else:
            for action in actions:
                valid_moves[action] = 1
                
        return valid_moves
    
    def getGameEnded(self, board, player):
        """
        檢查遊戲是否結束。
        參數:
            board: 當前遊戲狀態
            player: 當前玩家
            
        返回:
            0: 遊戲未結束
            1: 玩家獲勝
            -1: 玩家失敗
            小數: 平局 (平局是罕見的，所以用小數值)
        """
        # 檢查遊戲結束條件
        if self._is_game_over(board):
            # 計算比分
            result = self._calculate_result(board)
            player_order = int(board[3, 0, 0] * 2 + 0.5)  # 獲取當前玩家順序
            my_result = result[player_order]
            
            # 從當前玩家角度返回結果
            if my_result > 0:
                return 1
            elif my_result < 0:
                return -1
            else:
                return 0.0001  # 小數表示平局
        return 0
    
    def getCanonicalForm(self, board, player):
        """
        返回狀態的標準形式（從當前玩家視角）。
        在三國翻轉棋中，狀態已經是從當前玩家視角設計的，所以直接返回。
        """
        return board
    
    def getSymmetries(self, board, pi):
        """
        棋盤的對稱變換，用於數據增強。
        對於六邊形棋盤，這裡只考慮六次旋轉對稱。
        
        參數:
            board: 遊戲狀態
            pi: 策略向量
            
        返回:
            sym: 對稱變換後的(狀態, 策略)列表
        """
        # 對於六邊形棋盤，對稱變換較為複雜
        # 這裡為了示範，暫時只返回原始狀態
        # 實際實現中，可以根據六邊形棋盤的特性實現更複雜的對稱變換
        return [(board, pi)]
    
    def stringRepresentation(self, board):
        """
        返回狀態的字符串表示，用於MCTS字典的鍵。
        """
        # 將numpy數組轉換為唯一的字符串表示
        # 僅使用棋盤狀態(通道0-2)和玩家順序(通道3)
        board_flat = board[0:3].flatten()
        player_order = board[3, 0, 0]
        pass_turn = board[4, 0, 0]
        
        # 連接成一個字符串
        board_str = ''.join([str(int(x)) for x in board_flat])
        board_str += f"_{player_order}_{pass_turn}"
        
        return board_str
    
    # 輔助方法，轉換私有方法名稱以適應新的數據結構
    
    def _flip_direction(self, board, x, y, dx, dy):
        """
        沿著指定方向翻轉棋子。
        """
        # 當前玩家的棋子在通道0
        # 其他玩家的棋子在通道1和通道2
        
        x, y = x + dx, y + dy
        
        # 檢查是否越界
        if (x < 0 or y < 0 or x > 10 or y > 10 or 
            board[0, y, x] == -1 or 
            (board[1, y, x] != 1 and board[2, y, x] != 1)):
            return False
            
        enemy = 1 if board[2, y, x] == 1 else 2
        
        # 搜索可夾住的敵方棋子
        temp_x, temp_y = x, y
        can_flip = False
        
        while True:
            temp_x += dx
            temp_y += dy
            
            # 檢查是否越界
            if (temp_x < 0 or temp_y < 0 or temp_x > 10 or temp_y > 10 or 
                board[0, temp_y, temp_x] == -1):
                break
                
            # 如果遇到空格，不能翻轉
            if (board[0, temp_y, temp_x] == 0 and 
                board[1, temp_y, temp_x] == 0 and 
                board[2, temp_y, temp_x] == 0):
                break
                
            # 如果找到己方棋子或第三方棋子，可以翻轉
            if (board[0, temp_y, temp_x] == 1 or 
                (enemy == 1 and board[1, temp_y, temp_x] == 1) or 
                (enemy == 2 and board[2, temp_y, temp_x] == 1)):
                can_flip = True
                break
        
        # 如果可以翻轉，進行翻轉
        if can_flip:
            while True:
                # 回溯到起始位置
                temp_x -= dx
                temp_y -= dy
                
                # 到達起始位置則停止
                if temp_x == x and temp_y == y:
                    break
                    
                # 翻轉
                board[0, temp_y, temp_x] = 1
                if enemy == 1:
                    board[2, temp_y, temp_x] = 0
                else:
                    board[1, temp_y, temp_x] = 0
                    
        return can_flip
    
    def _get_legal_actions(self, board):
        """
        獲取所有合法動作。
        """
        actions = []
        
        for y in range(11):
            for x in range(11):
                # 跳過棋盤外的位置
                if board[0, y, x] == -1:
                    continue
                # 跳過已有棋子的位置
                if board[0, y, x] == 1 or board[1, y, x] == 1 or board[2, y, x] == 1:
                    continue
                
                # 檢查是否是合法動作
                temp_board = board.copy()
                if self._check_legal_action(temp_board, x, y):
                    actions.append(y * 11 + x)
                    
        return actions
    
    def _check_legal_action(self, board, x, y):
        """
        檢查在(x,y)處下子是否合法。
        """
        # 已有棋子的位置不合法
        if board[0, y, x] == 1 or board[1, y, x] == 1 or board[2, y, x] == 1:
            return False
            
        # 在六個方向檢查是否可以翻轉
        for dx, dy in [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]:
            temp_board = board.copy()
            if self._check_direction(temp_board, x, y, dx, dy):
                return True
        return False
    
    def _check_direction(self, board, x, y, dx, dy):
        """
        檢查在指定方向上是否可以翻轉棋子。
        """
        x, y = x + dx, y + dy
        
        # 檢查是否越界
        if (x < 0 or y < 0 or x > 10 or y > 10 or 
            board[0, y, x] == -1):
            return False
            
        # 檢查該方向上是否有敵方棋子
        if board[1, y, x] != 1 and board[2, y, x] != 1:
            return False
            
        enemy = 1 if board[2, y, x] == 1 else 2
        
        while True:
            x += dx
            y += dy
            
            # 檢查是否越界
            if (x < 0 or y < 0 or x > 10 or y > 10 or 
                board[0, y, x] == -1):
                return False
                
            # 如果遇到空格，不能翻轉
            if (board[0, y, x] == 0 and 
                board[1, y, x] == 0 and 
                board[2, y, x] == 0):
                return False
                
            # 如果找到己方棋子或第三方棋子，可以翻轉
            if (board[0, y, x] == 1 or 
                (enemy == 1 and board[1, y, x] == 1) or 
                (enemy == 2 and board[2, y, x] == 1)):
                return True
    
    def _is_game_over(self, board):
        """
        檢查遊戲是否結束。
        """
        # 檢查是否所有可用位置都被占用
        total_pieces = 0
        empty_spaces = 0
        for y in range(11):
            for x in range(11):
                if board[0, y, x] == -1:
                    continue  # 跳過棋盤外的位置
                if (board[0, y, x] == 1 or 
                    board[1, y, x] == 1 or 
                    board[2, y, x] == 1):
                    total_pieces += 1
                else:
                    empty_spaces += 1
        
        # 打印调试信息
        #print(f"当前棋子总数: {total_pieces}, 空位数量: {empty_spaces}, 弃权计数: {board[4, 0, 0]}")
        
        # 棋盤已滿
        if total_pieces == 91:  # 11x11的六邊形棋盤有91個有效位置
            #print("游戏结束：棋盘已满")
            return True
        
        # 三方連續棄權
        if board[4, 0, 0] >= 3:
            #print("游戏结束：连续弃权")
            return True
        
        return False
    
    def _calculate_result(self, board):
        """
        計算遊戲結果，考慮貼目。
        
        返回:
            result: 三個玩家的得分，[紅, 綠, 藍]
        """
        # 貼目值
        red_komi = 0  # 可以根據需要調整
        green_komi = 1
        blue_komi = 2
        
        # 計算三方的棋子數量
        player_order = int(board[3, 0, 0] * 2 + 0.5)  # player_order 0:紅, 1:綠, 2:藍
        
        # 由於玩家順序在board中已經輪轉，需要重新映射
        if player_order == 0:  # 紅方回合
            red_count = np.sum(board[0] == 1)
            green_count = np.sum(board[1] == 1)
            blue_count = np.sum(board[2] == 1)
        elif player_order == 1:  # 綠方回合
            green_count = np.sum(board[0] == 1)
            blue_count = np.sum(board[1] == 1)
            red_count = np.sum(board[2] == 1)
        else:  # 藍方回合
            blue_count = np.sum(board[0] == 1)
            red_count = np.sum(board[1] == 1)
            green_count = np.sum(board[2] == 1)
        
        # 考慮當前玩家順序
        if player_order == 0:  # 紅方回合
            red_count -= red_komi
            green_count -= green_komi
            blue_count -= blue_komi
            
            # 根據棋子數量計算分數 - 紅方視角
            if red_count >= green_count and red_count >= blue_count:
                if green_count >= blue_count:
                    return [1, 0, -1]  # 紅勝、綠平、藍負
                else:
                    return [1, -1, 0]  # 紅勝、綠負、藍平
            elif green_count >= blue_count:
                if red_count >= blue_count:
                    return [0, 1, -1]  # 紅平、綠勝、藍負
                else:
                    return [-1, 1, 0]  # 紅負、綠勝、藍平
            else:
                if red_count >= green_count:
                    return [0, -1, 1]  # 紅平、綠負、藍勝
                else:
                    return [-1, 0, 1]  # 紅負、綠平、藍勝
                    
        elif player_order == 1:  # 綠方回合
            green_count -= green_komi
            blue_count -= blue_komi
            red_count -= red_komi
            
            # 根據棋子數量計算分數 - 綠方視角
            if green_count > red_count and green_count >= blue_count:
                if blue_count > red_count:
                    return [-1, 1, 0]  # 紅負、綠勝、藍平
                else:
                    return [0, 1, -1]  # 紅平、綠勝、藍負
            elif blue_count > red_count:
                if green_count > red_count:
                    return [-1, 0, 1]  # 紅負、綠平、藍勝
                else:
                    return [0, -1, 1]  # 紅平、綠負、藍勝
            else:
                if green_count >= blue_count:
                    return [1, 0, -1]  # 紅勝、綠平、藍負
                else:
                    return [1, -1, 0]  # 紅勝、綠負、藍平
                    
        else:  # 藍方回合
            blue_count -= blue_komi
            red_count -= red_komi
            green_count -= green_komi
            
            # 根據棋子數量計算分數 - 藍方視角
            if blue_count > green_count and blue_count > red_count:
                if red_count >= green_count:
                    return [0, -1, 1]  # 紅平、綠負、藍勝
                else:
                    return [-1, 0, 1]  # 紅負、綠平、藍勝
            elif red_count >= green_count:
                if blue_count > green_count:
                    return [1, -1, 0]  # 紅勝、綠負、藍平
                else:
                    return [1, 0, -1]  # 紅勝、綠平、藍負
            else:
                if blue_count > red_count:
                    return [-1, 1, 0]  # 紅負、綠勝、藍平
                else:
                    return [0, 1, -1]  # 紅平、綠勝、藍負
        
        # 平局情況，如果前面的條件都不滿足（這種情況很少見）
        return [0, 0, 0]

    def get_player_order(self, board):
        """
        返回當前輪到的玩家序號。
        0: 紅方, 1: 綠方, 2: 藍方
        """
        return int(board[3, 0, 0] * 2 + 0.5)  # 將0.0/0.5/1.0轉回0/1/2
    
    def playMoves(self):
        """
        根據MCTS搜索結果選擇動作，並更新遊戲狀態。
        """
        for i in range(self.batch_size):
            # 根據溫度參數確定是否使用確定性策略
            temp = int(self.turn[i] < self.args.tempThreshold)
            
            # 根據MCTS訪問次數選擇動作
            policy = self.mcts[i].getExpertProb(
                self.canonical[i], temp, not self.fast)
            
            # 確保策略向量有效且可用於抽樣
            policy = np.array(policy, dtype=np.float64)  # 轉換為數組並使用高精度
            
            # 處理NaN和Inf值
            policy = np.nan_to_num(policy, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 處理總和為0的情況
            policy_sum = np.sum(policy)
            if policy_sum <= 0:
                # 如果策略總和為0，創建均勻分布
                valid_moves = self.game.getValidMoves(self.canonical[i], 1)
                valid_count = sum(valid_moves)
                if valid_count > 0:
                    # 使用有效移動創建均勻分布
                    policy = np.array(valid_moves, dtype=np.float64) / valid_count
                else:
                    # 如果沒有有效移動（不應該發生），使用前幾個位置
                    policy = np.zeros_like(policy)
                    policy[0] = 1.0
            else:
                # 正常歸一化
                policy = policy / policy_sum
            
            # 最後檢查確保總和為1，處理可能的浮點誤差
            policy_sum = np.sum(policy)
            if abs(policy_sum - 1.0) > 1e-10:
                # 強制調整最後一個非零元素使總和為1
                nonzero_indices = np.nonzero(policy)[0]
                if len(nonzero_indices) > 0:
                    last_idx = nonzero_indices[-1]
                    policy[last_idx] += 1.0 - policy_sum
            
            # 重新檢查確保沒有NaN值
            if np.isnan(policy).any() or np.isinf(policy).any():
                # 如果仍有NaN或Inf值，使用均勻分布
                print("Warning: NaN or Inf in policy after normalization!")
                policy = np.ones_like(policy) / len(policy)
            
            try:
                action = np.random.choice(len(policy), p=policy)
            except ValueError as e:
                # 如果仍然出錯，打印詳細信息並使用均勻分布
                print(f"Error sampling from policy: {e}")
                print(f"Policy: {policy}")
                print(f"Policy sum: {np.sum(policy)}")
                policy = np.ones(len(policy)) / len(policy)
                action = np.random.choice(len(policy), p=policy)
            
            # 如果不是快速模擬，記錄遊戲歷史（用於訓練）
            # ... 其餘代碼不變 ... 

    def getFullGameResult(self, board):
        """
        返回完整的遊戲結果（三方勝負）。
        僅在遊戲結束時調用。
        
        參數:
            board: 當前遊戲狀態
            
        返回:
            result: 三個玩家的結果數組 [紅方, 綠方, 藍方]，每個元素為1(贏),-1(輸)或0(平)
                   如果遊戲未結束，返回[0,0,0]
        """
        if not self._is_game_over(board):
            return [0, 0, 0]  # 遊戲未結束
        
        # 計算三方的勝負結果
        result = self._calculate_result(board)
        
        # 由於board中玩家順序已經輪轉，需要將結果映射回原始玩家順序
        player_order = int(board[3, 0, 0] * 2 + 0.5)  # 0:紅, 1:綠, 2:藍
        
        # 根據當前玩家順序調整結果順序
        if player_order == 0:  # 紅方回合
            return result  # 結果已經是[紅, 綠, 藍]順序
        elif player_order == 1:  # 綠方回合
            # 棋盤中的player_order=1(綠方回合)，意味著第0通道是綠方，第1通道是藍方，第2通道是紅方
            # 因此結果順序是[綠, 藍, 紅]，需要調整回[紅, 綠, 藍]
            return [result[2], result[0], result[1]]
        else:  # 藍方回合(player_order=2)
            # 棋盤中的player_order=2(藍方回合)，意味著第0通道是藍方，第1通道是紅方，第2通道是綠方
            # 因此結果順序是[藍, 紅, 綠]，需要調整回[紅, 綠, 藍]
            return [result[1], result[2], result[0]] 