# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
import cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from libc.stdlib cimport rand, RAND_MAX
import torch    
import time
import uuid
import random

# 設置 NumPy 數據類型
np.import_array()
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

# 從遊戲模塊導入必要的類
from game_nogroup import State
from game_nogroup import FlipTable

# 神經網絡相關的常數
from dual_network_3his import DN_INPUT_SHAPE

# 常數定義
PV_EVALUATE_COUNT = 400
BATCH_SIZE = 8  # 批處理大小，可根據 GPU 內存調整

# 獲取 FLIPTABLE 並進行標準化
FLIPTABLE = FlipTable
# 正規化 FLIPTABLE 到 0-1 範圍
FLIPTABLE_MIN = min(x for x in FLIPTABLE if x != -1)  # 忽略 -1（非法位置）
FLIPTABLE_MAX = max(FLIPTABLE)
# 修改正規化邏輯，將 -1 轉換為 0
NORMALIZED_FLIPTABLE = [(x - FLIPTABLE_MIN) / (FLIPTABLE_MAX - FLIPTABLE_MIN) if x != -1 else 0 for x in FLIPTABLE]

def get_device():
    """獲取設備 (CPU or CUDA)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

def prepare_input_tensor(state, path, int history_length=11):
    """準備單個輸入張量，方便批處理"""
    board_states = []
    # 輸入統一為紅、綠、藍
    for i in range(history_length + 1):  # +1 包括當前狀態
        if len(path) > i:
            current_node = path[-i - 1]     # -i 指向倒數第 i 個元素
            current_player = current_node.state.get_player_order()
            
            if current_player == 0:
                red_pieces = current_node.state.mine_pieces.copy()
                green_pieces = current_node.state.next_pieces.copy()
                blue_pieces = current_node.state.prev_pieces.copy()
            elif current_player == 1:
                red_pieces = current_node.state.prev_pieces.copy()
                green_pieces = current_node.state.mine_pieces.copy()
                blue_pieces = current_node.state.next_pieces.copy()
            else:
                red_pieces = current_node.state.next_pieces.copy()
                green_pieces = current_node.state.prev_pieces.copy()
                blue_pieces = current_node.state.mine_pieces.copy()
                
            # 將 -1 邊界值轉換為 0，使所有特徵都在 0~1 範圍內
            red_pieces = [0 if x == -1 else x for x in red_pieces]
            green_pieces = [0 if x == -1 else x for x in green_pieces]
            blue_pieces = [0 if x == -1 else x for x in blue_pieces]
                
            # 模型的輸入狀態始終按照紅、綠、藍的順序
            state_tensor = torch.tensor([
                red_pieces, 
                green_pieces,
                blue_pieces
            ])
            # 創建玩家通道，正規化到 0-1 範圍
            normalized_player = current_player / 2.0
            player_channel = torch.full((121,), normalized_player, dtype=torch.float32)
            # 使用正規化後的 FLIPTABLE
            FLIPTABLE_tensor = torch.tensor(NORMALIZED_FLIPTABLE, dtype=torch.float32).unsqueeze(0)
            full_state_tensor = torch.cat((state_tensor, player_channel.unsqueeze(0), FLIPTABLE_tensor), dim=0)
            board_states.append(full_state_tensor)
        else:
            # 如果沒有歷史狀態，使用零張量
            zero_state = torch.zeros((5, 121))
            board_states.append(zero_state)
    
    # 堆疊所有狀態
    x = torch.stack(board_states).float()
    x = x.view(1, 5 * (history_length + 1), 11, 11)
    return x

def predict(model, state, path): 
    """使用神經網絡預測下一步"""    
    x = prepare_input_tensor(state, path)
    x = x.to(get_device())
    model = model.to(get_device())
    
    with torch.no_grad():
        y = model(x)

    policy_softmax = torch.nn.functional.softmax(y[0][0], dim=0)
    policies = policy_softmax[list(state.all_legal_actions())]
    policies /= sum(policies) if sum(policies) else 1  # 總和為 0 就除以 1
    value = y[1][0]
    
    policies = policies.cpu().numpy()
    value = value.cpu().numpy()
    return policies, value.item()

def predict_batch(model, states_paths, legal_actions_list):
    """批量預測多個狀態"""
    if not states_paths:
        return {}
    
    device = get_device()
    model = model.to(device)
    batch_tensors = []
    
    # 為每個狀態準備輸入張量
    for state, path in states_paths:
        x = prepare_input_tensor(state, path)
        batch_tensors.append(x)
    
    # 合併為一個批次
    batch_input = torch.cat(batch_tensors, dim=0).to(device)
    
    # 批量預測
    results = {}
    with torch.no_grad():
        y = model(batch_input)
    
    # 處理每個預測結果
    for i, (state, path) in enumerate(states_paths):
        policy_logits = y[0][i]
        value = y[1][i].cpu().numpy().item()
        
        # 對策略進行 softmax
        policy_softmax = torch.nn.functional.softmax(policy_logits, dim=0)
        
        # 獲取合法動作的策略
        legal_actions = legal_actions_list[i]
        policies = policy_softmax[legal_actions].cpu().numpy()
        
        # 歸一化策略
        sum_p = np.sum(policies)
        if sum_p > 0:
            policies = policies / sum_p
        
        # 確保能夠正確識別節點
        if len(path) > 0 and hasattr(path[-1], 'id'):
            node_id = path[-1].id
        else:
            # 使用一個不容易衝突的標識
            node_id = f"node_{i}_{hash(str(state.mine_pieces) + str(state.next_pieces) + str(state.prev_pieces))}"
            
        results[node_id] = (policies, value)
    
    return results

@cython.cdivision(True)
def nodes_to_scores(nodes):
    """返回某節點的所有子節點的訪問次數"""
    cdef list scores = []
    for c in nodes:
        scores.append(c.n)
    return scores

@cython.cdivision(True)
def boltzman(xs, double temperature):
    """波茲曼分佈計算"""
    cdef list xs_temp = []
    cdef double sum_value = 0.0
    
    for x in xs:
        value = x ** (1.0 / temperature)
        xs_temp.append(value)
        sum_value += value
    
    return [x / sum_value for x in xs_temp]

@cython.cdivision(True)
def pv_mcts_scores(model, state, double temperature, bint add_noise=True, double dirichlet_alpha=0.03):
    """使用蒙特卡羅樹搜索計算概率分佈"""
    cdef int current_batch_size = BATCH_SIZE
    cdef int simulations_done = 0
    cdef int batch_counter = 0  # 統計批處理次數
    cdef int stalled_counter = 0  # 計數連續沒有評估新節點的次數
    cdef int num_batch_collect = 0
    cdef int successful_evals = 0
    cdef int original_queue_size = 0
    
    class Node:
        """表示 MCTS 樹中的一個節點"""
        def __init__(self, state, p, prev_node=None):
            self.state = state  # 盤面
            self.id = str(uuid.uuid4())  # 添加唯一 ID 以便在批處理中識別節點
            if prev_node is None and add_noise:     # dirichlet noise 用在 root node
                epsilon = 0.25
                p = np.array(p)
                noise = np.random.dirichlet([dirichlet_alpha] * len(p))
                p = (1 - epsilon) * p + epsilon * noise
            self.p = p  # 策略
            self.n = 0  # 場數
            self.w = 0  # score
            self.child_nodes = None  # 下一回合盤面可能性
            self.parent = prev_node
            self.need_eval = False  # 標記是否需要評估
            self.eval_path = None   # 保存評估路徑
        
        def count_nodes(self):
            """計算節點及其子節點的總數"""
            if self.child_nodes is None:
                return 1  # 葉節點（Leaf Node）
            count = 1
            for child in self.child_nodes:
                count += child.count_nodes()  # 遞迴計算子節點數量
            return count
            
        def count_tree_depth(self):
            """統計 MCTS 樹的最大深度"""
            if self.child_nodes is None:
                return 1  # 葉節點深度為 1

            return 1 + max(child.count_tree_depth() for child in self.child_nodes)
        
        def evaluate_root(self, eval_queue=None):
            """評估根節點，收集需要評估的節點"""
            if self.state.is_done():  # 終局
                finish = np.array(self.state.finish_reward())
                value = finish[self.state.get_player_order()]
                # 重要：只有尚未評估過的終局才增加計數
                if self.n == 0:
                    self.w += value
                    self.n += 1
                return value
                
            if not self.child_nodes:  # 當前 node 沒有子節點 -> expand
                if self.parent is None:  # root expand
                    path = []
                    initial_policy, _ = predict(model, self.state, path)  # 根節點的初始策略還是單獨計算
                    self.child_nodes = []
                    for action, policy in zip(self.state.all_legal_actions(), initial_policy):
                        next_state = self.state.next(action)
                        self.child_nodes.append(Node(next_state, policy, self))
                    return 0
                else:
                    # 標記節點需要評估
                    self.need_eval = True
                    self.eval_path = self.trace_back()
                    if eval_queue is not None:
                        eval_queue.append(self)
                    return 0  # 臨時返回 0，後續會更新
                    
            else:  # 子節點存在
                # 檢查是否所有子節點都已評估過
                all_children_visited = all(child.n > 0 for child in self.child_nodes)
                if all_children_visited and len(self.child_nodes) > 0:
                    # 如果所有子節點都已訪問，但我們仍選擇了一個節點，可能是陷入了循環
                    pass
                
                next_node = self.next_child_node()
                value = next_node.evaluate_root(eval_queue)  # 繼續向下搜索
                self.w += value
                self.n += 1
                return value
                
        def evaluate(self):
            """使用預計算的評估結果展開節點"""
            if self.need_eval and hasattr(self, 'eval_result'):
                policies, value = self.eval_result
                
                # 更新節點並創建子節點
                self.w += value
                self.n += 1
                self.child_nodes = []
                for action, policy in zip(self.state.all_legal_actions(), policies):
                    next_state = self.state.next(action)
                    self.child_nodes.append(Node(next_state, policy, self))
                
                # 清除臨時數據
                self.need_eval = False
                self.eval_path = None
                delattr(self, 'eval_result')
                
                return value
                
            # 終局情況
            if self.state.is_done():
                finish = np.array(self.state.finish_reward())
                value = finish[self.state.get_player_order()]
                if self.n == 0:
                    self.w += value
                    self.n += 1
                return value
            
            # 子節點存在時
            else:
                value = self.next_child_node().evaluate()
                self.w += value
                self.n += 1
                return value
            
        @cython.cdivision(True)
        def next_child_node(self):
            """選擇下一個要探索的子節點"""
            cdef double C_PUCT = sqrt(3)
            cdef int t = sum(nodes_to_scores(self.child_nodes))  # t = 子節點訪問次數的總和
            
            unvisited_nodes = [child for child in self.child_nodes if child.n == 0]
            if unvisited_nodes:
                chosen_node = random.choice(unvisited_nodes)
                return chosen_node
                
            pucb_values = []
            for child_node in self.child_nodes:
                q_value = (child_node.w / (child_node.n)) if child_node.n > 0 else 0.0
                p_value = child_node.p
                exploration_term = C_PUCT * p_value * sqrt(t) / (1 + child_node.n)
                pucb_value = q_value + exploration_term
                pucb_values.append(pucb_value)
                
            best_node = self.child_nodes[np.argmax(pucb_values)]
            return best_node
        
        def trace_back(self):
            """回溯路徑從根節點到當前節點"""
            path = []
            current = self
            while current is not None:
                path.append(current)
                current = current.parent
            return path[::-1]  # 反轉得到從根到當前節點的路徑
    
    # 初始化根節點
    path = []
    initial_policy, _ = predict(model, state, path)
    root_node = Node(state, initial_policy)
    
    # 記錄運行時間
    start_time = time.time()
    
    # 主循環，使用批處理模式
    while simulations_done < PV_EVALUATE_COUNT:
        # 收集需要評估的節點
        eval_queue = []
        
        # 進行一定數量的模擬以填充評估隊列
        num_batch_collect = min(current_batch_size, PV_EVALUATE_COUNT - simulations_done)
        for _ in range(num_batch_collect):
            root_node.evaluate_root(eval_queue)
            
        # 如果有需要評估的節點，進行批量評估
        if eval_queue:
            batch_counter += 1
            
            # 避免重複評估類似狀態
            # 添加一個簡單的去重機制
            node_states_hash = set()
            original_queue_size = len(eval_queue)
            
            for queue_node in eval_queue[:]:  # 使用切片創建副本以避免在迭代時修改
                # 創建狀態的簡單哈希(使用盤面信息)
                state_hash = str(queue_node.state.mine_pieces) + str(queue_node.state.next_pieces) + str(queue_node.state.prev_pieces)
                if state_hash not in node_states_hash:
                    node_states_hash.add(state_hash)
                else:
                    eval_queue.remove(queue_node)  # 移除類似狀態
            
            # 批量預測
            states_paths = [(queue_node.state, queue_node.eval_path) for queue_node in eval_queue]
            legal_actions_list = [queue_node.state.all_legal_actions() for queue_node in eval_queue]
            
            # 批量預測
            eval_results = predict_batch(model, states_paths, legal_actions_list)
            
            # 將預測結果應用到節點
            successful_evals = 0
            for queue_node in eval_queue:
                if queue_node.id in eval_results:
                    queue_node.eval_result = eval_results[queue_node.id]
                    # 完成節點的評估
                    queue_node.evaluate()
                    successful_evals += 1
            
            simulations_done += successful_evals
            
            # 檢測是否卡住
            if successful_evals == 0:
                stalled_counter += 1
                
                if stalled_counter >= 3:
                    # 重置評估隊列
                    eval_queue = []
                    # 調整批處理大小為1，簡化搜索
                    current_batch_size = 1
                    stalled_counter = 0
            else:
                stalled_counter = 0
        else:
            # 如果評估隊列為空，可能是樹已經完全展開，或搜索卡住
            stalled_counter += 1
            
            if stalled_counter >= 3:
                break
    
    # 獲取根節點子節點的訪問次數並計算得分
    scores = nodes_to_scores(root_node.child_nodes)
    
    # 應用溫度參數
    if temperature == 0:  # 取最大值
        action = np.argmax(scores)
        scores = np.zeros(len(scores))
        scores[action] = 1
    else:
        scores = boltzman(scores, temperature)
        
    return scores

def pv_mcts_action(model, temperature=0):
    """返回一個函數，該函數使用 MCTS 計算最佳行動"""
    def pv_mcts_action(state):
        scores = pv_mcts_scores(model, state, temperature, False)
        return np.random.choice(state.all_legal_actions(), p=scores)
    return pv_mcts_action 