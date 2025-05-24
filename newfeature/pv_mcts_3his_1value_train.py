from game import State
from game_nogroup import FlipTable
from dual_network_3his import DN_INPUT_SHAPE
from math import sqrt
import torch    
import numpy as np
import matplotlib.pyplot as plt
import time
import random

PV_EVALUATE_COUNT = 300
FLIPTABLE = FlipTable
# 正規化FLIPTABLE到0-1範圍
FLIPTABLE_MIN = min(x for x in FLIPTABLE if x != -1)  # 忽略-1（非法位置）
FLIPTABLE_MAX = max(FLIPTABLE)
# 修改正規化邏輯，將-1轉換為0
NORMALIZED_FLIPTABLE = [(x - FLIPTABLE_MIN) / (FLIPTABLE_MAX - FLIPTABLE_MIN) if x != -1 else 0 for x in FLIPTABLE]

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

def predict(model, state, path): # 利用對偶網路做下一步的預測    
    history_length = 11
    board_states = []
    # 輸入統一為紅、綠、藍
    # Retrieve the last few states from the path
    for i in range(history_length + 1):  # +1 to include the current state
        if len(path) > i:
            current_node = path[-i - 1]     # -i指向倒數第i個元素，path的第一個元素是當前state，最後一個是root
            current_player = current_node.state.get_player_order()  # 用目前回合數來判斷當前mine是什麼顏色
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
                
            # 將-1邊界值轉換為0，使所有特徵都在0~1範圍內
            red_pieces = [0 if x == -1 else x for x in red_pieces]
            green_pieces = [0 if x == -1 else x for x in green_pieces]
            blue_pieces = [0 if x == -1 else x for x in blue_pieces]
                
            # 模型的輸入狀態始终按照紅、綠、藍的順序
            state_tensor = torch.tensor([
                red_pieces, 
                green_pieces,
                blue_pieces
            ])
            # 创建玩家通道
            normalized_player = current_player / 2.0
            player_channel = torch.full((121,), normalized_player, dtype=torch.float32)
            # 使用正規化的FLIPTABLE
            FLIPTABLE_tensor = torch.tensor(NORMALIZED_FLIPTABLE, dtype=torch.float32).unsqueeze(0)  # 将FLIPTABLE转换为Tensor，并增加一个维度
            full_state_tensor = torch.cat((state_tensor, player_channel.unsqueeze(0), FLIPTABLE_tensor), dim=0)
            board_states.append(full_state_tensor)
        else:
            # 如果没有历史状态，使用零张量
            zero_state = torch.zeros((5, 121))
            board_states.append(zero_state)
               
    # Stack all states to create a tensor of shape [9, 5, 11, 11]    
    x = torch.stack(board_states).float()   # Ensure float type for model processing
    x = x.view(1, 5 * (history_length + 1), 11, 11)            # Reshape to [1, 45, 11, 11]
    #print("Input tensor shape:", x.shape)  # 打印x的形狀來驗證
    x = x.to(get_device())  # 確保數據在正確的設備上    
    model = model.to(get_device())
    
    with torch.no_grad(): # 預測不需要計算梯度        
        y = model(x)

    #print(y[0][0])
    policy_softmax = torch.nn.functional.softmax(y[0][0], dim=0)   # 將policy輸出做softmax
    policies = policy_softmax[list(state.all_legal_actions())]
    #policies = y[0][0][list(state.all_legal_actions())]
    policies /= sum(policies) if sum(policies) else 1 # 總合為0就除以1
    value = y[1][0]
    
    policies = policies.cpu().numpy()
    value = value.cpu().numpy()
    return policies, value.item()


def nodes_to_scores(nodes): # 回傳某節點的所有子節點的拜訪次數
    scores = []
    for c in nodes:
        scores.append(c.n)
    return scores

def boltzman(xs, temperature):
    xs = [x ** (1 / temperature) for x in xs]
    return [x / sum(xs) for x in xs]

def pv_mcts_scores(model, state, temperature, add_noise=True, dirichlet_alpha=0.03):
    class node:
        def __init__(self, state, p, prev_node=None):
            self.state = state # 盤面
            if prev_node is None and add_noise:     # dirichlet noise用在root node
                epsilon = 0.15
                p = np.array(p)
                noise = np.random.dirichlet([dirichlet_alpha] * len(p))
                p = (1 - epsilon) * p + epsilon * noise
            self.p = p # 策略
            self.n = 0 # 場數
            self.w = 0 # score
            self.child_nodes = None # 下一回合盤面可能性
            self.parent = prev_node
        
        def count_nodes(self):
            if self.child_nodes is None:
                return 1  # 葉節點（Leaf Node）
            count = 1
            for child in self.child_nodes:
                count += child.count_nodes()  # 遞迴計算子節點數量
            return count
        def count_tree_depth(self):
            """ 統計 MCTS 樹的最大深度 """
            if self.child_nodes is None:
                return 1  # 葉節點深度為 1

            return 1 + max(child.count_tree_depth() for child in self.child_nodes)
        
        def evaluate(self): # 給固定節點測試親眼看到拜訪每個節點的順序
            if self.state.is_done(): # 終局
                finish = np.array(self.state.finish())
                value = finish[self.state.get_player_order()]
                self.w += value     # 根據該節點深度(因為是終局，get_player_order大部分為0，但有少部分為1 or 2)
                self.n += 1
                #print("end state")
                return value
        
            if not self.child_nodes: # 當前node沒有子節點 -> expand
                if self.parent is None:
                    self.child_nodes = []
                    for action, policy in zip(self.state.all_legal_actions(), self.p):    # root expand
                        next_state = self.state.next(action)
                        self.child_nodes.append(node(next_state, policy, self))
                else:        
                    path = self.trace_back()
                    policies, value = predict(model, self.state, path)
                    #print("value: ", value)
                    self.w += value     # 模型預測當前盤面狀態的價值(-1~1)
                    self.n += 1
                    self.child_nodes = []
                    for action, policy in zip(self.state.all_legal_actions(), policies):    # expand
                        next_state = self.state.next(action)
                        self.child_nodes.append(node(next_state, policy, self))
                    return value
            
            else: # 子節點存在
                value = self.next_child_node().evaluate() # 取得pucb最大的子節點算出來的分數 (select & playout(evaluate))
                self.w += value
                self.n += 1
                return value
            
        def next_child_node(self): # select
            C_PUCT = sqrt(3)
            t = sum(nodes_to_scores(self.child_nodes))     #t=子節點拜訪次數的總和
            if t == 0:
                p_values = [child.p for child in self.child_nodes]
                best_node = self.child_nodes[np.argmax(p_values)]
                return best_node
            pucb_values = []
            # **找到所有 n==0 的節點**
            unvisited_nodes = [child for child in self.child_nodes if child.n == 0]
            # **若有未訪問過的節點，則從中隨機選擇一個**
            if unvisited_nodes:
                chosen_node = random.choice(unvisited_nodes)
                #print(f"  -> Randomly selected unvisited node: {id(chosen_node)}")
                return chosen_node
            
            for child_node in self.child_nodes:
                # if child_node.n == 0:
                #     p_value = child_node.p
                #     pucb_value = 100.0
                #     pucb_values.append(pucb_value)
                # else:
                    q_value = (child_node.w / child_node.n) if child_node.n > 0 else 0.0
                    p_value = child_node.p
                    exploration_term = C_PUCT * p_value * sqrt(t) / (1 + child_node.n)
                    pucb_value = q_value + exploration_term
                    # if child_node.n == 0:
                    #     print("pucb: ", pucb_value)
                    #     print("p: ", p_value)
                    pucb_values.append(pucb_value)
            
                # Debug: Print the calculation details
                #print("t: ", t)
                #print(f"exploration_term: {exploration_term:.4f}, p Value: {child_node.p:.4f}, t: {t:.4f}, sqrt(t): {sqrt(t):.4f}, n Term: {child_node.n:.4f}")
                #print(f"PUCB Value: {pucb_value:.4f}")
            best_node = self.child_nodes[np.argmax(pucb_values)]    # 回傳PUCT分數最大的node，如果有一樣大的值則回傳第一個
            return best_node
        
        def trace_back(self):        
            path = []
            current = self
            while current is not None:
                path.append(current)
                current = current.parent
            return path[::-1]  # Reverse to get path from root to this node
        
        
    path = []
    initial_policy, _ = predict(model, state, path)
    root_node = node(state, initial_policy)
    start_time = time.time()
    for _ in range(PV_EVALUATE_COUNT): 
        root_node.evaluate()
        # print(f"Node visits: {[child.n for child in root_node.child_nodes]}")
        #print(f"PUCT scores: {[PUCT_value(child) for child in root_node.child_nodes]}")
    end_time = time.time()
    total_nodes = root_node.count_nodes()
    tree_depth = root_node.count_tree_depth() - 1
    # print(f"Search time: {end_time-start_time}s")
    # print(f"Total MCTS nodes: {total_nodes}")
    # print(f"Max MCTS tree depth: {tree_depth}")
    
    scores = nodes_to_scores(root_node.child_nodes) # 取得當前節點所有子節點的次數
    #print("scores: ", scores)
    #visualize_tree(root_node)
    if temperature == 0: # 取最大值
        action = np.argmax(scores)
        scores = np.zeros(len(scores))
        scores[action] = 1
    else:
        scores = boltzman(scores, temperature)
    return scores

def pv_mcts_action(model, temperature=0): # 回傳函式
    def pv_mcts_action(state):
        scores = pv_mcts_scores(model, state, temperature, True)
        return np.random.choice(state.all_legal_actions(), p = scores)  # scores中只有最多拜訪次數的是1，其他是0
    return pv_mcts_action

def model_score(model):
    def score(state, path):
        reversed_path = path[::-1]
        policy, _ = predict(model, state, reversed_path)
        return np.random.choice(state.all_legal_actions(), p = policy)  # 根據每個動作的機率挑選
    return score

def visualize_tree(root_node):
    visits = [child.n for child in root_node.child_nodes]
    actions = range(len(visits))
    plt.bar(actions, visits)
    plt.xlabel("Actions")
    plt.ylabel("Visit Counts")
    plt.title("PV-MCTS Visit Distribution")
    plt.show()

if __name__ == "__main__":
    model = torch.jit.load('./model/best.pt')

    state = State() # 產生新遊戲
    next_action = pv_mcts_action(model, 1.0)
    while True:
        if state.is_done():
            break
        action = next_action(state)
        state = state.next(action)
        print(state)