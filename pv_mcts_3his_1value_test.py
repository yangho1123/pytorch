from game import State
from game_nogroup import FlipTable
from dual_network_3his import DN_INPUT_SHAPE
from math import sqrt
import torch    
import numpy as np

PV_EVALUATE_COUNT = 1600
FLIPTABLE = FlipTable
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

def predict(model, state, path): # 利用對偶網路做下一步的預測    
    history_length = 8
    board_states = []
    # 輸入統一為紅、綠、藍
    # Retrieve the last few states from the path
    for i in range(history_length + 1):  # +1 to include the current state
        if len(path) > i:
            current_node = path[-i - 1]     # -i指向倒數第i個元素，path的第一個元素是當前state，最後一個是root
            current_player = current_node.state.get_player_order()  # 用目前回合數來判斷當前mine是什麼顏色
            if current_player == 0:
                red_pieces = current_node.state.mine_pieces
                green_pieces = current_node.state.next_pieces
                blue_pieces = current_node.state.prev_pieces
            elif current_player == 1:
                red_pieces = current_node.state.prev_pieces
                green_pieces = current_node.state.mine_pieces
                blue_pieces = current_node.state.next_pieces
            else:
                red_pieces = current_node.state.next_pieces
                green_pieces = current_node.state.prev_pieces
                blue_pieces = current_node.state.mine_pieces
            # 模型的輸入狀態始终按照紅、綠、藍的順序
            state_tensor = torch.tensor([
                red_pieces, 
                green_pieces,
                blue_pieces
            ])
            # 创建玩家通道
            player_channel = torch.full((121,), current_player, dtype=torch.float32)
            FLIPTABLE_tensor = torch.tensor(FLIPTABLE, dtype=torch.float32).unsqueeze(0)  # 将FLIPTABLE转换为Tensor，并增加一个维度
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
    # print(policies)
    # print(value)
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
                p = np.array(p)
                noise = np.random.dirichlet([dirichlet_alpha] * len(p))
                p = 0.75 * p + 0.25 * noise
            self.p = p # 策略
            self.n = 0 # 場數
            self.w = 0 # score
            self.child_nodes = None # 下一回合盤面可能性
            self.parent = prev_node

        def evaluate(self):
            if self.state.is_done(): # 終局
                value = np.array(self.state.finish())
                value1 = value[self.state.get_player_order()]                
                self.w += value1 # 根據該節點深度(因為是終局大部分為0，但有少部分為1 or 2)
                self.n += 1
                
                return value1
        
            if not self.child_nodes: # 當前node沒有子節點 -> expand
                path = self.trace_back()
                policies, value = predict(model, self.state, path)
                self.w += value     # 模型預測的該節點的價值(-1~1)
                self.n += 1                
                self.child_nodes = []
                for action, policy in zip(self.state.all_legal_actions(), policies):
                    next_state = self.state.next(action)
                    self.child_nodes.append(node(next_state, policy, self))
                return value
            
            else: # 子節點存在
                value = self.next_child_node().evaluate() # 取得子節點算出來的分數   
                self.w += value
                self.n += 1
                return value
            
        def next_child_node(self): # select
            C_PUCT = 1.0
            t = sum(nodes_to_scores(self.child_nodes))
            pucb_values = []
            for child_node in self.child_nodes:
                pucb_values.append((child_node.w / child_node.n if child_node.n else 0.0) +
                                   C_PUCT * child_node.p * sqrt(t) / (1 + child_node.n))
            return self.child_nodes[np.argmax(pucb_values)] # 回傳PUCT分數最大者
        
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

    for _ in range(PV_EVALUATE_COUNT): 
        root_node.evaluate()

    scores = nodes_to_scores(root_node.child_nodes) # 取得當前節點所有子節點的次數

    if temperature == 0: # 取最大值
        action = np.argmax(scores)
        scores = np.zeros(len(scores))
        scores[action] = 1
    else:
        scores = boltzman(scores, temperature)
    return scores

def pv_mcts_action(model, temperature=0): # 回傳函式
    def pv_mcts_action(state):
        scores = pv_mcts_scores(model, state, temperature, False)
        return np.random.choice(state.all_legal_actions(), p = scores)
    return pv_mcts_action

def model_score(model):
    def score(state, path):
        reversed_path = path[::-1]
        policy, _ = predict(model, state, reversed_path)
        return np.random.choice(state.all_legal_actions(), p = policy)  # 根據每個動作的機率挑選
    return score

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