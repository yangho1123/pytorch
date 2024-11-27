from game import State
import torch
from dual_network import DN_INPUT_SHAPE,DN_FILTERS,DN_OUTPUT_SIZE,DN_RESIDUAL_NUM
from dual_network import DualNetwork
from math import sqrt
from pathlib import Path
import numpy as np
import time,copy

PV_EVALUATE_COUNT = 50 #模擬次數(原始為1600)

def predict(model, state):
    #states = [copy.deepcopy(state) for _ in range(2, 11)]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a, b, c = DN_INPUT_SHAPE
    
    x = torch.Tensor([state.mine_pieces, state.next_pieces, state.prev_pieces]).float()
    x = x.reshape(1, a, b, c)
    # 重塑為 (1, 通道數, 高度, 寬度)

    # 將NumPy數組轉換為PyTorch張量
    x = torch.tensor(x, dtype=torch.float32).to(device)
    model = model.to(device)
    # 確保模型處於評估模式
    model.eval()
    start_time = time.time()
    with torch.no_grad():  # 在這個塊中，不計算梯度
        # 進行預測
        policies, value = model(x)

    end_time = time.time()
    total_time = end_time - start_time 
    #print(f"GPU處理時間: {total_time} 秒")
    
    # 將策略輸出從PyTorch張量轉換為NumPy數組
    policies = policies.cpu().numpy()

    # 獲取合法動作的策略
    legal_policies = policies[0][list(state.legal_actions())]
    legal_policies /= np.sum(legal_policies) if np.sum(legal_policies) else 1  # 轉換為概率分佈

    # 將值輸出從PyTorch張量轉換為標量
    #value = value.item()
    value = value[0].item()
    

    return legal_policies, value

def nodes_to_scores(nodes):
    scores = []
    for c in nodes:
        scores.append(c.n)    # 將節點的試驗次數依序存入串列中
    return scores

# 取得試驗次數最多的棋步：傳入模型、盤面、溫度參數
def pv_mcts_scores(model, state, temperature):    
    class node: #定義MCTS的節點
        def __init__(self, state, p):
            self.state = state        # 盤面狀態
            self.p = p                # 策略
            self.w = 0                # 累計價值
            self.n = 0                # 試驗次數
            self.child_nodes = None   # 子節點群
        
        def evaluate(self):           # 將節點傳進來計算試驗次數、累計價值
            
            if self.state.is_done():  # 遊戲結束時
                value = np.array(self.state.finish())             
                # value = -1 if self.state.is_lose() else 0
                self.w += value[self.state.get_player_order()]               
                self.n += 1
                return value
                
            if not self.child_nodes:  # 子節點不存在時，以神經網路預測(評估與更新)
                policies, value = predict(model, self.state)
                self.w += value[self.state.get_player_order()] # 更新累計價值
                self.n += 1     # 更新試驗次數                
                # 擴充子節點
                self.child_nodes = []       # 利用網路預測的先驗機率擴充子節點
                for action, policy in zip(self.state.legal_actions(), policies):
                    self.child_nodes.append(node(self.state.next(action), policy))
                return value
            else: # 往下選擇
                value = -self.next_child_node().evaluate()  #以遞迴方式先找到PUCT最大子節點，再計算                
                self.w += value[self.state.get_player_order()]   # 更新累計價值
                self.n += 1       # 更新試驗次數
                return value
            
        def next_child_node(self):      # 取得PUCT最大的子節點
            C_PUCT = 1.0
            t = sum(nodes_to_scores(self.child_nodes))
            pucb_values = []
            for child_node in self.child_nodes:  # 計算所有子節點的PUCT值
                pucb_values.append((-child_node.w / child_node.n if child_node.n else 0.0) +
                                    C_PUCT * child_node.p * sqrt(t)/(1+child_node.n))

            return self.child_nodes[np.argmax(pucb_values)] # 回傳PUCT最大的子節點
        
    # 建構當前局勢的節點 (樹的根節點)
    root_node = node(state, 0)

    #  執行多次模擬 (總共要跑幾次mcts)
    for _ in range(PV_EVALUATE_COUNT):
        root_node.evaluate()

    # 取得合法棋步的機率分布
    scores = nodes_to_scores(root_node.child_nodes) # 取出該根節點的試驗次數
    
    if temperature == 0:
        action = np.argmax(scores)              #這三行在找出機率值最大的動作改成1，其他為0
        scores = np.zeros(len(scores))
        scores[action] = 1
    else:
        scores = boltzman(scores, temperature)  # 使用波茲曼分佈
    return scores

def pv_mcts_action(model, temperature=0): # 根據試驗次數(MCTS樹)取得真正要走的動作
    def pv_mcts_action(state):
        scores = pv_mcts_scores(model, state, temperature)
        return np.random.choice(state.legal_actions(), p=scores)
    return pv_mcts_action

def boltzman(xs, temperature):
    xs = [x ** (1 / temperature) for x in xs]
    return [x / sum(xs) for x in xs]

if __name__ == '__main__':
    model_files = sorted(Path('./model').glob('*.pt'))[-1]
    model = torch.jit.load('./model/best.pt')
    #model = DualNetwork(DN_INPUT_SHAPE, DN_FILTERS, DN_RESIDUAL_NUM, DN_OUTPUT_SIZE)
    if model_files:
        latest_model_path = model_files[-1]
        model.load_state_dict(torch.load(str(latest_model_path)))
    
    state = State()                             # 產生新的對局
    next_action = pv_mcts_action(model, 1.0)    # 取得動作
    while True:
        if state.is_done():
            break
        action = next_action(state)             # 取得動作
        state = state.next(action)              # 取得下一個盤面 
        print(state)