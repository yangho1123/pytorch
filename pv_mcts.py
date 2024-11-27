from game import State
from dual_network import DN_INPUT_SHAPE
from math import sqrt
import torch    
import numpy as np
import time

PV_EVALUATE_COUNT = 500
# 記得改all_legal_action
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

def predict(model, state): # 利用對偶網路做下一步的預測
    a, b, c = DN_INPUT_SHAPE
    # 輸入統一為紅、綠、藍
    if state.get_player_order() == 0:
        x = torch.tensor([state.mine_pieces, state.next_pieces, state.prev_pieces]).float()
    if state.get_player_order() == 1:
        x = torch.tensor([state.prev_pieces, state.mine_pieces, state.next_pieces]).float()
    if state.get_player_order() == 2:
        x = torch.tensor([state.next_pieces, state.prev_pieces, state.mine_pieces]).float()
    x = x.reshape(1, a, b, c)
    
    device = get_device()
    model.to(device)
    with torch.no_grad(): # 預測不需要計算梯度
        x = x.to(device)
        y = model(x)

    #print(y[0][0])
    policies = y[0][0][list(state.legal_actions())]
    policies /= sum(policies) if sum(policies) else 1 # 總合為0就除以1

    value = y[1][0]
    
    policies = policies.cpu().numpy()
    value = value.cpu().numpy()
    # print(policies)
    # print(value)
    return policies, value


def nodes_to_scores(nodes): # 回傳某節點的所有子節點的拜訪次數
    scores = []
    for c in nodes:
        scores.append(c.n)
    return scores


def boltzman(xs, temperature):
    xs = [x ** (1 / temperature) for x in xs]
    return [x / sum(xs) for x in xs]

def pv_mcts_scores(model, state, temperature):
    class node:
        def __init__(self, state, p):
            self.state = state # 盤面
            self.p = p # 策略
            self.n = 0 # 場數
            self.w = [0,0,0] # score
            self.child_nodes = None # 下一回合盤面可能性

        def evaluate(self):
            if self.state.is_done(): # 終局
                value = np.array(self.state.finish())                
                self.w += value
                self.n += 1
                return value
        
            if not self.child_nodes: # 當前node沒有子節點 -> expand
                policies, value = predict(model, self.state)
                self.w += value
                self.n += 1
                self.child_nodes = []
                for action, policy in zip(self.state.legal_actions(), policies):
                    self.child_nodes.append(node(self.state.next(action), policy))
                return value
            
            else: # 子節點存在
                value = self.next_child_node().evaluate() # 取得子節點算出來的分數(負號要改)                
                self.w += value
                self.n += 1
                return value
            
        def next_child_node(self): # select
            player = self.state.get_player_order()
            C_PUCT = 1.0
            t = sum(nodes_to_scores(self.child_nodes))
            pucb_values = []
            for child_node in self.child_nodes:
                pucb_values.append((child_node.w[player] / child_node.n if child_node.n else 0.0) +
                                   C_PUCT * child_node.p * sqrt(t) / (1 + child_node.n))    #(價值的負號取消)
            
            return self.child_nodes[np.argmax(pucb_values)] # 回傳PUCT分數最大者
  
    root_node = node(state, 0)

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
        scores = pv_mcts_scores(model, state, temperature)
        return np.random.choice(state.legal_actions(), p = scores)
    return pv_mcts_action


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