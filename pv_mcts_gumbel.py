from game import State
from dual_network import DN_INPUT_SHAPE
from math import sqrt, log
import torch    
import numpy as np
import math
# Q值越大的動作會模擬越多次
PV_EVALUATE_COUNT = 200 
TOPM = 16
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
    policies = y[0][0][list(state.all_legal_actions())]
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
def nodes_to_softmax(child_nodes, model):  #傳入root的child_nodes，將模擬後的子節點們帶入vmix公式計算價值
    vmix = []
    q_sum = 0    
    for node in child_nodes:    # 加總每個節點的(機率*q值)
        if node.n > 0:
            player = node.state.get_player_order()
            q_sum += node.p * (node.w[player] / node.n)

    for node in child_nodes:        
        player = node.state.get_player_order()        
        
        if node.n > 0:  # 有拜訪過的節點
            Nb = sum(c.n for c in node.child_nodes) # 該節點的所有子節點拜訪次數加總
            Pb = sum(c.p for c in node.child_nodes) # 該節點的所有子節點機率值加總    
            vmix_value = 1 / (1 + Nb) * (node.v[player] + (Nb / Pb) * q_sum) 
        else:           # 沒有拜訪過的節點 
            _, value = predict(model, node.state)   # 網路預測
            vmix_value = 1 * value[player]

        vmix.append(vmix_value)
    logits = []
    for c in child_nodes:
        logits.append(c.logits)
    print("vmix: ", vmix)       # 目前的vmix每個動作差異很小，怪怪的待驗證
    print("logits: ", logits)   # 測試是否大約等於probabilities，如是則代表vmix沒效果
    softmax_input = logits + np.array(vmix)  # 將logits與vmix相加作為softmax的輸入
    probabilities = np.exp(softmax_input - np.max(softmax_input))  # 進行softmax操作
    probabilities /= probabilities.sum()
    # print("probabilities: ", probabilities)
    return probabilities

def pv_mcts_scores(model, state, temperature):
    class node:
        def __init__(self, state, p, logits, gumbel, is_root=False):
            self.state = state # 盤面
            self.p = p # 策略
            self.logits = logits  # Logits approximated from softmax probabilities
            self.gumbel = gumbel  # Gumbel noise applied to logits          
            self.is_root = is_root
            self.n = 0 # 場數
            self.w = [0,0,0]    # 價值會累加
            self.v = [0,0,0]    # 紀錄網路預測的價值(不會累加)
            self.child_nodes = None # 下一回合盤面可能性         

        def evaluate(self):            
            if self.state.is_done(): # 終局
                value = np.array(self.state.finish())                
                self.w += value
                self.n += 1
                self.v = value
                return value
        
            if not self.child_nodes: # 當前node沒有子節點 -> expand
                policies, value = predict(model, self.state)
                self.w += value
                self.n += 1
                self.v = value
                self.child_nodes = []
                logits_approx = np.log(policies + 1e-10)  # Log-transform of probabilities to approximate logits
                gumbels = np.random.gumbel(loc=0.0, scale=1.0, size=len(policies))  # Sample Gumbel noise for each action      
                # 展開該節點的子節點並把對應的各項值加進去
                for action, policy, logit, gumbel in zip(self.state.all_legal_actions(), policies, logits_approx, gumbels):
                    self.child_nodes.append(node(self.state.next(action), policy, logit, gumbel, is_root=False))

                if self.is_root:   # if its root
                    self.is_root = False
                    m = TOPM                    
                    # Initial selection based on gumbel + logits
                    # 第一階段先用網路預測過一次取前m個
                    sorted_child_nodes = sorted(self.child_nodes, key=lambda c: c.gumbel + c.logits, reverse=True)[:m]
                    # 再將前m個丟去第二階段
                    selected_action = subsequent_halving_steps(sorted_child_nodes, m)                   
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
    
    def subsequent_halving_steps(child_nodes, m):
        """
        要先每個節點模擬N次，N=math.floor(PV_EVALUATE_COUNT/logm*m)
        從第二次挑选开始的Sequential Halving步骤。
        根据g + logits + σ(q(a))进行进一步选择，直到剩余一个动作。
        """
        if len(child_nodes) == 1:
            return child_nodes
        N_sum = 0
        m = len(child_nodes)
        c_m = m
        while c_m > 1:
            if c_m > 2:
                Na = math.floor(PV_EVALUATE_COUNT/(math.log2(m)*c_m))   # log會有小數，導致模擬次數不到200
            else:   # last 2 actions
                Na = (PV_EVALUATE_COUNT - N_sum)//2
            print("N(a): ", Na)                                   
            for child_node in child_nodes:
                for _ in range(Na): 
                    child_node.evaluate()   #把每個child當根節點去模擬N次
            N_sum += Na * c_m
            c_m //= 2 
            # 計算scores並排序            
            scores = [(child.gumbel + child.logits + scale_adjusted_q(child), i)
                    for i, child in enumerate(child_nodes)]
            print("len(scores): ",len(scores), "c_m: ", c_m)
            scores.sort(reverse=True, key=lambda x: x[0])       # 基於元組中的第0個元素來排序。reverse=true由大到小
            child_nodes = [child_nodes[i] for _, i in scores[:c_m]] # 挑前半            
        
        return child_nodes     # 剩一個
    
    def scale_adjusted_q(child):
        """
        根据σ公式计算q的调整值。
        σ(q(a)) = (c_visit + max(N(b))) * scale * q(a)
        其中N(b)是同级所有节点的访问次数的最大值。
        """
        player = child.state.get_player_order()
        q = child.w[player] / child.n
        max_n = max(ch.n for ch in child.child_nodes)  # 獲取該節點的最多訪問次數的子節點的訪問次數max(N(b))
        c_visit = 50
        scale = 100
        scale_adjusted_q = (c_visit + max_n) * scale * q    # 如何驗證Q值的值域
        print("q: ", scale_adjusted_q)
        return scale_adjusted_q      
    
    # root做mcts模擬直到剩下一個動作
    root_node = node(state, 1.0, 0.0, 0.0, True)
    
    #for _ in range(PV_EVALUATE_COUNT): 
    root_node.evaluate()
    
    if len(root_node.child_nodes) != 1:
        scores = nodes_to_softmax(root_node.child_nodes, model)  # 要得到softmax過後的機率分佈
    else:
        scores = [1.0]
    # scores = nodes_to_scores(root_node.child_nodes) # 原本的是取得當前節點所有子節點的次數當作分數

    if temperature == 0: # 取最大值
        print("scores(softmax(logits+vmix)): ", scores)
        action = np.argmax(scores)
        scores = np.zeros(len(scores))  # 初始化一個元素0的串列長度與scores相同        
        scores[action] = 1              # 將第action個元素設為1
        
        return scores
    else:      
        # scores = boltzman(scores, temperature)
        # print("len(scores): ", len(scores)) 
        print("scores(softmax(logits+vmix)): ", scores)
        return scores
        # if isinstance(scores, np.ndarray):          
        #     return scores
        # else:
        #     return scores.numpy()    

def pv_mcts_action(model, temperature=0): # 回傳函式
    def pv_mcts_action(state):
        scores = pv_mcts_scores(model, state, temperature)
        return np.random.choice(state.all_legal_actions(), p = scores)
    return pv_mcts_action

if __name__ == "__main__":
    model = torch.jit.load('./model/best.pt')

    state = State()
    next_action = pv_mcts_action(model, 1.0)
    while True:
        if state.is_done():
            break
        action = next_action(state)
        state = state.next(action)
        print(state)