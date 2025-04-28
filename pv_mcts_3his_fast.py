from game import State
from game_nogroup import FlipTable
from dual_network_3his import DN_INPUT_SHAPE
from math import sqrt
import torch    
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import uuid  # 添加uuid库，用于节点唯一标识

PV_EVALUATE_COUNT = 1600
FLIPTABLE = FlipTable
# 添加批处理大小配置
BATCH_SIZE = 8  # 可以调整这个值，根据GPU内存大小

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

def prepare_input_tensor(state, path, history_length=11):
    """准备单个输入张量，从predict函数抽取出来方便批处理"""
    board_states = []
    # 输入统一为红、绿、蓝
    for i in range(history_length + 1):  # +1 to include the current state
        if len(path) > i:
            current_node = path[-i - 1]     # -i指向倒数第i个元素
            current_player = current_node.state.get_player_order()
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
            # 模型的输入状态始终按照红、绿、蓝的顺序
            state_tensor = torch.tensor([
                red_pieces, 
                green_pieces,
                blue_pieces
            ])
            # 创建玩家通道
            player_channel = torch.full((121,), current_player, dtype=torch.float32)
            FLIPTABLE_tensor = torch.tensor(FLIPTABLE, dtype=torch.float32).unsqueeze(0)
            full_state_tensor = torch.cat((state_tensor, player_channel.unsqueeze(0), FLIPTABLE_tensor), dim=0)
            board_states.append(full_state_tensor)
        else:
            # 如果没有历史状态，使用零张量
            zero_state = torch.zeros((5, 121))
            board_states.append(zero_state)
    
    # Stack all states
    x = torch.stack(board_states).float()
    x = x.view(1, 5 * (history_length + 1), 11, 11)
    return x

def predict(model, state, path): # 利用對偶網路做下一步的預測    
    x = prepare_input_tensor(state, path)
    x = x.to(get_device())
    model = model.to(get_device())
    
    with torch.no_grad(): # 預測不需要計算梯度        
        y = model(x)

    policy_softmax = torch.nn.functional.softmax(y[0][0], dim=0)   # 將policy輸出做softmax
    policies = policy_softmax[list(state.all_legal_actions())]
    policies /= sum(policies) if sum(policies) else 1 # 總合為0就除以1
    value = y[1][0]
    
    policies = policies.cpu().numpy()
    value = value.cpu().numpy()
    return policies, value.item()

def predict_batch(model, states_paths, legal_actions_list):
    """批量预测多个状态"""
    if not states_paths:
        return {}
    
    device = get_device()
    model = model.to(device)
    batch_tensors = []
    
    # 为每个状态准备输入张量
    for state, path in states_paths:
        x = prepare_input_tensor(state, path)
        batch_tensors.append(x)
    
    # 合并为一个批次
    batch_input = torch.cat(batch_tensors, dim=0).to(device)
    
    # 批量预测
    results = {}
    with torch.no_grad():
        y = model(batch_input)
    
    # 处理每个预测结果
    for i, (state, path) in enumerate(states_paths):
        policy_logits = y[0][i]
        value = y[1][i].cpu().numpy().item()
        
        # 对策略进行softmax
        policy_softmax = torch.nn.functional.softmax(policy_logits, dim=0)
        
        # 获取合法动作的策略
        legal_actions = legal_actions_list[i]
        policies = policy_softmax[legal_actions].cpu().numpy()
        
        # 归一化策略
        sum_p = np.sum(policies)
        if sum_p > 0:
            policies = policies / sum_p
        
        # 确保能够正确识别节点
        if len(path) > 0 and hasattr(path[-1], 'id'):
            node_id = path[-1].id
        else:
            # 使用一个不容易冲突的标识
            node_id = f"node_{i}_{hash(str(state.mine_pieces) + str(state.next_pieces) + str(state.prev_pieces))}"
            
        results[node_id] = (policies, value)
    
    return results

def nodes_to_scores(nodes): # 回傳某節點的所有子節點的拜訪次數
    scores = []
    for c in nodes:
        scores.append(c.n)
    return scores

def boltzman(xs, temperature):
    xs = [x ** (1 / temperature) for x in xs]
    return [x / sum(xs) for x in xs]

def pv_mcts_scores(model, state, temperature, add_noise=True, dirichlet_alpha=0.03):
    # 使用總棋子數判斷殘局
    # total_pieces = sum(1 for p in state.mine_pieces if p > 0) + \
    #               sum(1 for p in state.next_pieces if p > 0) + \
    #               sum(1 for p in state.prev_pieces if p > 0)
    
    # # 殘局階段（棋子數>70）降低批處理大小
    # if total_pieces > 79:
    #     current_batch_size = 1  # 殘局使用較小的批處理大小
    #     print(f"殘局階段（共{total_pieces}顆棋子）- 使用較小批處理大小: {current_batch_size}")
    # else:
    current_batch_size = BATCH_SIZE
    
    class Node:  # 将类名改为大写开头，避免与变量冲突
        def __init__(self, state, p, prev_node=None):
            self.state = state # 盤面
            self.id = str(uuid.uuid4())  # 添加唯一ID以便在批处理中识别节点
            if prev_node is None and add_noise:     # dirichlet noise用在root node
                epsilon = 0.25
                p = np.array(p)
                noise = np.random.dirichlet([dirichlet_alpha] * len(p))
                p = (1 - epsilon) * p + epsilon * noise
            self.p = p # 策略
            self.n = 0 # 場數
            self.w = 0 # score
            self.child_nodes = None # 下一回合盤面可能性
            self.parent = prev_node
            self.need_eval = False  # 标记是否需要评估
            self.eval_path = None   # 保存评估路径
        
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
        
        def evaluate_root(self, eval_queue=None):
            """評估根節點，收集需要評估的節點"""
            if self.state.is_done():  # 終局
                finish = np.array(self.state.finish())
                value = finish[self.state.get_player_order()]
                # 重要：只有尚未評估過的終局才增加計數
                if self.n == 0:
                    self.w += value
                    self.n += 1
                return value
                
            if not self.child_nodes:  # 當前node沒有子節點 -> expand
                if self.parent is None:  # root expand
                    path = []
                    initial_policy, _ = predict(model, self.state, path)  # 根节点的初始策略还是单独计算
                    self.child_nodes = []
                    for action, policy in zip(self.state.all_legal_actions(), initial_policy):
                        next_state = self.state.next(action)
                        self.child_nodes.append(Node(next_state, policy, self))
                    return 0
                else:
                    # 标记节点需要评估
                    self.need_eval = True
                    self.eval_path = self.trace_back()
                    if eval_queue is not None:
                        eval_queue.append(self)
                    return 0  # 临时返回0，后续会更新
                    
            else:  # 子節點存在
                # 檢查是否所有子節點都已評估過
                all_children_visited = all(child.n > 0 for child in self.child_nodes)
                if all_children_visited and len(self.child_nodes) > 0:
                    # 如果所有子節點都已訪問，但我們仍選擇了一個節點，可能是陷入了循環
                    pass
                
                next_node = self.next_child_node()
                value = next_node.evaluate_root(eval_queue)  # 继续向下搜索
                self.w += value
                self.n += 1
                return value
                
        def evaluate(self):
            """使用预计算的评估结果展开节点"""
            if self.need_eval and hasattr(self, 'eval_result'):
                policies, value = self.eval_result
                
                # 更新节点并创建子节点
                self.w += value
                self.n += 1
                self.child_nodes = []
                for action, policy in zip(self.state.all_legal_actions(), policies):
                    next_state = self.state.next(action)
                    self.child_nodes.append(Node(next_state, policy, self))
                
                # 清除临时数据
                self.need_eval = False
                self.eval_path = None
                delattr(self, 'eval_result')
                
                return value
                
            # 終局情況
            if self.state.is_done():
                finish = np.array(self.state.finish())
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
            
        def next_child_node(self): # select
            C_PUCT = sqrt(3)
            t = sum(nodes_to_scores(self.child_nodes))      #t=子節點拜訪次數的總和
            
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
            path = []
            current = self
            while current is not None:
                path.append(current)
                current = current.parent
            return path[::-1]  # Reverse to get path from root to this node
    
    # 初始化根节点
    path = []
    initial_policy, _ = predict(model, state, path)
    root_node = Node(state, initial_policy)
    
    # 记录运行时间
    start_time = time.time()
    
    # 主循环，使用批处理模式
    simulations_done = 0
    batch_counter = 0  # 统计批处理次数
    stalled_counter = 0  # 计数连续没有评估新节点的次数
    
    while simulations_done < PV_EVALUATE_COUNT:
        # 收集需要评估的节点
        eval_queue = []
        
        # 进行一定数量的模拟以填充评估队列
        num_batch_collect = min(current_batch_size, PV_EVALUATE_COUNT - simulations_done)   # 当前批次要收集的节点数量
        for _ in range(num_batch_collect):
            root_node.evaluate_root(eval_queue)
            
        # 如果有需要评估的节点，进行批量评估
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
            
            # 減少冗餘輸出
            # if original_queue_size > len(eval_queue):
            #     print(f"去重前節點數: {original_queue_size}, 去重後: {len(eval_queue)}, 去除了 {original_queue_size - len(eval_queue)} 個重複節點")
            
            #print(f"批處理評估 {len(eval_queue)} 個節點")
            
            # 批量預測
            states_paths = [(queue_node.state, queue_node.eval_path) for queue_node in eval_queue]
            legal_actions_list = [queue_node.state.all_legal_actions() for queue_node in eval_queue]
            
            # 批量預測
            eval_results = predict_batch(model, states_paths, legal_actions_list)
            
            # 确保每个节点都有对应的评估结果
            missing_nodes = [node.id for node in eval_queue if node.id not in eval_results]
            # if missing_nodes:
            #     print(f"警告: {len(missing_nodes)}/{len(eval_queue)} 個節點缺少評估結果")
            
            # 将预测结果应用到节点
            successful_evals = 0
            for queue_node in eval_queue:
                if queue_node.id in eval_results:
                    queue_node.eval_result = eval_results[queue_node.id]
                    # 完成节点的评估
                    queue_node.evaluate()
                    successful_evals += 1
            
            #print(f"成功評估 {successful_evals}/{len(eval_queue)} 個節點")
            simulations_done += successful_evals
            
            # 检测是否卡住
            if successful_evals == 0:
                stalled_counter += 1
                #print(f"警告: 第 {stalled_counter} 次沒有成功評估任何節點")
                
                if stalled_counter >= 3:
                    #print("檢測到搜索可能卡住，嘗試不同策略...")
                    # 重置评估队列
                    eval_queue = []
                    # 调整批处理大小为1，简化搜索
                    current_batch_size = 1
                    stalled_counter = 0
            else:
                stalled_counter = 0
        else:
            # 如果评估队列为空，可能是树已经完全展开，或搜索卡住
           #print("警告: 評估隊列為空")
            
            # 检查根节点状态 (只在第一次警告时输出详细信息)
            if stalled_counter == 0:
                legal_actions_count = len(state.all_legal_actions())
                expandable_nodes = sum(1 for child in root_node.child_nodes if child.n == 0)
                expanded_nodes = sum(1 for child in root_node.child_nodes if child.n > 0)
                #print(f"根節點狀態: 合法動作數 {legal_actions_count}, 可展開節點數 {expandable_nodes}, 已展開節點數 {expanded_nodes}")
                
                
            
            stalled_counter += 1
            
            if stalled_counter >= 3:
                #print("搜索似乎已完成或卡住，提前結束")
                break
    
    # 计算统计信息
    end_time = time.time()
    total_nodes = root_node.count_nodes()
    tree_depth = root_node.count_tree_depth() - 1
    
    # 添加模拟次数统计输出
    search_time = end_time - start_time
    completion_rate = (simulations_done / PV_EVALUATE_COUNT) * 100
    
    # 明確檢查是否達到目標模擬次數
    # if simulations_done < PV_EVALUATE_COUNT:
    #     warning_symbol = "!" * 20
    #     print(f"{warning_symbol}")
    #     print(f"警告: 搜索未完成目標模擬次數! 只完成了 {simulations_done}/{PV_EVALUATE_COUNT} 次 ({completion_rate:.2f}%)")
    #     print(f"{warning_symbol}")
    # else:
    #     print(f"搜索已完成: {simulations_done}/{PV_EVALUATE_COUNT} 次模擬 (100.00%)")
        
    # print(f"搜索耗時: {search_time:.2f}秒")
    # print(f"總批處理次數: {batch_counter}")
    # print(f"總節點數: {total_nodes}")
    # print(f"最大樹深度: {tree_depth}")
    
    # 获取根节点子节点的访问次数并计算得分
    scores = nodes_to_scores(root_node.child_nodes)
    # 输出访问分布统计 (精简输出，只显示关键信息)
    if len(scores) > 0:
        max_score = max(scores)
        total_score = sum(scores)
        if total_score > 0:
            max_ratio = (max_score / total_score) * 100
            # print(f"訪問分佈: 最高/總數 = {max_score}/{total_score} ({max_ratio:.2f}%)")
            # if max_ratio > 80:
            #     print(f"警告: 訪問高度集中在單一節點 ({max_ratio:.2f}%)")
        # 只在開發調試階段輸出詳細分數
        # print(f"分數: {scores}")
    
    # 应用温度参数
    if temperature == 0:  # 取最大值
        action = np.argmax(scores)
        scores = np.zeros(len(scores))
        scores[action] = 1
    else:
        scores = boltzman(scores, temperature)
        
    return scores

def pv_mcts_action(model, temperature=0): # 回傳函式
    def pv_mcts_action(state):
        scores = pv_mcts_scores(model, state, temperature, False)
        return np.random.choice(state.all_legal_actions(), p = scores)  # scores中只有最多拜訪次數的是1，其他是0
    return pv_mcts_action

# def model_score(model):
#     def score(state, path):
#         reversed_path = path[::-1]
#         policy, _ = predict(model, state, reversed_path)
#         return np.random.choice(state.all_legal_actions(), p = policy)  # 根據每個動作的機率挑選
#     return score

# def visualize_tree(root_node):
#     visits = [child.n for child in root_node.child_nodes]
#     actions = range(len(visits))
#     plt.bar(actions, visits)
#     plt.xlabel("Actions")
#     plt.ylabel("Visit Counts")
#     plt.title("PV-MCTS Visit Distribution")
#     plt.show()

# if __name__ == "__main__":
#     model = torch.jit.load('./model/best.pt')

#     state = State() # 產生新遊戲
#     next_action = pv_mcts_action(model, 1.0)
#     while True:
#         if state.is_done():
#             break
#         action = next_action(state)
#         state = state.next(action)
#         print(state)