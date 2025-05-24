# self_play_single.py - 單進程版本的自我對弈程序
import numpy as np
import torch
import os
import time
from tqdm import tqdm

# 導入必要的組件
from Game import Game
from MCTS import MCTS
from NNetWrapper import NNetWrapper

class Args:
    """簡化的參數類"""
    def __init__(self):
        # MCTS參數
        self.cpuct = 1.0
        self.numMCTSSims = 50  # 每步搜索的模擬次數
        self.tempThreshold = 15  # 溫度參數閾值
        self.temp = 1  # 初始溫度
        self.arenaTemp = 0.1  # Arena模式的溫度
        
        # 自我對弈參數
        self.gamesPerIteration = 5  # 測試階段減少遊戲數量
        self.symmetricSamples = True  # 是否使用對稱樣本
        
        # 專家價值權重
        class ExpertValueWeight:
            def __init__(self):
                self.current = 0.5
                self.start = 0.5
                self.end = 0.5
                self.iterations = 1
        self.expertValueWeight = ExpertValueWeight()
        
        # 調試選項
        self.debug = True  # 是否輸出詳細調試信息

class SingleProcessSelfPlay:
    """單進程自我對弈類"""
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        
    def execute_episode(self):
        """執行一個完整的對弈回合"""
        # 初始化遊戲狀態
        board = self.game.getInitBoard()
        player = 1
        step = 0
        history = []
        
        print("開始新的遊戲...")
        
        # 直到遊戲結束
        while self.game.getGameEnded(board, player) == 0:
            step += 1
            player_order = self.game.get_player_order(board)
            print(f"\n===== 回合 {step}, 玩家 {player_order} =====")
            
            # 準備當前狀態
            canonical_board = self.game.getCanonicalForm(board, player)
            
            # 計算合法動作
            valid_moves = self.game.getValidMoves(canonical_board, player)
            valid_count = sum(valid_moves)
            valid_indices = [i for i, v in enumerate(valid_moves) if v == 1]
            
            print(f"合法走步數量: {valid_count}")
            print(f"合法走步位置: {valid_indices}")
            
            # 進行MCTS搜索
            print(f"開始執行 {self.args.numMCTSSims} 次MCTS模擬...")
            mcts_stats = {
                'leaf_nodes': 0,
                'value_predictions': [],
                'policy_predictions': []
            }
            
            for sim in range(self.args.numMCTSSims):
                if self.args.debug and sim % 10 == 0:
                    print(f"  MCTS模擬 {sim}/{self.args.numMCTSSims}")
                
                # 尋找葉節點
                leaf = self.mcts.findLeafToProcess(canonical_board, False)
                
                if leaf is not None:
                    mcts_stats['leaf_nodes'] += 1
                    
                    # 準備輸入張量
                    input_tensor = self.prepare_input(leaf)
                    input_tensor = input_tensor.unsqueeze(0)  # 添加批次維度
                    
                    # 使用神經網絡預測
                    with torch.no_grad():
                        policy, value = self.nnet.predict(leaf)
                        
                    # 記錄策略和價值預測
                    mcts_stats['policy_predictions'].append(policy)
                    mcts_stats['value_predictions'].append(value[0])
                    
                    # 處理MCTS結果
                    self.mcts.processResults(policy, value[0])
            
            # 打印MCTS統計信息
            if self.args.debug:
                print(f"MCTS找到的葉節點數量: {mcts_stats['leaf_nodes']}")
                if mcts_stats['value_predictions']:
                    avg_value = sum(mcts_stats['value_predictions']) / len(mcts_stats['value_predictions'])
                    print(f"平均價值預測: {avg_value:.4f}")
                    
                    if len(mcts_stats['policy_predictions']) > 0:
                        # 取最後一個策略預測作為範例
                        sample_policy = mcts_stats['policy_predictions'][-1]
                        top_moves = sorted(enumerate(sample_policy), key=lambda x: x[1], reverse=True)[:5]
                        print(f"神經網絡的前5高概率動作:")
                        for pos, prob in top_moves:
                            print(f"  位置 {pos}: 概率 {prob:.4f} (是否合法: {valid_moves[pos] == 1})")
            
            # 獲取策略
            temp = self.args.temp if step < self.args.tempThreshold else self.args.arenaTemp
            policy = self.mcts.getExpertProb(canonical_board, temp=temp)
            
            # 確保策略合法
            policy = np.array(policy, dtype=np.float64)
            policy = np.nan_to_num(policy, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 處理總和為0的情況
            policy_sum = np.sum(policy)
            if policy_sum <= 0:
                print("警告: 策略總和為0，使用均勻分佈！")
                valid_moves = self.game.getValidMoves(canonical_board, 1)
                valid_count = sum(valid_moves)
                if valid_count > 0:
                    policy = np.array(valid_moves, dtype=np.float64) / valid_count
                else:
                    policy = np.zeros_like(policy)
                    policy[0] = 1.0
            else:
                policy = policy / policy_sum
            
            # 確保總和為1
            policy_sum = np.sum(policy)
            if abs(policy_sum - 1.0) > 1e-10:
                nonzero_indices = np.nonzero(policy)[0]
                if len(nonzero_indices) > 0:
                    last_idx = nonzero_indices[-1]
                    policy[last_idx] += 1.0 - policy_sum
            
            # 顯示MCTS策略
            if self.args.debug:
                print(f"MCTS策略總和: {np.sum(policy):.6f}")
                top_policy_moves = sorted(enumerate(policy), key=lambda x: x[1], reverse=True)[:5]
                print(f"MCTS的前5高概率動作:")
                for pos, prob in top_policy_moves:
                    print(f"  位置 {pos}: 概率 {prob:.4f} (是否合法: {valid_moves[pos] == 1})")
                
                if np.isnan(policy).any() or np.isinf(policy).any():
                    print("警告: 策略中存在NaN或Inf！")
            
            # 選擇動作
            try:
                action = np.random.choice(len(policy), p=policy)
                print(f"選擇動作: {action}")
            except ValueError as e:
                print(f"選擇動作時出錯: {e}")
                print(f"策略: {policy}")
                print(f"策略總和: {np.sum(policy)}")
                policy = np.ones_like(policy) / len(policy)
                action = np.random.choice(len(policy), p=policy)
            
            # 獲取MCTS專家價值評估
            expert_value = self.mcts.getExpertValue(canonical_board)
            print(f"MCTS專家價值評估: {expert_value:.4f}")
            
            # 保存訓練數據
            history.append((canonical_board, policy, expert_value, player))
            
            # 進行下一步
            board, player = self.game.getNextState(board, player, action)
        
        # 獲取遊戲結果
        game_result = self.game._calculate_result(board)
        print(f"\n遊戲結束，結果: {game_result}")
        
        # 打印棋子數量統計
        red_count = np.sum(board[0] == 1)
        green_count = np.sum(board[1] == 1)
        blue_count = np.sum(board[2] == 1)
        print(f"最終棋子數量 - 紅: {red_count}, 綠: {green_count}, 藍: {blue_count}")
        
        # 收集訓練數據
        training_data = []
        for hist_board, hist_policy, hist_value, hist_player in history:
            # 混合專家價值和實際結果
            player_order = self.game.get_player_order(hist_board)
            player_result = game_result[player_order]
            
            mixed_value = hist_value * self.args.expertValueWeight.current + \
                         player_result * (1 - self.args.expertValueWeight.current)
            
            # 應用對稱變換
            if self.args.symmetricSamples:
                symmetries = self.game.getSymmetries(hist_board, hist_policy)
                for sym_board, sym_policy in symmetries:
                    training_data.append((sym_board, sym_policy, mixed_value))
            else:
                training_data.append((hist_board, hist_policy, mixed_value))
        
        # 驗證訓練數據格式
        if self.args.debug and training_data:
            sample_data = training_data[0]
            print("\n訓練數據格式驗證:")
            print(f"棋盤狀態形狀: {sample_data[0].shape}")
            print(f"策略向量長度: {len(sample_data[1])}")
            print(f"價值標量: {sample_data[2]}")
            
        return training_data, game_result
    
    def prepare_input(self, board):
        """
        將遊戲狀態轉換為神經網絡輸入格式。
        
        參數:
            board: 遊戲狀態 (numpy 數組格式)
            
        返回:
            tensor: 適合神經網絡輸入的張量
        """
        # board 已经是 (5, 11, 11) 的 numpy 數組
        # 只需要返回前四個通道
        return torch.FloatTensor(board[0:4])
    
    def self_play(self):
        """執行自我對弈並收集訓練數據"""
        all_training_data = []
        results = []
        
        # 進行多場遊戲
        for i in tqdm(range(self.args.gamesPerIteration), desc="正在進行自我對弈"):
            # 重置MCTS搜索樹，確保每場遊戲都是獨立的
            self.mcts = MCTS(self.game, self.nnet, self.args)
            
            # 執行一場遊戲
            try:
                episode_data, result = self.execute_episode()
                all_training_data.extend(episode_data)
                results.append(result)
                
                print(f"完成第 {i+1}/{self.args.gamesPerIteration} 場遊戲")
                print(f"收集了 {len(episode_data)} 個訓練樣本")
                print(f"遊戲結果: {result}")
                print("-" * 50)
            except Exception as e:
                print(f"遊戲 {i+1} 執行過程中出錯: {e}")
                import traceback
                traceback.print_exc()
        
        # 統計結果
        red_wins = sum(1 for r in results if r[0] > 0)
        green_wins = sum(1 for r in results if r[1] > 0)
        blue_wins = sum(1 for r in results if r[2] > 0)
        draws = len(results) - red_wins - green_wins - blue_wins
        
        print("\n自我對弈統計:")
        print(f"總遊戲數: {len(results)}")
        print(f"紅方勝: {red_wins}")
        print(f"綠方勝: {green_wins}")
        print(f"藍方勝: {blue_wins}")
        print(f"平局: {draws}")
        
        return all_training_data

def main():
    print("初始化單進程自我對弈測試...")
    
    # 初始化參數
    args = Args()
    
    # 初始化遊戲和神經網絡
    game = Game()
    nnet = NNetWrapper(game)
    
    # 加載模型（如果存在）
    checkpoint_path = 'models/model_best.pth'
    if os.path.exists(checkpoint_path):
        nnet.load_checkpoint(folder='models', filename='model_best.pth')
        print(f"已加載模型: {checkpoint_path}")
        
        # 驗證模型是否正確加載
        try:
            # 創建一個測試輸入
            test_board = game.getInitBoard()
            test_input = torch.FloatTensor(test_board[0:4]).unsqueeze(0)
            
            # 進行測試預測
            with torch.no_grad():
                policy, value = nnet.predict(test_board)
            
            print(f"模型測試 - 策略形狀: {policy.shape}, 價值形狀: {value.shape}")
            print(f"模型測試 - 價值輸出: {value}")
            
            # 檢查策略總和
            policy_sum = np.sum(policy)
            print(f"模型測試 - 策略總和: {policy_sum:.6f}")
            
            # 檢查值範圍
            print(f"模型測試 - 策略最小值: {np.min(policy):.6f}, 最大值: {np.max(policy):.6f}")
            
        except Exception as e:
            print(f"模型測試失敗: {e}")
    else:
        print("沒有找到預訓練模型，使用隨機初始化的網絡")
    
    # 創建單進程自我對弈類
    sp = SingleProcessSelfPlay(game, nnet, args)
    
    # 執行自我對弈
    print("開始自我對弈...")
    training_data = sp.self_play()
    
    # 保存訓練數據（可選）
    save_data = True
    if save_data:
        output_dir = 'test_data'
        os.makedirs(output_dir, exist_ok=True)
        
        # 提取數據
        boards = []
        policies = []
        values = []
        
        for board, policy, value in training_data:
            boards.append(board)
            policies.append(policy)
            values.append(value)
        
        # 保存為文件
        np.save(f'{output_dir}/boards.npy', boards)
        np.save(f'{output_dir}/policies.npy', policies)
        np.save(f'{output_dir}/values.npy', values)
        
        # 驗證保存的數據
        try:
            # 加載保存的數據
            loaded_boards = np.load(f'{output_dir}/boards.npy')
            loaded_policies = np.load(f'{output_dir}/policies.npy')
            loaded_values = np.load(f'{output_dir}/values.npy')
            
            print(f"\n保存數據驗證:")
            print(f"boards 形狀: {loaded_boards.shape}")
            print(f"policies 形狀: {loaded_policies.shape}")
            print(f"values 形狀: {loaded_values.shape}")
            
            # 檢查隨機樣本
            if len(loaded_boards) > 0:
                idx = np.random.randint(0, len(loaded_boards))
                print(f"\n隨機樣本 {idx} 檢查:")
                print(f"棋盤形狀: {loaded_boards[idx].shape}")
                print(f"策略總和: {np.sum(loaded_policies[idx]):.6f}")
                print(f"價值: {loaded_values[idx]:.6f}")
        except Exception as e:
            print(f"數據驗證失敗: {e}")
        
        print(f"已保存 {len(training_data)} 個訓練樣本到 {output_dir}")
    
    print("測試完成！")

if __name__ == "__main__":
    main()
