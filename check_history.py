import pickle
import numpy as np
import os

def read_and_write_history(pickle_file_path, output_file_path):
    # 讀取pickle文件
    with open(pickle_file_path, 'rb') as f:
        history = pickle.load(f)
    
    # 打開輸出文本文件
    with open(output_file_path, 'w', encoding='utf-8') as out_file:
        for game_step, (state, policy, value) in enumerate(history):
            out_file.write(f"步驟 {game_step + 1}:\n")
            
            # 寫入狀態
            out_file.write("狀態:\n")
            for i, layer in enumerate(state):
                if i < 3:
                    layer_name = ["紅色", "綠色", "藍色"][i]
                else:
                    layer_name = "當前局面輪到哪個玩家下棋"
                out_file.write(f"  {layer_name}:\n")
                np.savetxt(out_file, layer, fmt='%d', delimiter=' ')
            
            # 寫入策略（動作概率分佈）
            out_file.write("策略 (動作概率分佈):\n")
            non_zero_policies = [(i, p) for i, p in enumerate(policy) if p > 0]
            for action, prob in non_zero_policies:
                out_file.write(f"  動作 {action}: {prob:.4f}\n")
            
            # 寫入價值
            out_file.write(f"價值: {value}\n")
            
            out_file.write("\n" + "="*50 + "\n\n")

# 使用示例
if __name__ == "__main__":
    # 指定pickle文件路徑
    pickle_dir = '../torchdata/'
    pickle_files = [f for f in os.listdir(pickle_dir) if f.endswith('.history')]
    
    if not pickle_files:
        print("未找到.history文件")
    else:
        # 選擇最新的文件
        latest_file = max(pickle_files)
        pickle_file_path = os.path.join(pickle_dir, latest_file)
        
        # 指定輸出文本文件路徑
        output_file_path = 'history_content.txt'
        
        read_and_write_history(pickle_file_path, output_file_path)
        print(f"已將歷史記錄內容寫入 {output_file_path}")