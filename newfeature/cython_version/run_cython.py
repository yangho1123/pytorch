#!/usr/bin/env python
import os
import sys
import subprocess
import time
import importlib.util

def print_header(message):
    print("\n" + "=" * 80)
    print(f" {message}")
    print("=" * 80)

def run_command(command, cwd=None):
    """執行命令並實時顯示輸出"""
    print(f"執行: {command}")
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        cwd=cwd
    )
    
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
    
    process.stdout.close()
    return_code = process.wait()
    if return_code:
        print(f"命令執行失敗，退出碼: {return_code}")
        return False
    return True

def check_cython_installed():
    """檢查是否安裝了 Cython"""
    try:
        import cython
        print(f"檢測到 Cython 版本: {cython.__version__}")
        return True
    except ImportError:
        print("未找到 Cython。請使用 pip install cython 安裝。")
        return False

def check_numpy_installed():
    """檢查是否安裝了 NumPy"""
    try:
        import numpy
        print(f"檢測到 NumPy 版本: {numpy.__version__}")
        return True
    except ImportError:
        print("未找到 NumPy。請使用 pip install numpy 安裝。")
        return False

def check_torch_installed():
    """檢查是否安裝了 PyTorch"""
    try:
        import torch
        print(f"檢測到 PyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA 版本: {torch.version.cuda}")
            print(f"GPU 設備: {torch.cuda.get_device_name(0)}")
        return True
    except ImportError:
        print("未找到 PyTorch。請使用官方指南安裝: https://pytorch.org/get-started/locally/")
        return False

def setup_dependencies():
    """設置依賴關係"""
    print_header("設置依賴關係")
    
    # 檢查 setup_links.py 是否存在
    script_dir = os.path.dirname(os.path.abspath(__file__))
    setup_links_path = os.path.join(script_dir, "setup_links.py")
    
    if os.path.exists(setup_links_path):
        print("找到 setup_links.py，正在設置依賴關係...")
        # 導入並運行 setup_links 模塊
        try:
            spec = importlib.util.spec_from_file_location("setup_links", setup_links_path)
            setup_links = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(setup_links)
            
            if hasattr(setup_links, "setup_dependencies"):
                result = setup_links.setup_dependencies()
                if result:
                    print("依賴關係設置成功!")
                    return True
                else:
                    print("依賴關係設置失敗，請手動運行 setup_links.py")
                    return False
            else:
                print("setup_links.py 中沒有找到 setup_dependencies 函數")
                return False
        except Exception as e:
            print(f"運行 setup_links.py 時出錯: {e}")
            return False
    else:
        print(f"警告: 找不到 setup_links.py 文件: {setup_links_path}")
        print("請確保 game_nogroup.py 和 dual_network_3his.py 文件可用")
        return True  # 繼續執行，可能用戶已經手動設置了依賴

def compile_cython_module():
    """編譯 Cython 模組"""
    print_header("編譯 Cython 模組")
    
    # 獲取當前腳本的目錄
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 檢查 .pyx 文件是否存在
    pyx_file = os.path.join(script_dir, "pv_mcts_3his_fast.pyx")
    if not os.path.exists(pyx_file):
        print(f"錯誤: 找不到 Cython 源文件: {pyx_file}")
        return False
    
    # 檢查 setup.py 文件是否存在
    setup_file = os.path.join(script_dir, "setup.py")
    if not os.path.exists(setup_file):
        print(f"錯誤: 找不到 setup.py 文件: {setup_file}")
        return False
    
    # 執行編譯命令
    build_command = f"{sys.executable} setup.py build_ext --inplace"
    return run_command(build_command, cwd=script_dir)

def run_self_play():
    """運行自我對弈腳本"""
    print_header("運行自我對弈")
    
    # 獲取當前腳本的目錄
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 檢查自我對弈腳本是否存在
    self_play_script = os.path.join(script_dir, "self_play_3his_1value_test.py")
    if not os.path.exists(self_play_script):
        print(f"錯誤: 找不到自我對弈腳本: {self_play_script}")
        return False
    
    # 檢查編譯後的模組是否存在
    compiled_module_found = False
    for filename in os.listdir(script_dir):
        if filename.startswith("pv_mcts_3his_fast_cy") and (filename.endswith(".so") or filename.endswith(".pyd")):
            compiled_module_found = True
            print(f"找到編譯後的模組: {filename}")
            break
    
    if not compiled_module_found:
        print("警告: 未找到編譯後的 Cython 模組。請確保編譯成功。")
        response = input("是否繼續運行? (y/n): ")
        if response.lower() != 'y':
            return False
    
    # 運行自我對弈腳本
    run_command = f"{sys.executable} self_play_3his_1value_test.py"
    return run_command(run_command, cwd=script_dir)

def main():
    """主函數"""
    print_header("Cython 加速版 MCTS 自動編譯和運行")
    
    # 檢查必要的包是否已安裝
    if not all([
        check_cython_installed(),
        check_numpy_installed(),
        check_torch_installed()
    ]):
        print("請安裝所有必要的依賴後再運行此腳本。")
        return
    
    # 設置依賴關係
    if not setup_dependencies():
        print("無法設置依賴關係，腳本可能無法正常運行。")
        response = input("是否繼續? (y/n): ")
        if response.lower() != 'y':
            return
    
    # 詢問用戶要執行的操作
    print("\n選擇操作:")
    print("1. 僅編譯 Cython 模組")
    print("2. 編譯並運行自我對弈")
    print("3. 僅運行自我對弈 (假設已編譯)")
    print("q. 退出")
    
    choice = input("\n您的選擇: ")
    
    if choice == '1':
        compile_cython_module()
    elif choice == '2':
        if compile_cython_module():
            print("\n編譯成功! 準備運行自我對弈...\n")
            time.sleep(2)  # 讓用戶有時間閱讀編譯訊息
            run_self_play()
    elif choice == '3':
        run_self_play()
    elif choice.lower() == 'q':
        print("退出程序")
    else:
        print("無效的選擇")

if __name__ == "__main__":
    main() 