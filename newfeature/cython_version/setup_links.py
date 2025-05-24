#!/usr/bin/env python
import os
import sys
import shutil
import platform

def print_status(message):
    print(f">>> {message}")

def create_symlink(src, dst):
    """創建符號鏈接或文件副本"""
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)

    # 檢查源文件是否存在
    if not os.path.exists(src):
        print_status(f"錯誤: 源文件不存在: {src}")
        return False

    # 如果目標已存在，刪除它
    if os.path.exists(dst):
        if os.path.islink(dst) or os.path.isfile(dst):
            os.remove(dst)
        elif os.path.isdir(dst):
            shutil.rmtree(dst)

    # 創建符號鏈接或副本
    try:
        if platform.system() == 'Windows':
            # Windows 可能需要管理員權限創建符號鏈接，所以我們直接複製文件
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            print_status(f"已複製: {src} -> {dst}")
        else:
            # Linux/Mac 創建符號鏈接
            os.symlink(src, dst)
            print_status(f"已創建符號鏈接: {src} -> {dst}")
        return True
    except Exception as e:
        print_status(f"錯誤: 無法創建鏈接或副本: {e}")
        return False

def setup_dependencies():
    """設置 Cython 版本所需的依賴關係"""
    print_status("設置 Cython 版本所需的依賴關係...")

    # 獲取當前目錄和父目錄
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    # 需要鏈接的文件列表
    files_to_link = [
        "game_nogroup.py",
        "dual_network_3his.py"
    ]

    # 創建符號鏈接
    success = True
    for file in files_to_link:
        src = os.path.join(parent_dir, file)
        dst = os.path.join(current_dir, file)
        if not create_symlink(src, dst):
            success = False

    if success:
        print_status("所有依賴關係已成功設置!")
    else:
        print_status("警告: 有些依賴關係設置失敗。請查看上面的錯誤訊息。")

    return success

if __name__ == "__main__":
    setup_dependencies() 