# Cython 加速版 MCTS 算法

這個目錄包含使用 Cython 優化的蒙特卡羅樹搜索 (MCTS) 算法實現，用於 AlphaZero 三人遊戲。

## 特點

- 使用 Cython 加速計算密集型部分
- 類型定義和邊界檢查消除提升性能
- 保持與原始 Python 版本相同的功能
- 支持批處理模式以提高 GPU 利用率

## 編譯指南

要編譯 Cython 模塊，請按照以下步驟操作：

### 前提條件

確保您已安裝了以下軟件包：

```bash
pip install cython numpy torch setuptools
```

如果需要編譯 C 擴展，還需要安裝 C 編譯器（如 GCC 或 Visual C++）。

### 編譯步驟

1. 進入 `cython_version` 目錄：

```bash
cd newfeature/cython_version
```

2. 運行編譯命令：

```bash
python setup.py build_ext --inplace
```

這將在當前目錄生成 `pv_mcts_3his_fast_cy.*.so` 文件（在 Linux/Mac 上）或 `pv_mcts_3his_fast_cy.*.pyd` 文件（在 Windows 上）。

## 使用方法

在編譯完成後，您可以直接運行自我對弈腳本：

```bash
python self_play_3his_1value_test.py
```

## 性能對比

Cython 版本相比原始 Python 版本，在 MCTS 搜索部分可以獲得顯著的速度提升：

- 減少了解釋器開銷
- 通過類型定義和邊界檢查消除提高效率
- 使用 C 級別的運算提升性能

## 注意事項

- 確保 Python 模塊和 Cython 模塊使用相同版本的依賴庫
- 如果修改了 `.pyx` 文件，需要重新編譯
- 在不同操作系統上編譯的擴展模塊不能通用

## 故障排除

如果遇到導入錯誤，可能是編譯過程中出現問題，請嘗試：

1. 檢查是否生成了 `.so` 或 `.pyd` 文件
2. 確認編譯時沒有錯誤訊息
3. 檢查 Python 和所有依賴庫的版本是否兼容 