from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# 定義 Cython 模塊
extensions = [
    Extension(
        "pv_mcts_3his_fast_cy",  # 輸出模塊名
        ["pv_mcts_3his_fast.pyx"],  # 源文件
        include_dirs=[np.get_include()],  # NumPy 頭文件路徑
        libraries=[],  # 額外的庫
        extra_compile_args=["-O3"],  # 優化等級
    )
]

# 設置編譯參數
setup(
    name="pv_mcts_3his_fast_cy",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "cdivision": True,
        },
    ),
) 