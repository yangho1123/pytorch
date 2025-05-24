from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "MCTS",
        ["MCTS.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    ),
    Extension(
        "SelfPlayAgent",
        ["SelfPlayAgent.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    ),
]

setup(
    name="fast-alphazero-tri-othello",
    ext_modules=cythonize(extensions, language_level=3),
) 