from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
setup(
    include_dirs = [np.get_include()],
    ext_modules = cythonize("umatrix_routine.pyx")
)
