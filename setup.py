#from setuptools import setup
#from Cython.Build import cythonize
#import Cython.Build
#import numpy as np
#setup(name = "pyDANDIA",
#    include_dirs = [np.get_include()], 
#    setup_requires=["Cython>=0.2"],
#    ext_modules = cythonize("./pyDANDIA/umatrix_routine.pyx"),
#    cmdclass={'build_ext': Cython.Build.build_ext}
#)






from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy as np

NAME = "pyDANDIA"
VERSION = "0.1"
DESCR = "pyDANDIA DIA CODE, AMAZING :)"
URL = "https://github.com/pyDANDIA/"
REQUIRES = ['numpy', 'cython']

AUTHOR = "The ROBONET TEAM"
EMAIL = ""

LICENSE = ""

SRC_DIR = "pyDANDIA"
PACKAGES = [SRC_DIR]

ext_1 = Extension('umatrix_routine',
                  [SRC_DIR + "/umatrix_routine.pyx"],
                  libraries=[],
                  include_dirs=[np.get_include()])


EXTENSIONS = [ext_1]

if __name__ == "__main__":
    setup(install_requires=REQUIRES,
          packages=PACKAGES,
          zip_safe=False,
          name=NAME,
          version=VERSION,
          description=DESCR,
          author=AUTHOR,
          author_email=EMAIL,
          url=URL,
          license=LICENSE,
          cmdclass={"build_ext": build_ext},
          ext_modules=EXTENSIONS
)
