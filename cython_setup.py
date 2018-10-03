import os
from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(ext_modules=cythonize("cython_func.pyx"),
      include_dirs=[numpy.get_include(), os.path.join(numpy.get_include(), 'numpy')])