#!/usr/bin/env python3
# MIT License
# 
# Copyright (c) 2018 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy as np

extensions = [
    Extension('sandbox_go.rules.color', ['sandbox_go/rules/color.pyx'], include_dirs=['.', np.get_include()]),
    Extension('sandbox_go.rules.board', ['sandbox_go/rules/board.pyx'], include_dirs=['.', np.get_include()]),
    Extension('sandbox_go.rules.features', ['sandbox_go/rules/features.pyx'], include_dirs=['.', np.get_include()]),
    Extension('sandbox_go.sgf', ['sandbox_go/sgf.pyx'], include_dirs=['.', np.get_include()]),
]

setup(
    name='sandbox_go',
    packages=['sandbox_go', 'sandbox_go.rules'],
    ext_modules=cythonize(extensions)
)
