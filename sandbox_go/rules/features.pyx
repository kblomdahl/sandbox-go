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

import cython

@cython.boundscheck(False)
cdef void get_features(Board board, int color, float[:,:] out) nogil:
    """ Returns the input features for the given board state and player """

    cdef float is_black = 1.0 if color == BLACK else 0.0
    cdef float is_white = 1.0 if color == WHITE else 0.0
    cdef int other = opposite(color)
    cdef int x, y, index

    for y in range(19):
        for x in range(19):
            index = 19 * y + x

            out[0, index] = is_black
            out[1, index] = is_white

            if board.vertices[index] == color:
                out[2, index] = 1.0
            elif board.vertices[index] == other:
                out[3, index] = 1.0
            else:
                out[4, index] = 1.0
