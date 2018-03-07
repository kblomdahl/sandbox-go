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

from .rules.board cimport Board
from .rules.color cimport BLACK, WHITE
from .rules.features cimport get_features

import cython
from libc.stdlib cimport rand, RAND_MAX
cimport numpy as np
import numpy as np

@cython.boundscheck(False)
cdef int count_moves(unsigned char *line, int line_length) nogil:
    cdef int i = 5
    cdef int count = 0

    while i < line_length:
        if line[i] == 93 and line[i-3] == 91 and line[i-5] == 59:  # `;?[??]`
            if line[i-4] == 66 or line[i-4] == 87:  # `;B[??]` or `;W[??]`
                count += 1
        elif line[i] == 93 and line[i-1] == 91 and line[i-3] == 59:  # `;?[]`
            if line[i-2] == 66 or line[i-2] == 87:  # `;B[??]` or `;W[??]`
                count += 1

        i += 1

    return count

@cython.boundscheck(False)
cdef void _augment(int xx, int xy, int *x, int yx, int yy, int *y) nogil:
    cdef int x_ = x[0]
    cdef int y_ = y[0]
    cdef int cx = x_ - 9
    cdef int cy = y_ - 9

    x[0] = 9 + (xx * cx + xy * cy)
    y[0] = 9 + (yx * cx + yy * cy)

    assert x[0] >= 0 and x[0] < 19
    assert y[0] >= 0 and y[0] < 19

@cython.boundscheck(False)
cdef void augment(int s, int *x, int *y) nogil:
    if s == 0:  # identity
        pass
    elif s == 1:  # flip across horizontal axis
        _augment(-1, 0, x, 0, 1, y)
    elif s == 2:  # flip across the vertical axis
        _augment(1, 0, x, 0, -1, y)
    elif s == 3:  # flip across the main diagonal
        _augment(0, 1, x, 1, 0, y)
    elif s == 4:  # flip across the anti diagonal
        _augment(0, -1, x, -1, 0, y)
    elif s == 5:  # rotate 90 degrees clockwise
        _augment(0, 1, x, -1, 0, y)
    elif s == 6:  # rotate 180 degrees clockwise
        _augment(-1, 0, x, 0, -1, y)
    elif s == 7:  # rotate 270 degrees clockwise
        _augment(0, -1, x, 1, 0, y)

@cython.boundscheck(False)
cdef int _one(Board board, unsigned char *line, int line_length) nogil:
    # count the number of moves played without doing anything, this is done so
    # that we do not need to playout more of the game than we need to.
    #
    # It also helps since it allows us to discard games that are too short, and
    # the winner of said game is therefore uncertain.
    #
    # We also add two moves to include artificial passing moves at the end of
    # the game, so that the engine can learn to pass.
    cdef int total_moves = count_moves(line, line_length) + 2

    if total_moves <= 20:
        return 0

    cdef int pluck_move = <int>(total_moves * (<double>rand() / RAND_MAX))
    cdef int symmetry = <int>(8 * (<double>rand() / RAND_MAX))

    # scan for the following _patterns_ in the line without actually parsing it
    # in any meaningful way. We do it in this stupid way for performance
    # reasons:
    #
    # - 'RE[...]'
    # - ';B[...]'
    # - ';W[...]'

    cdef int x, y, index = 0
    cdef int color = 0
    cdef int winner = 0
    cdef int move_count = 0
    cdef int i = 5  # max size of a match is 4 bytes, e.g. `;B[aa]`

    while i < line_length:
        if line[i] == 66 and line[i-1] == 91 and line[i-2] == 69 and line[i-3] == 82:  # RE[B
            winner = BLACK
        elif line[i] == 87 and line[i-1] == 91 and line[i-2] == 69 and line[i-3] == 82:  # RE[W
            winner = WHITE
        elif line[i] == 93 and line[i-1] == 91 and line[i-3] == 59:  # `;?[]`
            if line[i-2] == 66 or line[i-2] == 87:  # `;B[]` or `;W[]`
                color = BLACK if line[i-2] == 66 else WHITE
                index = 361

                if move_count == pluck_move:
                    break

                move_count += 1
                color = 0
        elif line[i] == 93 and line[i-3] == 91 and line[i-5] == 59:  # `;?[??]`
            if line[i-4] == 66 or line[i-4] == 87:  # `;B[??]` or `;W[??]`
                color = BLACK if line[i-4] == 66 else WHITE
                x = line[i-2] - 97
                y = line[i-1] - 97
                augment(symmetry, &x, &y)

                index = 19 * y + x

                if x >= 19 and y >= 19:  # alternative notation for `pass`
                    index = 361
                    if move_count == pluck_move:
                        break
                elif board._is_valid(color, index):  # don't bother with ko
                    if move_count == pluck_move:
                        break

                    board._place(color, index)
                else:
                    return 0

                move_count += 1
                color = 0

        i += 1

    # return the tuple (winner, index) packed as a single word, this works
    # because `winner` is in the range [0,3), and `color` is in the range [0,3)
    # and `index` is in the range [0,361).
    return winner | (color << 2) | (index << 4)

def one(line):
    """ Returns **one** sample `features, value, policy` from the given SGF
    file """

    board = Board()

    # release the GIL (Global Interpreter Lock) while parsing the SGF file since
    # this can get fairly expensive
    cdef unsigned char *line_ptr = <unsigned char*>line
    cdef int line_length = len(line)
    cdef np.ndarray features = np.zeros((5, 361), 'f4')
    cdef float[:,:] features_view = features
    cdef int winner_color_index, winner, color, index

    with nogil:
        winner_color_index = _one(board, line_ptr, line_length)
        winner = (winner_color_index >> 0) & 0x03
        color = (winner_color_index >> 2) & 0x03
        index = winner_color_index >> 4

        if winner != 0 and color != 0:
            get_features(board, color, features_view)

    if winner == 0 or color == 0:
        raise ValueError
    else:
        value = np.asarray([1.0 if winner == color else -1.0], 'f4')
        policy = np.zeros((362,), 'f4')
        policy[index] = 1.0

        return (features, value, policy)
