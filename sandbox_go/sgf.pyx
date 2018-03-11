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
from .rules.color cimport BLACK, WHITE, opposite
from .rules.features cimport get_features
from .rules.features import NUM_FEATURES

import cython
from libc.stdlib cimport malloc, free, rand, RAND_MAX
cimport numpy as np
import numpy as np


@cython.boundscheck(False)
cdef void _augment(int xx, int xy, int *x, int yx, int yy, int *y) nogil:
    cdef int x_ = x[0]
    cdef int y_ = y[0]
    cdef int cx = x_ - 9
    cdef int cy = y_ - 9

    x[0] = 9 + (xx * cx + xy * cy)
    y[0] = 9 + (yx * cx + yy * cy)


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
cdef int gather_moves(
    const unsigned char *line,
    const int line_length,
    int *color,
    int *x,
    int *y,
    int *winner
) nogil:
    cdef int i = 5
    cdef int resigned = 0
    cdef int count = 0

    winner[0] = 0

    # collect all played moves in the given SGF into pre-allocated arrays by
    # a stupid scanning algorithm that looks for the following patterns:
    #
    # - 'RE[...]'
    # - `;B[??]`
    # - `;W[??]`
    # - `;B[]`
    # - `;W[]`
    # - `Resign`
    #
    cdef int symmetry = <int>(8 * (<double>rand() / (<double>RAND_MAX + 1)))

    while i < line_length:
        if line[i] == 66 and line[i-1] == 91 and line[i-2] == 69 and line[i-3] == 82:  # RE[B
            winner[0] = BLACK
        elif line[i] == 87 and line[i-1] == 91 and line[i-2] == 69 and line[i-3] == 82:  # RE[W
            winner[0] = WHITE
        elif line[i] == 93 and line[i-3] == 91 and line[i-5] == 59:  # `;?[??]`
            if line[i-4] == 66 or line[i-4] == 87:  # `;B[??]` or `;W[??]`
                x[count] = line[i-2] - 97
                y[count] = line[i-1] - 97
                augment(symmetry, &x[count], &y[count])

                color[count] = BLACK if line[i-4] == 66 else WHITE
                count += 1
        elif line[i] == 93 and line[i-1] == 91 and line[i-3] == 59:  # `;?[]`
            if line[i-2] == 66 or line[i-2] == 87:  # `;B[]` or `;W[]`
                x[count] = 19
                y[count] = 19
                color[count] = BLACK if line[i-2] == 66 else WHITE
                count += 1
        elif line[i] == 100 and line[i-1] == 103 and line[i-2] == 105 and line[i-3] == 115 and line[i-4] == 101 and line[i-5] == 82:  # `Resign`
            resigned = 1

        i += 1

    # if this game was played to finish, then add potentially missing passing
    # moves at the end so that the engine can learn when it is appropriate to
    # pass
    if resigned == 0 and count > 2 and x[count-1] != 19 and y[count-1] != 19:
        x[count+0] = 19
        x[count+1] = 19
        y[count+0] = 19
        y[count+1] = 19

        color[count+0] = opposite(color[count-1])
        color[count+1] = color[count-1]
        count += 2

    return count

@cython.boundscheck(False)
cdef int _one(
    const unsigned char *line,
    const int line_length,
    Board board,
    int *winner,
    int *next1_color,
    int *next1_index,
    int *next2_index
) nogil:
    # gather the moves played without doing anything, this is done so that we
    # do not need to playout more of the game than we need to, and be able to
    # perform look-a-head during learning.
    #
    # It also helps since it allows us to discard games that are too short, and
    # the winner of said game is therefore uncertain.
    #
    cdef int *color = <int*>malloc(sizeof(int) * 1024)
    cdef int *x = <int*>malloc(sizeof(int) * 1024)
    cdef int *y = <int*>malloc(sizeof(int) * 1024)
    cdef int total_moves, pluck_move, index, i

    try:
        total_moves = gather_moves(line, line_length, color, x, y, winner)

        if total_moves <= 20:
            return 0

        # playout the game until the move that we are going to pluck, this is
        # necessary because we have to extract the features.
        pluck_move = <int>(total_moves * (<double>rand() / (<double>RAND_MAX + 1)))

        for i in range(pluck_move - 1):
            if x[i] < 19 and y[i] < 19:  # not pass
                index = 19 * y[i] + x[i]

                if board._is_valid(color[i], index):
                    board._place(color[i], index)

        # return the tuple `(winner, color, index1, index2)`, where `index1` is the
        # next move and `index2` is the move after (or pass if EOF)
        next1_color[0] = color[pluck_move]

        if x[pluck_move] < 19:
            next1_index[0] = 19 * y[pluck_move] + x[pluck_move]
        else:
            next1_index[0] = 361

        if pluck_move + 1 < total_moves and x[pluck_move+1] < 19:
            next2_index[0] = 19 * y[pluck_move+1] + x[pluck_move+1]
        else:
            next2_index[0] = 361

        return 1
    finally:
        free(color)
        free(x)
        free(y)

def one(line):
    """ Returns **one** sample `features, value, policy` from the given SGF
    file """

    cdef Board board = Board()

    # release the GIL (Global Interpreter Lock) while parsing the SGF file since
    # this can get fairly expensive
    cdef unsigned char *line_ptr = <unsigned char*>line
    cdef int line_length = len(line)
    cdef np.ndarray features = np.zeros((49, 9), 'f4')
    cdef np.ndarray policy = np.zeros((49, 10), 'f4')
    cdef float[:,:] features_view = features
    cdef float[:,:] policy_view = policy
    cdef int winner = 0, next1_color = 0, next1_index, next2_index

    with nogil:
        if _one(line_ptr, line_length, board, &winner, &next1_color, &next1_index, &next2_index):
            get_features(board, next1_color, next1_index, features_view, policy_view)

    # allocate the appropriate NumPy arrays to contain the policies and features
    if winner == 0 or next1_color == 0:
        raise ValueError
    else:
        return (features, policy)
