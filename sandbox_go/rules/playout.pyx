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

from .board cimport Board, neighbours
from .color cimport opposite

import cython

@cython.boundscheck(False)
cdef int xorshift(int[1] state) nogil:
    cdef int x = state[0]

    x ^= x << 13
    x ^= x >> 17
    x ^= x << 5

    state[0] = x  # update state
    return x

cdef int[17] EYES = {
    70996, 86356, 87316, 87376, 87380, 87384, 87444, 87548,
    88404, 90068, 90108, 103764, 130388, 131028, 251228,
    251388, 261468
}

@cython.boundscheck(False)
cdef int is_eye(Board board, int next_color, int index) nogil:
    cdef int lo = 0
    cdef int hi = 16
    cdef int mid
    cdef int value = board._get_pattern(next_color, index)

    while lo <= hi:
        mid = (lo + hi) // 2

        if EYES[mid] < value:
            lo = mid + 1
        elif value < EYES[mid]:
            hi = mid - 1
        else:
            return 1

    return 0

@cython.boundscheck(False)
cdef void playout(Board board, int next_color, int[1] state) nogil:
    cdef int remaining, index, other, temp
    cdef int[361] policy

    for index in range(361):
        policy[index] = index

    # play until we hit 722 moves, or both players run out of valid moves
    cdef int pass_count = 0
    cdef int move_count = 0

    while move_count < 722 and pass_count < 2:
        # pick a move at random, only checking if it is valid when it gets
        # picked, in which case we try again without that move among the
        # candidates.
        remaining = 361

        while remaining > 0:
            other = xorshift(state) % remaining
            remaining -= 1

            index = policy[other]

            if board._is_valid(next_color, index) and not board._is_ko(next_color, index) not is_eye(board, next_color, index):
                break

            # move the candidate that is invalid to the end of the remaining
            # candidates. We will then decrease the list length, so this move
            # gets excluded.
            temp = policy[remaining]
            policy[remaining] = policy[other]
            policy[other] = temp

        if remaining == 0:  # pass
            pass_count += 1
        else:
            board._place(next_color, index)
            pass_count = 0

        move_count += 1
        next_color = opposite(next_color)

    # remove any stones that are in atari from the board (this only makes sense
    # if we hit 722 moves).
    if pass_count < 2:
        for index in range(361):
            if board.vertices[index] != 0 and not board._has_two_liberty(index):
                board._capture(index)


@cython.boundscheck(False)
cdef int is_reachable(Board board, int color, int index) nogil:
    cdef char[361] visited
    cdef int[361] queue
    cdef int remaining = 1
    cdef int ns[4], n

    # zero out the visited array
    for n in range(361):
        visited[n] = 0

    # depth-first-search over all connected vertices
    queue[0] = index
    visited[index] = 1

    while remaining > 0:
        remaining -= 1
        index = queue[remaining]

        neighbours(index, ns)
        for n in ns:
            if board.vertices[n] == color:
                return 1
            elif board.vertices[n] == 0 and visited[n] == 0:
                visited[n] = 1
                queue[remaining] = n
                remaining += 1

    return 0


@cython.boundscheck(False)
cdef void score_votes(Board board, int next_color, int[361] votes):
    cdef int other_color = opposite(next_color)

    # initial random state, this is fixed so that the training is deterministic
    cdef int[1] state
    state[0] = 0xdeadbeef

    # playout some random games and then vote on who each vertex belongs to
    # based on those games
    cdef int black_reachable, white_reachable
    cdef Board other
    cdef int i, j

    for j in range(361):
        votes[j] = 0

    for i in range(5):
        other = board.copy()
        playout(other, next_color, state)

        for j in range(361):
            if other.vertices[j] == 0:  # eye
                black_reachable = is_reachable(other, next_color, j)
                white_reachable = is_reachable(other, other_color, j)

                if black_reachable and white_reachable:
                    pass
                elif black_reachable:
                    votes[j] += 1
                else:
                    votes[j] -= 1
            elif other.vertices[j] == next_color:
                votes[j] += 1
            else:
                votes[j] -= 1  # opponent
