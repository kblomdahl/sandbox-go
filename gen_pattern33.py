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

from sandbox_go.rules.board import Board

def _wildcard_aux(board, coordinates, callback):
    if coordinates:
        x, y = coordinates[0]

        def _place(board, color):
            if color:
                other = board.copy()

                if other.is_valid(color, x, y):
                    other.place(color, x, y)

                return other
            else:
                return board

        _wildcard_aux(_place(board, None), coordinates[1:], callback)
        _wildcard_aux(_place(board, 1), coordinates[1:], callback)
        _wildcard_aux(_place(board, 2), coordinates[1:], callback)
    else:
        callback(board)

def _wildcard(cx, cy, callback):
    def _is_valid(x, y):
        """ Returns true if the given coordinate is a valid vertex on the
        board. """

        return x in range(19) and y in range(19)

    # find all coordinates that might be part of this coordinate
    vertices = [
        (cx + 0, cy + 1),  # N
        (cx + 1, cy + 1),  # NE
        (cx + 1, cy + 0),  # E
        (cx + 1, cy - 1),  # SE
        (cx + 0, cy - 1),  # S
        (cx - 1, cy - 1),  # SW
        (cx - 1, cy + 0),  # W
        (cx - 1, cy + 1),  # NW
        (cx + 0, cy + 0),  # C
    ]

    vertices = [(x, y) for (x, y) in vertices if _is_valid(x, y)]

    # recurse into all possible combinations
    _wildcard_aux(Board(), vertices, callback)


def get_all_patterns():
    visited = set()

    for cx in range(19):
        for cy in range(19):
            def _add_to_visited(board):
                pattern = board.get_pattern(1, cx, cy)
                visited.add(pattern)

            _wildcard(cx, cy, _add_to_visited)

    return visited

#                                          NNNEEESESSSWWWNWCC
assert Board().get_pattern(1,  0,  0) == 0b000000111111111100
assert Board().get_pattern(1, 18,  0) == 0b001111111111000000
assert Board().get_pattern(1,  0, 18) == 0b111100000011111100
assert Board().get_pattern(1, 18, 18) == 0b111111110000001100

assert Board().get_pattern(1,  0,  9) == 0b000000000011111100
assert Board().get_pattern(1,  9,  0) == 0b000000111111000000
assert Board().get_pattern(1, 18,  9) == 0b001111110000000000
assert Board().get_pattern(1,  9, 18) == 0b111100000000001100

assert Board().get_pattern(1,  9,  9) == 0b000000000000000000

if __name__ == '__main__':
    # 
    PATTERNS = get_all_patterns()

    print(len(PATTERNS))
    assert len(PATTERNS) == 22665

    for i, pattern in enumerate(sorted(PATTERNS)):
        if i > 0 and i % 9 == 0:
            print('')
        if i % 9 == 0:
            print('   ', end='')

        print('{: 7},'.format(pattern), end='')
    print()
