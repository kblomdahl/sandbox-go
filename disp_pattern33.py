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

from gen_pattern33 import get_all_patterns
import sys

PATTERNS = list(sorted(get_all_patterns()))

def c(p):
    if p == 0:
        return ' '
    elif p == 1:
        return 'X'
    elif p == 2:
        return 'O'
    else:
        return '-'

print('  X  Player   O  Opponent')
print('')

for pattern in sys.argv[1:]:
    pattern = PATTERNS[int(pattern)]
    parts = [
        (pattern >> 16) & 0x3,
        (pattern >> 14) & 0x3,
        (pattern >> 12) & 0x3,
        (pattern >> 10) & 0x3,
        (pattern >>  8) & 0x3,
        (pattern >>  6) & 0x3,
        (pattern >>  4) & 0x3,
        (pattern >>  2) & 0x3,
        (pattern >>  0) & 0x3
    ]

    print('%s:' % pattern)
    print('%s %s %s' % (c(parts[7]), c(parts[0]), c(parts[1])))
    print('%s %s %s' % (c(parts[6]), c(parts[8]), c(parts[2])))
    print('%s %s %s' % (c(parts[5]), c(parts[4]), c(parts[3])))
    print('')
