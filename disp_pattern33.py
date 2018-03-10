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

def to_ch(p):
    if p == 0:
        return ' '
    elif p == 1:
        return 'X'
    elif p == 2:
        return 'O'
    else:
        return '-'

def pad(line, to_length, left):
    while len(line) < to_length:
        if left:
            line = ' ' + line
        else:
            line += ' '
    return line

print('  X  Player   O  Opponent')
print('')

patterns = []

for index in sys.argv[1:]:
    code = PATTERNS[int(index)]
    parts = [
        (code >> 16) & 0x3,
        (code >> 14) & 0x3,
        (code >> 12) & 0x3,
        (code >> 10) & 0x3,
        (code >>  8) & 0x3,
        (code >>  6) & 0x3,
        (code >>  4) & 0x3,
        (code >>  2) & 0x3,
        (code >>  0) & 0x3
    ]

    pattern  = '%s\n' % index
    pattern += '-----\n'
    pattern += '%s %s %s\n' % (to_ch(parts[7]), to_ch(parts[0]), to_ch(parts[1]))
    pattern += '%s %s %s\n' % (to_ch(parts[6]), to_ch(parts[8]), to_ch(parts[2]))
    pattern += '%s %s %s\n' % (to_ch(parts[5]), to_ch(parts[4]), to_ch(parts[3]))
    patterns += [pattern]

# 
for i in range(0, len(patterns), 8):
    slices = patterns[i:i+8]

    for i in range(5):
        print('| ', end='')

        for pattern in slices:
            parts = pattern.split('\n')

            print(pad(parts[i], 5, i == 0), end=' | ')
        print()
    print()
