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

cdef class Board:
    # -------- Instance variables --------

    cdef char vertices[368]
    cdef int next_vertex[368]
    cdef unsigned long zobrist_hash

    # Set of the most recent board positions, used to detect super-ko. This can
    # miss some very long ko's but those should not occur in real games anyway.
    cdef unsigned long zobrist_hashes[8]
    cdef int zobrist_hashes_index

    # -------- Methods --------

    cdef int _has_one_liberty(self, int index) nogil
    cdef int _has_two_liberty(self, int index) nogil

    cdef unsigned long _capture_ko(self, int index) nogil

    cdef int _is_valid(self, int color, int index) nogil
    cdef int _is_ko(self, int color, int index) nogil
    cpdef int is_valid(self, int color, int x, int y)

    cdef void _capture(self, int index) nogil
    cdef void _connect_with(self, int index, int other) nogil
    cdef void _place(self, int color, int index) nogil
    cpdef void place(self, int color, int x, int y)

    cdef int _get_pattern_code(self, int color, int index) nogil
    cdef int _get_pattern(self, int color, int index) nogil

    cdef int get_num_liberties(self, int index) nogil
