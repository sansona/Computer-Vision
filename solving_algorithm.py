from itertools import product
from copy import copy

import time

#------------------------------------------------------------------------------


class SudokuGrid:
    # data structure to store values from OCR
    #--------------------------------------------------------------------------

    def __init__(self):
        self.n = 9  # in case want to solve n != 9 puzzle

        # self.flat = []
        '''
        self.flat = [3, 0, 6, 5, 0, 8, 4, 0, 0, 5, 2,
                     0, 0, 0, 0, 0, 0, 0, 0,
                     8, 7, 0, 0, 0, 0, 3, 1,
                     0, 0, 3, 0, 1, 0, 0, 8,
                     0, 9, 0, 0, 8, 6, 3, 0,
                     0, 5, 0, 5, 0, 0, 9, 0,
                     6, 0, 0, 1, 3, 0, 0, 0,
                     0, 2, 5, 0, 0, 0, 0, 0,
                     0, 0, 0, 7, 4, 0, 0, 5,
                     2, 0, 6, 3, 0, 0]
        '''
        test = '400000805030000000000700000020000060000080400' + \
            '000010000000603070500200000104000000'
        assert len(test) == self.n*self.n
        self.flat = [int(x) for x in test]

    #--------------------------------------------------------------------------

    def to_grid(self):
        self.grid = [self.flat[i:i + self.n]
                     for i in range(0, len(self.flat), self.n)]
        return self.grid

    #--------------------------------------------------------------------------

    def to_subgrids(self, grid):
        # converts grid into 9x9 smaller subgrid grids
        assert len(grid) == self.n

        left_cols = []
        center_cols = []
        right_cols = []
        # divides grid into 3 large columns
        for row, idx in product(range(self.n), repeat=2):
            if idx < self.n/3:
                left_cols.append(grid[row][idx])
            elif idx >= self.n/3 and idx < 2*self.n/3:
                center_cols.append(grid[row][idx])
            elif idx >= 2*self.n/3 and idx < self.n:
                right_cols.append(grid[row][idx])

        # divides each of three col into 3 smaller subgrid grids
        self.subgrids = []
        cols = [left_cols, center_cols, right_cols]
        for subgrid in cols:
            self.subgrids.append([subgrid[i*self.n: i*self.n+self.n] for i in
                                  range(int(len(subgrid)/self.n))])
        # use [hori_idx][vert_idx][value_idx] to access value
        return self.subgrids

    #--------------------------------------------------------------------------

    def is_valid(self, grid):
        # checks that grid adheres to sudoku starting position 0
        is_right_size = (len(grid) == self.n*self.n)
        contain_only_ints = all(type(grid[i]) == int
                                for i in range(1, len(self.flat), 2))
        contains_legal_values = (all(val <= self.n for val in self.flat))
        # check for legal sudoku grid
        return (is_right_size
                and contain_only_ints
                and contains_legal_values)

    #--------------------------------------------------------------------------

    def num_zeros(self):
        # return number empty cells (0s) in self.grid
        num_check_zeros = 0
        for row_idx in range(self.n):
            if self.grid[row_idx].count(0) > 0:
                num_check_zeros += self.grid[row_idx].count(0)

        return num_check_zeros

    #--------------------------------------------------------------------------

    def add_value(self, value):
        # adds to self.flat, not self.grid
        assert type(value) == int
        if len(self.flat) != self.n*self.n:
            self.flat.append(value)

    #--------------------------------------------------------------------------

    def replace_value(self, grid, row, col, value):
        # replaces value in self.grid given [x][y] position
        args = [row, col, value]

        assert (all(type(item) == int for item in args))
        assert (all(item <= self.n for item in args))

        grid[row][col] = value

    #--------------------------------------------------------------------------

    def is_distinct_list(self, inp_list, check_zeros=False):
        '''
        used for checking distinct values in each row. For checking row, just call this w/ row as input. For columns & subgrids, use below functions since need more formatting
        '''
        assert len(inp_list) == self.n
        used_val = []
        for val in inp_list:
            if check_zeros == True:
                if val == 0:
                    return False
            if val == 0:
                continue
            if val in used_val:
                return False
            used_val.append(val)
        return True

    #--------------------------------------------------------------------------

    def is_solved_column(self, grid, col, check_zeros=False):
        # should work. May need a bit more testing since untested on true grid
        col_val = [grid[row][col] for row in range(self.n)]
        return self.is_distinct_list(col_val, check_zeros)

    #--------------------------------------------------------------------------

    def is_solved_subgrid(self, grid, hori_idx, vert_idx, check_zeros=False):
        # should work. May need a bit more testing since untested on true grid
        subgrid_grid = self.to_subgrids(grid)
        subgrid = subgrid_grid[hori_idx][vert_idx]
        return self.is_distinct_list(subgrid, check_zeros)

    #--------------------------------------------------------------------------

    def is_solved(self, grid, check_zeros=False):
        # final check that puzzle is solved
        for i in range(self.n):
            if self.is_distinct_list(grid[i], check_zeros) == False:
                print('List not distinct')
                return False
            elif self.is_solved_column(grid, i, check_zeros) == False:
                print('Column not distinct')
                return False
        '''
        #shouldn't be checking entire grid for distinct subgrids, only subgrid working in
        for h, v in product(range(int(self.n/3)), repeat=2):
            if self.is_solved_subgrid(grid, h, v, check_zeros) == False:
                print('Subgrid not distinct')
                return False
        '''
        print('is solved')
        return True

   #---------------------------------------------------------------------------

    def display_grid(self, grid):
        for row in grid:
            print(row)

   #---------------------------------------------------------------------------

    def solve(self, board, empty):
        # recursive method for solving board. Not currently working
        self.display_grid(board)
        print('empty:%s' % empty)
        time.sleep(0.5)
        if empty == 0:
            return self.is_solved(board, check_zeros=True)
        for row, col in product(range(self.n), repeat=2):
            val = board[row][col]
            if val != 0:
                continue
            grid_copy = copy(board)
            while 0 in board[row]:
                for x in range(1, self.n+1):
                    self.replace_value(grid_copy, row, col, x)
                    if self.is_solved(grid_copy) and \
                            self.solve(grid_copy, empty-1):
                        # something seems to be going on during the recursion
                        print('recursion')
                        return True
                        print('replacing value')
                    self.replace_value(grid_copy, row, col, 0)

        return False


#------------------------------------------------------------------------------
