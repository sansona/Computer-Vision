from itertools import product
from copy import copy
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
        self.flat = [int(x) for x in test]

    #--------------------------------------------------------------------------

    def to_grid(self):
        self.grid = [self.flat[i:i + self.n]
                     for i in range(0, len(self.flat), self.n)]
        return self.grid

    #--------------------------------------------------------------------------

    def to_corners(self, grid):
        # converts grid into 9x9 smaller corner grids
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

        # divides each of three col into 3 smaller corner grids
        self.corners = []
        cols = [left_cols, center_cols, right_cols]
        for corner in cols:
            self.corners.append([corner[i*self.n: i*self.n+self.n] for i in
                                 range(int(len(corner)/self.n))])
        # use [hori_idx][vert_idx][value_idx] to access value
        return self.corners

    #--------------------------------------------------------------------------

    def is_proper_shape(self):
        # checks that grid is nxn
        for row in range(len(self.grid)):
            if len(self.grid[row]) != self.n:
                return False
        return True

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
                and contains_legal_values
                and self.is_proper_shape())

    #--------------------------------------------------------------------------

    def num_zeros(self):
        # return number empty cells (0s) in self.grid
        num_zeros = 0
        for row_idx in range(self.n):
            if self.grid[row_idx].count(0) > 0:
                num_zeros += self.grid[row_idx].count(0)

        return num_zeros

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

    def is_distinct_list(self, inp_list):
        '''
        used for checking distinct values in each row. For checking row, just call this w/ row as input. For columns & corners, use below functions since need more formatting
        '''
        assert len(inp_list) == self.n
        used_val = []
        for val in inp_list:
            if val == 0:
                continue
            if val in used_val:
                return False
            used_val.append(val)
        return True

    #--------------------------------------------------------------------------

    def is_solved_column(self, col):
        # should work. May need a bit more testing since untested on true grid
        col_val = [self.grid[row][col] for row in range(self.n)]
        return self.is_distinct_list(col_val)

    #--------------------------------------------------------------------------

    def is_solved_corner(self, grid, hori_idx, vert_idx):
        # should work. May need a bit more testing since untested on true grid
        corner_grid = self.to_corners(grid)
        corner = corner_grid[hori_idx][vert_idx]
        return self.is_distinct_list(corner)

    #--------------------------------------------------------------------------

    def is_solved(self, grid):
        # final check that puzzle is solved
        solved_cols_rows = True
        for i in range(self.n):
            if (self.is_distinct_list(grid[i]) != True or
                    self.is_solved_column(i) != True):
                return False

        solved_corners = True
        for h, v in product(range(int(self.n/3)), repeat=2):
            if self.is_solved_corner(grid, h, v) != True:
                return False

        return solved_cols_rows and solved_corners

   #---------------------------------------------------------------------------

    def display_grid(self, grid):
        for row in grid:
            print(row)

   #---------------------------------------------------------------------------

    def solve(self, board, empty):
        # recursive method for solving board. Not currently working
        self.display_grid(board)
        print('empty:%s' % empty)
        if empty == 0:
            self.display_grid(board)
            return self.is_solved(board)
        for row, col in product(range(self.n), repeat=2):
            val = board[row][col]
            if val != 0:
                continue
            grid_copy = copy(board)
            for x in range(1, self.n+1):
                self.replace_value(grid_copy, row, col, x)
                if self.is_solved(grid_copy) and self.solve(
                        grid_copy, empty - 1):
                    return True
                # something in this line is causing the infinite loop. If try 9 before solve, will just replace w/ 0 and leave as is
                self.replace_value(grid_copy, row, col, 0)
        return False


#------------------------------------------------------------------------------
