
from itertools import product


class SudokuGrid:
    # a lot of this is slightly rewritten code from:
    # https://www.geeksforgeeks.org/sudoku-backtracking-7/
    #--------------------------------------------------------------------------

    def __init__(self, n=9):
        self.n = n  # in case want to solve n != 9 puzzle

        self.flat = []

    #--------------------------------------------------------------------------

    def to_grid(self):
        assert len(self.flat) == self.n*self.n
        self.grid = [self.flat[i:i + self.n]
                     for i in range(0, len(self.flat), self.n)]
        return self.grid

    #--------------------------------------------------------------------------

    def add_value(self, val):
        assert len(self.flat) <= self.n*self.n
        self.flat.append(val)

    #--------------------------------------------------------------------------

    def display_grid(self):
        for row in self.grid:
            print(row)

    #--------------------------------------------------------------------------

    def locate_zero(self, location):
        for row, col in product(range(self.n), repeat=2):
            if self.grid[row][col] == 0:
                location[0], location[1] = row, col
                return True
        return False

    #--------------------------------------------------------------------------

    def in_row(self, row, num):
        for x in range(self.n):
            if self.grid[row][x] == num:
                return True
        return False

    #--------------------------------------------------------------------------

    def in_col(self, col, num):
        for y in range(self.n):
            if self.grid[y][col] == num:
                return True
        return False

    #--------------------------------------------------------------------------

    def in_subgrid(self, row, col, num):
        for x, y in product(range(int(self.n/3)), repeat=2):
            if self.grid[x+row][y+col] == num:
                return True
        return False

    #--------------------------------------------------------------------------

    def is_legal_location(self, row, col, num):
        return not self.in_row(row, num) and not \
            self.in_col(col, num) and not \
            self.in_subgrid(row-row % 3, col-col % 3, num)

    #--------------------------------------------------------------------------

    def solve_grid(self):
        location = [0, 0]

        if not self.locate_zero(location):
            return True

        row, col = location[0], location[1]

        for num in range(1, self.n+1):
            if self.is_legal_location(row, col, num):
                self.grid[row][col] = num
                if self.solve_grid():
                    return True

                self.grid[row][col] = 0

        return False

#------------------------------------------------------------------------------


# This structure & algorithm tends to get stuck in an infinite
# loop in some situations. Plan on fixing it eventually out of
# principle, but the above class provides a fully functional algorithm
#------------------------------------------------------------------------------


class OldSudokuGrid:
    # data structure to store values from OCR
    #--------------------------------------------------------------------------

    def __init__(self, n=9):
        self.n = n  # in case want to solve n != 9 puzzle
        self.flat = []

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

        # shouldn't be checking entire grid for distinct subgrids,
        # only subgrid working in
        for h, v in product(range(int(self.n/3)), repeat=2):
            if self.is_solved_subgrid(grid, h, v, check_zeros) == False:
                print('Subgrid not distinct')
                return False

        print('is solved')
        return True

    #--------------------------------------------------------------------------

    def display_grid(self, grid):
        for row in grid:
            print(row)

    #--------------------------------------------------------------------------

    def solve(self, board, empty):
        # recursive method for solving board. Not currently working
        self.display_grid(board)
        print('empty:%s' % empty)
        if empty == 0:
            return self.is_solved(board, check_zeros=True)
        for row, col in product(range(self.n), repeat=2):
            val = board[row][col]
            if val != 0:
                continue
            grid_copy = board.copy()
            for x in range(1, self.n+1):
                grid_copy[row][col] = x
                if self.is_solved(grid_copy) and \
                        self.solve(grid_copy, empty-1):
                    # something seems to be going on during the recursion
                    return True
                grid_copy[row][col] = x

        return False

    #--------------------------------------------------------------------------
