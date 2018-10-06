from itertools import product
#------------------------------------------------------------------------------


class SudokuGrid:
    # data structure to store values from OCR
    #--------------------------------------------------------------------------

    def __init__(self):
        self.n = 9  # in case want to solve n != 9 puzzle

        # self.flat = []
        test = '40000080503000000000070000002000006' + \
            '0000080400000' + '010000000603070500200000104000000'
        self.flat = [int(x) for x in test]

    #--------------------------------------------------------------------------

    def to_grid(self):
        self.grid = [self.flat[i:i + self.n]
                     for i in range(0, len(self.flat), self.n)]
        return self.grid

    #--------------------------------------------------------------------------

    def to_corners(self):
        # converts grid into 9x9 smaller corner grids
        assert len(self.grid) == self.n

        left_cols = []
        center_cols = []
        right_cols = []
        # divides grid into 3 large columns
        for row, idx in product(range(self.n), repeat=2):
            if idx < self.n/3:
                left_cols.append(self.grid[row][idx])
            elif idx >= self.n/3 and idx < 2*self.n/3:
                center_cols.append(self.grid[row][idx])
            elif idx >= 2*self.n/3 and idx < self.n:
                right_cols.append(self.grid[row][idx])

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

    def replace_value(self, row, col, value):
        # replaces value in self.grid given [x][y] position
        args = [row, col, value]

        assert (all(type(item) == int for item in args))
        assert (all(item <= self.n for item in args))

        self.grid[row][col] = value

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

    def is_solved_corner(self, hori_idx, vert_idx):
        # should work. May need a bit more testing since untested on true grid
        corner = self.corners[hori_idx][vert_idx]
        return self.is_distinct_list(corner)

    #--------------------------------------------------------------------------

    def is_solved(grid):
        # final check that puzzle is solved
        solved_cols_rows = False
        for i in range(self.n):
            if (self.is_distinct_list(grid[i]) == True and
                    self.is_solved_column(i) == True):
                continue
            else:
                print('Row/col not solved')
                return False
                break
            solved_cols_rows = True

        solved_corners = False
        for h, v in product(range(int(self.n/3)), repeat=2):
            if self.is_solved_corner(h, v) == True:
                continue
            else:
                print('Corner not solved')
                return False
                break
            solved_corners = True

        return solved_cols_rows and solved_corners

   #---------------------------------------------------------------------------

    def solve(self, board, empty):
        # recursive method for solving board. Not currently working
        if empty == 0:
            return self.is_solved(board)
        for row, col in product(range(self.n), repeat=2):
            val = board[row][col]
            if val != 0:
                continue
            grid_copy = board
            for x in range(self.n):
                if self.is_valid(grid_copy) and solve(grid_copy, empty - 1):
                    return True
                grid_copy[row][col] = 0
        return False


#------------------------------------------------------------------------------
b = SudokuGrid()
grid = b.to_grid()
c = b.to_corners()
num_em = b.num_zeros()
print(b.solve(grid, num_em))
