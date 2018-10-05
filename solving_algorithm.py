#------------------------------------------------------------------------------
class SudokuGrid:
    # data structure to store values from OCR

    def __init__(self, n=9):
        self.grid = []
        # may just want to make this a flat and roll it up when need to access
        # cols & rows. Think just accepting a list as an arg may work too.
        for row in range(1, n + 1):
            for col in range(1, n + 1):
                self.grid.append([(row, col), 0])
        self.flat = [value for row in self.grid for value in row]
        self.permitted_values = range(1, n+1)

    def is_right_size(self, n):
        return len(self.flat) == 2*n*n

    def contain_only_ints(self):
        return all(type(self.flat[i]) == int
                   for i in range(1, len(self.flat), 2))

    def is_valid(self, n):
        return (self.is_right_size(n) and self.contain_only_ints())

    def is_legal_row(self, row):
        # may have to restructure grid object
        print('WIP')

    def is_legal_column(self, column):
        print('WIP')

    def fill_starting values(self, starting_val):
        print('WIP')

    def fill_value(self, position, value):
        self.grid[position][1] = value

    def solve(self):
        # TODO: Implement algorithm to solve grid
        print('WIP')

#------------------------------------------------------------------------------
