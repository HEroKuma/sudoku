class Solver:
    def __init__(self, puzzle):
        self.puzzle = puzzle
        self.box_size = 3

    def get_row(self, row_number):
        return self.puzzle[row_number]

    def get_column(self, column_number):
        return [row[column_number] for row in self.puzzle]

    def find_possibilities(self, row_number, column_number):
        possibilities = set(range(1, 10))
        row = self.get_row(row_number)
        column = self.get_column(column_number)
        box = self.get_box(row_number, column_number)
        for item in row + column + box:
            if not isinstance(item, list) and item in possibilities:
                possibilities.remove(item)
        return possibilities
