

mmy_gird = [0,0,1,1,1,1,1,0,1,1,1,1,0,1,0,1,1,0,1,1,0,1,1,0]


def make_grid(n, m, my_list):

    gird = [[0]*m for i in range(n)]

    for i in range(n):
        for j in range(m):
            temp = my_list[i*n + j]
            gird[i][j] = temp

    return gird


def max_store(n, m, grid):

    def can_open(i ,j):
        if grid[i][j] == 1 and grid[i][j+1] == 1 \
                and grid[i+1][j] == 1 and grid[i+1][j+1] == 1:
            grid[i][j] = grid[i][j+1] = \
                grid[i+1][j] = grid[i+1][j+1] = 8
            return True
        else:
            return False

    sum_count = 0
    row = 0
    col = 0

    while row < n-1 and col < m-1:

        if grid[row][col] == 1:
            if can_open(row, col):
                sum_count += 1
                col += 2
            else:
                col += 1
        else:
            col += 1

        if col + 2 > m:
            col = 0
            row += 1

        if row + 2 > n:
            break

    return sum_count


my_grid = make_grid(4, 6, mmy_gird)
res = max_store(4,6,my_grid)

print(res)


