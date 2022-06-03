from sparse_mat import gen_random_mat_a1, gen_random_mat_b, gen_mat_c, compare_mat
import numpy as np
import math


class TensorPE:
    def __init__(self, tile_size_a, tile_size_b):
        self.tile_size_a = tile_size_a
        self.tile_size_b = tile_size_b
        self.stored_val = np.zeros((self.tile_size_a, self.tile_size_b))
        self.p_mat_c = []
        self.down = [0]*(self.tile_size_a * self.tile_size_b)
        self.right = [0]*(self.tile_size_a ** 2)

    def mat_mult(self, p_a, p_b):
        acc = [[0] * self.tile_size_b for _ in range(self.tile_size_a)]
        for i in range(self.tile_size_a):
            for j in range(self.tile_size_a):
                for k in range(self.tile_size_b):
                    acc[i][k] += p_a[i * self.tile_size_a + j] * p_b[j * self.tile_size_b + k]
        return np.array(acc)

    def update_val(self, left, top):
        p_mat = self.mat_mult(left, top)
        self.stored_val = self.stored_val + p_mat
        self.p_mat_c.append(p_mat)

    def update_output(self, left, top):
        self.down = top
        self.right = left


class SystolicTensor:
    def __init__(self, num_rows, num_cols, tile_size_a, tile_size_b):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.tile_size_a = tile_size_a
        self.tile_size_b = tile_size_b
        self.cycles = 0
        self.row_fifos0 = {}
        self.row_fifos1 = {}
        self.col_fifos0 = {}
        self.col_fifos1 = {}
        for i in range(self.num_rows):
            self.row_fifos0[i] = []
            self.row_fifos1[i] = []

        for i in range(self.num_cols):
            self.col_fifos0[i] = []
            self.col_fifos1[i] = []

        self.PEs = {}
        for i in range(self.num_rows):
            PE_row = {}
            for j in range(self.num_cols):
                PE_row[j] = TensorPE(self.tile_size_a, self.tile_size_b)
            self.PEs[i] = PE_row

    def get_PE(self, i, j):
        return self.PEs[i][j]

    def read_col(self, i, pop):
        if pop:
            val = self.col_fifos0[i].pop(0)
        else:
            val = self.col_fifos0[i][0]
        return val

    def read_row(self, i, pop):
        if pop:
            val = self.row_fifos0[i].pop(0)
        else:
            val = self.row_fifos0[i][0]
        return val

    def update(self):
        for i in range(self.num_rows-1, -1, -1):
            for j in range(self.num_cols-1, -1, -1):
                if i == 0:
                    top = self.read_col(j, True)
                else:
                    top = self.get_PE(i - 1, j).down

                if j == 0:
                    left = self.read_row(i, True)
                else:
                    left = self.get_PE(i, j - 1).right

                self.get_PE(i, j).update_val(left, top)
                self.get_PE(i, j).update_output(left, top)

    def init_fifos(self):
        for i in range(1, self.num_rows):
            for j in range(i):
                self.row_fifos0[i].append([0] * (self.tile_size_a ** 2))

        for i in range(1, self.num_cols):
            for j in range(i):
                self.col_fifos0[i].append([0] * (self.tile_size_a * self.tile_size_b))

    def load_operand(self, mat_a, mat_b):
        r_a, c_a = mat_a.shape
        num_tiles = int(c_a / self.tile_size_a)
        for r_i in range(self.num_rows):
            if self.cycles < num_tiles:
                p_mat_a = mat_a[r_i * self.tile_size_a:(r_i + 1) * self.tile_size_a, self.cycles * self.tile_size_a:(self.cycles + 1) * self.tile_size_a]
                p_mat_a = np.reshape(p_mat_a, (1, -1))
                p_mat_a_1d = p_mat_a[0].tolist()
                self.row_fifos0[r_i].append(p_mat_a_1d)
                self.row_fifos1[r_i].append(p_mat_a_1d)
            else:
                self.row_fifos0[r_i].append([0] * (self.tile_size_a ** 2))

        for c_i in range(self.num_cols):
            if self.cycles < num_tiles:
                p_mat_b = mat_b[self.cycles * self.tile_size_a:(self.cycles + 1) * self.tile_size_a, c_i * self.tile_size_b:(c_i + 1) * self.tile_size_b]
                p_mat_b = np.reshape(p_mat_b, (1, -1))
                p_mat_b_1d = p_mat_b[0].tolist()
                self.col_fifos0[c_i].append(p_mat_b_1d)
                self.col_fifos1[c_i].append(p_mat_b_1d)
            else:
                self.col_fifos0[c_i].append([0] * (self.tile_size_a * self.tile_size_b))

    def combine_p_res(self):
        final_mat = np.zeros((self.num_rows * self.tile_size_a, self.num_cols * self.tile_size_b))
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                p_mat = self.get_PE(i, j).stored_val
                final_mat[i * self.tile_size_a:(i + 1) * self.tile_size_a, j * self.tile_size_b:(j + 1) * self.tile_size_b] = p_mat[:, :]
        return final_mat


def gen_mat_a(rows, cols):
    mat_a = np.zeros((rows, cols))
    l0 = range(cols)
    for i in range(rows):
        mat_a[i][:] = l0[:]
    return mat_a


def gen_mat_b(rows, cols):
    mat_b = np.ones((rows, cols))
    return mat_b


def write_mat_file0(file_name, w, mat, iteration):
    rows, cols = mat.shape
    total_num = rows * cols
    rem = total_num % 2
    if rem == 1:
        n_len = total_num + 1
    else:
        n_len = total_num

    arr0 = np.reshape(mat, (1, -1))
    f = open(file_name, w)
    for i in range(iteration):
        for j in range(0, n_len, 2):
            if j + 1 < total_num:
                f.write(f'{arr0[0][j]}  {arr0[0][j + 1]}\n')
            else:
                f.write(f'{arr0[0][j]}  {0.0}\n')
    f.close()


def write_fifos(fifos, num_cycles, dst_dir, base_name):
    for k in fifos.keys():
    #     for i in range(k):
    #         fifos[k].insert(0, [0] * (tile_size0 ** 2))
    #
    #     for j in range(num_cycles - len(fifos[k])):
    #         fifos[k].append([0] * (tile_size0 ** 2))

        p_mat = np.array(fifos[k])
        full_name = f'{dst_dir}/{base_name}{k}.txt'
        write_mat_file0(full_name, 'w', p_mat, 1)


def write_p_mat_c(systolic, dst_dir):
    for i in range(systolic.num_rows):
        for j in range(systolic.num_cols):
            idx0 = i * systolic.num_cols + j
            idx1 = 0
            p_mat_c = systolic.PEs[i][j].p_mat_c
            file_name = f'{dst_dir}/p_out{idx0}.txt'
            for m in p_mat_c:
                if idx1 == 0:
                    write_mat_file0(file_name, 'w', m, 1)
                else:
                    write_mat_file0(file_name, 'a', m, 1)
                idx1 += 1


if __name__ == "__main__":
    r_a = 512
    c_a = 608
    r_b = c_a
    c_b = 128
    r_c = r_a
    c_c = c_b
    tile_size_a = 32
    tile_size_b = 32

    mat_a = gen_random_mat_a1(r_a, c_a, 0.8)
    mat_b = gen_random_mat_b(r_b, c_b)
    # mat_a = gen_mat_a(r_a, c_a)
    # mat_b = gen_mat_b(r_b, c_b)

    mat_c = gen_mat_c(mat_a, mat_b)

    sys_arr = SystolicTensor(16, 4, tile_size_a, tile_size_b)
    sys_arr.init_fifos()
    # num_cycles = sys_arr.num_rows * 2 + sys_arr.num_cols - 2
    num_cycles = sys_arr.num_rows + sys_arr.num_cols - 2 + int(c_a / tile_size_a)
    for x in range(num_cycles):
        sys_arr.load_operand(mat_a, mat_b)
        sys_arr.update()
        sys_arr.cycles += 1

    sys_mat_c = sys_arr.combine_p_res()
    compare_mat(mat_c, sys_mat_c)
    # write_fifos(sys_arr.row_fifos1, num_cycles, './refer_mat', 'row_fifo')
    # write_fifos(sys_arr.col_fifos1, num_cycles, './refer_mat', 'col_fifo')
    # write_p_mat_c(sys_arr, './refer_mat')

    print('systolic tensor finish !!!')
