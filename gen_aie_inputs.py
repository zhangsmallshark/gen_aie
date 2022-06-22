import math
import numpy as np
from sparse_mat import write_mat_file1, write_mat_file_ind_d1, write_mat_file_ind_d0, write_mat_b_file


def reshape_a(mat, tile_size):
    r_a, c_a = mat.shape
    n_tiles = math.ceil(c_a / tile_size)
    arr0 = np.zeros((tile_size*n_tiles, tile_size))

    for i in range(0, tile_size*n_tiles, tile_size):
        if i + tile_size > c_a:
            e_p = c_a
            length = c_a - i
        else:
            e_p = i + tile_size
            length = tile_size

        arr0[i:i+r_a, 0:length] = mat[:, i:e_p]

    return arr0


def gen_row_fifos(mat, par_dics, tile_size, tile_indices_sparse, dst_dir):
    r_a, c_a = mat.shape
    row_tiles = int(r_a / tile_size)
    col_tiles = int(c_a / tile_size)
    for i in range(row_tiles):
        fifo_name = f'{dst_dir}/row_fifo{i}.txt'
        s_r = tile_size * i
        e_r = tile_size * (i + 1)

        for j in range(col_tiles):
            s_c = tile_size * j
            e_c = tile_size * (j + 1)
            p_mat = mat[s_r:e_r, s_c:e_c]

            if tile_indices_sparse[i] == 1:
                indices_name = f'{dst_dir}/indices{i}.txt'
                write_mat_file_ind_d0(p_mat, 'w', f'{dst_dir}/indices_v.txt', f'{dst_dir}/row_fifo_v.txt')
                if j == 0:
                    write_mat_file_ind_d1(p_mat, par_dics[i], 'w', indices_name, fifo_name)
                else:
                    write_mat_file_ind_d1(p_mat, par_dics[i], 'a', indices_name, fifo_name)
            else:
                if j == 0:
                    write_mat_file1(fifo_name, 'w', p_mat, 1)
                else:
                    write_mat_file1(fifo_name, 'a', p_mat, 1)


def gen_col_fifos(mat, tile_size_a, tile_size_b, dst_dir):
    r_b, c_b = mat.shape
    row_tiles = int(r_b / tile_size_a)
    col_tiles = int(c_b / tile_size_b)
    for j in range(col_tiles):
        fifo_name = f'{dst_dir}/col_fifo{j}.txt'
        s_c = tile_size_b * j
        e_c = tile_size_b * (j + 1)

        for i in range(row_tiles):
            s_r = tile_size_a * i
            e_r = tile_size_a * (i + 1)

            p_mat = mat[s_r:e_r, s_c:e_c]
            if i == 0:
                write_mat_b_file(fifo_name, 'w', p_mat, 1)
            else:
                write_mat_b_file(fifo_name, 'a', p_mat, 1)


def read_pad_adj(mat, tile_size):
    r_a, c_a = mat.shape
    if r_a % tile_size == 0 and c_a % tile_size == 0:
        return mat

    row_tiles = math.ceil(r_a / tile_size)
    col_tiles = math.ceil(c_a / tile_size)
    arr0 = np.zeros((row_tiles*tile_size, col_tiles*tile_size))
    arr0[0:r_a, 0:c_a] = mat[:, :]
    return arr0
