import math
import os
import numpy as np
from sparse_mat import inter_partition, gen_random_mat_a0, gen_random_mat_a2, gen_random_mat_b, gen_mat_c, gen_p_mat_c, compare_mat
from gen_aie_files import gen_settings_h, gen_aie_kernels_h, gen_graph_h, gen_graph_cpp, gen_sparse_PE
from gen_aie_inputs import *
from systolic_tensor import SystolicTensor, write_fifos, write_p_mat_c


def clean_dirs():
    commands = ['rm -rf /home/fpga/Desktop/graph_files/graph/refer_mat/*']
    commands.append('rm -rf /home/fpga/Desktop/systolic_array/systolic/Emulation-AIE/aiesimulator_output/*')
    commands.append('rm -rf /home/fpga/Desktop/systolic_array/systolic/data/*')
    # commands.append('rm -rf /home/fpga/Desktop/systolic_array/systolic/src/aie_kernels/*')
    for c in commands:
        os.system(c)


def calc_density(mat, file_name):
    rows, cols = mat.shape
    f = open(file_name, 'w')
    total_nnz = 0
    for i in range(rows):
        nnz = np.count_nonzero(mat[i, :])
        if i > 0 and i % 8 == 0:
            f.write('\n')
        else:
            f.write(f'{nnz} ')
        total_nnz += nnz
    print(f'ori density {total_nnz / (rows * cols)}')


if __name__ == "__main__":
    r_a = 16
    c_a = 16
    r_b = c_a
    c_b = 16
    r_c = r_a
    c_c = c_b
    tile_size_a = 16
    tile_size_b = 16
    density0 = 0.5

    clean_dirs()

    mat_a = gen_random_mat_a0(r_a, c_a, density0)

    # nnz_rows_a = [0, 0, 0, 0, 5, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]
    # nnz_rows_a = [0, 0, 0, 0, 5, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 30, 31, 30, 30]

    # nnz_rows_a = [4] * 32
    # mat_a = gen_random_mat_a2(nnz_rows_a, c_a)

    # mat_a = gen_random_mat_b(r_a, c_a)
    # np.save('./mat/mat_a.npy', mat_a)
    # mat_a = np.load('./mat/mat_a.npy')

    # p_adj = np.load('./mat/Reddit_part20.npy')
    # rows, cols = p_adj.shape
    # s_r = 10
    # mat_a = p_adj[s_r:s_r+r_a, s_r:s_r+c_a]

    calc_density(mat_a, './refer_mat/nnz_a.txt')

    par_dics, tile_indices_sparse, nnzs_tiles = inter_partition(mat_a, tile_size_a)
    tile_indices_sparse = [1] * int(r_a / tile_size_a)

    n_rows_PE = int(r_a / tile_size_a)
    n_rows_s_PE = sum(tile_indices_sparse)
    n_cols_PE = int(c_b / tile_size_b)

    settings_h = '/home/fpga/Desktop/systolic_array/systolic/src/system_settings.h'
    gen_settings_h(settings_h, tile_size_a, tile_size_b, nnzs_tiles, n_rows_PE, n_rows_s_PE, n_cols_PE, int(c_a/tile_size_a), tile_indices_sparse)

    o_aie_kernels_h = './aie_src/aie_kernels.h'
    n_aie_kernels_h = '/home/fpga/Desktop/systolic_array/systolic/src/aie_kernels.h'
    gen_aie_kernels_h(o_aie_kernels_h, n_aie_kernels_h, n_rows_PE, tile_indices_sparse)

    if n_rows_s_PE > 0:
        o_graph_h = './aie_src/graph_sparse.h'
    else:
        o_graph_h = './aie_src/graph_dense.h'

    n_graph_h = '/home/fpga/Desktop/systolic_array/systolic/src/graph.h'
    gen_graph_h(o_graph_h, n_graph_h, n_rows_PE, n_cols_PE, tile_indices_sparse)

    graph_cpp = '/home/fpga/Desktop/systolic_array/systolic/src/graph.cpp'
    gen_graph_cpp(graph_cpp, n_rows_PE, n_rows_s_PE, n_cols_PE, tile_indices_sparse)

    if n_rows_s_PE > 0:
        dst_dir = '/home/fpga/Desktop/systolic_array/systolic/src/aie_kernels'
        gen_sparse_PE(dst_dir, tile_size_a, tile_size_b, par_dics, nnzs_tiles, tile_indices_sparse)

    mat_b = gen_random_mat_b(r_b, c_b)
    # np.save('./mat/mat_b.npy', mat_b)
    # mat_b = np.load('./mat/mat_b.npy')

    mat_c = gen_mat_c(mat_a, mat_b)
    write_mat_file1('./refer_mat/c.txt', 'w', mat_c, 1)

    gen_p_mat_c(mat_a, mat_b, tile_size_a, tile_size_b, 0, 0, './refer_mat/p_c_0_0.txt')

    dst_dir = '/home/fpga/Desktop/systolic_array/systolic/data'
    gen_row_fifos(mat_a, par_dics, tile_size_a, tile_indices_sparse, dst_dir)
    gen_col_fifos(mat_b, tile_size_a, tile_size_b, dst_dir)
    read_pad_adj(mat_a, tile_size_a)

    sys_arr = SystolicTensor(n_rows_PE, n_cols_PE, tile_size_a, tile_size_b)
    sys_arr.init_fifos()
    num_cycles = sys_arr.num_rows + sys_arr.num_cols - 2 + int(c_a / tile_size_a)
    for x in range(num_cycles):
        sys_arr.load_operand(mat_a, mat_b)
        sys_arr.update()
        sys_arr.cycles += 1

    sys_mat_c = sys_arr.combine_p_res()
    compare_mat(mat_c, sys_mat_c)
    write_fifos(sys_arr.row_fifos1, num_cycles, './refer_mat', 'row_fifo')
    write_fifos(sys_arr.col_fifos1, num_cycles, './refer_mat', 'col_fifo')
    write_p_mat_c(sys_arr, './refer_mat')

    print('aie finish !!!')
