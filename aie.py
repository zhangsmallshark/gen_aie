import math
import os
import numpy as np
from sparse_mat import inter_partition, gen_random_mat_a, gen_random_mat_a2, add_val, gen_random_mat_b, gen_mat_c, gen_p_mat_c, compare_mat
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


def estimate_layer_time(r_a, c_a, c_x, r_w, c_w, tile_size_a, tile_size_x, tile_size_w, dense_engine_time, sparse_engine_time):
    num_rows_engines = 8
    num_cols_engines = 50
    r_x = c_a
    num_rows_dense_engine = 4
    num_cols_dense_engine = 50

    computed_rows_b = tile_size_x * num_cols_dense_engine
    computed_cols_b = tile_size_w * num_rows_dense_engine
    iterations_d = math.ceil(c_x / tile_size_x) * int(c_w / computed_cols_b)
    total_iterations_d = math.ceil(r_x / computed_rows_b) * iterations_d

    num_rows_sparse_engine = 4
    num_cols_sparse_engine = 50

    computed_rows_c = tile_size_a * num_cols_sparse_engine
    computed_cols_c = tile_size_b * num_rows_sparse_engine
    iterations_s = math.ceil(c_a / tile_size_a) * int(c_b / tile_size_b)
    total_iterations_d = math.ceil(r_a / computed_rows_c) * iterations_s


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
    print(f'current density {total_nnz / (rows * cols)}')


if __name__ == "__main__":
    r_a = 32
    c_a = 32
    r_b = c_a
    c_b = 32
    r_c = r_a
    c_c = c_b
    tile_size_a = 32
    tile_size_b = 32
    density0 = 0.1

    clean_dirs()

    # mat_a = gen_random_mat_a(r_a, c_a, density0)

    # nnz_rows_a = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]
    nnz_rows_a = [1] * 32
    nnz0, mat_a = gen_random_mat_a2(nnz_rows_a, c_a)
    # add_val(mat_a)

    # mat_a = gen_random_mat_b(r_a, c_a)
    # np.save('./mat/mat_a.npy', mat_a)
    # mat_a = np.load('./mat/mat_a.npy')

    # p_adj = np.load('./mat/Reddit_part20.npy')
    # rows, cols = p_adj.shape
    # s_r = 10
    # mat_a = p_adj[s_r:s_r+r_a, s_r:s_r+c_a]

    calc_density(mat_a, './refer_mat/nnz_a.txt')

    par_dics, row_indices_s, nnz0 = inter_partition(mat_a, tile_size_a)
    # par_dics = {0: {0: [8, 0, 8], 1: [8, 8, 8], 2: [8, 16, 8], 3: [8, 24, 8]}}

    # par_dics = {0: {0: [4, 0, 4], 1: [4, 4, 4], 2: [4, 8, 4], 3: [4, 12, 4], 4: [4, 16, 4], 5: [4, 20, 4], 6: [4, 24, 4], 7: [4, 28, 4]}}

    row_indices_s = [1] * int(r_a / tile_size_a)

    n_rows_PE = int(r_a / tile_size_a)
    n_rows_s_PE = sum(row_indices_s)
    n_cols_PE = int(c_b / tile_size_b)

    settings_h = '/home/fpga/Desktop/systolic_array/systolic/src/system_settings.h'
    gen_settings_h(settings_h, tile_size_a, tile_size_b, nnz0, n_rows_PE, n_rows_s_PE, n_cols_PE, int(c_a/tile_size_a), row_indices_s)

    o_aie_kernels_h = './aie_src/aie_kernels.h'
    n_aie_kernels_h = '/home/fpga/Desktop/systolic_array/systolic/src/aie_kernels.h'
    gen_aie_kernels_h(o_aie_kernels_h, n_aie_kernels_h, n_rows_PE, row_indices_s)

    if n_rows_s_PE > 0:
        o_graph_h = './aie_src/graph_sparse.h'
    else:
        o_graph_h = './aie_src/graph_dense.h'

    n_graph_h = '/home/fpga/Desktop/systolic_array/systolic/src/graph.h'
    gen_graph_h(o_graph_h, n_graph_h, n_rows_PE, n_cols_PE, row_indices_s)

    graph_cpp = '/home/fpga/Desktop/systolic_array/systolic/src/graph.cpp'
    gen_graph_cpp(graph_cpp, n_rows_PE, n_rows_s_PE, n_cols_PE, row_indices_s)

    if n_rows_s_PE > 0:
        o_sparse_tensor_PE_cpp = './aie_src/sparse_tensor_PE.cpp'
        dst_dir = '/home/fpga/Desktop/systolic_array/systolic/src/aie_kernels'
        gen_sparse_PE(o_sparse_tensor_PE_cpp, dst_dir, par_dics, nnz0, row_indices_s)

    mat_b = gen_random_mat_b(r_b, c_b)
    # np.save('./mat/mat_b.npy', mat_b)
    # mat_b = np.load('./mat/mat_b.npy')

    mat_c = gen_mat_c(mat_a, mat_b)
    write_mat_file1('./refer_mat/c.txt', 'w', mat_c, 1)

    gen_p_mat_c(mat_a, mat_b, tile_size_a, tile_size_b, 0, 0, './refer_mat/p_c_0_0.txt')

    # dst_dir = './mat'
    dst_dir = '/home/fpga/Desktop/systolic_array/systolic/data'
    gen_row_fifos(mat_a, par_dics, tile_size_a, row_indices_s, dst_dir)
    gen_col_fifos(mat_b, tile_size_a, tile_size_b, dst_dir)
    # read_pad_adj(mat_a, tile_size_a)

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
