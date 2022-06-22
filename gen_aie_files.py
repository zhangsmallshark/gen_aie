import os
import shutil
import math


def gen_settings_h(file_name, tile_size_a, tile_size_b, nnz0, n_rows_PE, n_rows_s_PE, n_cols_PE, iterations, tile_indices_sparse):
    left_brace = '{'
    right_brace = '}'

    f = open(file_name, 'w')
    f.write('/*\n')
    f.write('*/\n')
    f.write('#pragma once\n')
    f.write('\n')
    f.write('// Matrix size\n')
    f.write(f'#define R_A {tile_size_a}\n')
    f.write(f'#define C_A {tile_size_a}\n')
    f.write('#define R_B (C_A)\n')
    f.write(f'#define C_B {tile_size_b}\n')
    f.write('#define R_C (R_A)\n')
    f.write('#define C_C (C_B)\n')
    f.write('\n')
    f.write('// Window size\n')
    f.write('#define N_SAMPLES_WINDOW_A (R_A*C_A)\n')
    f.write('#define N_SAMPLES_WINDOW_B (R_B*C_B)\n')
    f.write('#define N_SAMPLES_WINDOW_C (R_C*C_C)\n')
    f.write('\n')
    f.write('#define N_BYTES_UINT8 1\n')
    f.write('#define N_BYTES_FLOAT 4\n')
    f.write('\n')
    f.write('// Systolic size\n')
    f.write(f'#define N_ROWS_PE {n_rows_PE}\n')
    f.write(f'#define N_ROWS_S_PE {n_rows_s_PE}\n')
    f.write(f'#define N_COLS_PE {n_cols_PE}\n')
    f.write('\n')
    f.write('// Sparse size\n')
    nnz0_str = [str(i) for i in nnz0]
    f.write(f'constexpr int kNNZ0[N_ROWS_PE] = {left_brace}{", ".join(nnz0_str)}{right_brace};\n')
    sparse0_str = [str(i) for i in tile_indices_sparse]
    f.write(f'constexpr int kSparse0[N_ROWS_PE] = {left_brace}{", ".join(sparse0_str)}{right_brace};\n')
    f.write('\n')
    f.write(f'#define N_ITERATIONS {iterations}\n')
    f.write('\n')


def gen_aie_kernels_h(src0, src1, n_rows_PE, tile_indices_sparse):
    shutil.copy(src0, src1)
    f = open(src1, 'a')
    for i in range(n_rows_PE):
        if tile_indices_sparse[i] == 1:
            f.write('\n')
            f.write(f'void SparseTensorPE{i}(\n')
            f.write('  input_window_uint8* __restrict left0,\n')
            f.write('  input_window_float* __restrict left1,\n')
            f.write('  input_window_float* __restrict top,\n')
            f.write('  output_window_uint8* __restrict right0,\n')
            f.write('  output_window_float* __restrict right1,\n')
            f.write('  output_window_float* __restrict down,\n')
            f.write('  output_window_float* __restrict p_mat_c);\n')
    f.close()


def gen_systolic_graph0_func(lines, n_rows_PE, n_cols_PE, tile_indices_sparse):
    program = []
    left_brace = '{'
    right_brace = '}'
    space = ' '
    i = 0
    for r in range(n_rows_PE):
        if tile_indices_sparse[r] == 1:
            if i == 0:
                program.append(f'{space*10}if (i == {r}) {left_brace}\n')
                program.append(f'{space*12}PEs[idx0] = kernel::create(SparseTensorPE{r});\n')
                program.append(f'{space*12}source(PEs[idx0]) = "aie_kernels/sparse_tensor_PE{r}.cpp";\n')
                program.append(f'{space * 12}if (j == 0) {left_brace}\n')
                program.append(f'{space * 14}connect< window<kNNZ0[{r}]*N_BYTES_UINT8> > (row_fifos_i[idx1], PEs[idx0].in[0]);\n')
                program.append(f'{space * 14}idx1 = idx1 + 1;\n')
                program.append(f'{space * 14}connect< window<kNNZ0[{r}]*N_BYTES_FLOAT> > (row_fifos_d[i], PEs[idx0].in[1]);\n')
                program.append(f'{space * 12}{right_brace}\n')
                program.append(f'{space * 12}else {left_brace}\n')
                program.append(f'{space * 14}connect< window<kNNZ0[{r}]*N_BYTES_UINT8> > (PEs[idx0-1].out[0], PEs[idx0].in[0]);\n')
                program.append(f'{space * 14}connect< window<kNNZ0[{r}]*N_BYTES_FLOAT> > (PEs[idx0-1].out[1], PEs[idx0].in[1]);\n')
                program.append(f'{space * 12}{right_brace}\n')
                program.append(f'{space*10}{right_brace}\n')
            else:
                program.append(f'{space*10}else if (i == {r}) {left_brace}\n')
                program.append(f'{space*12}PEs[idx0] = kernel::create(SparseTensorPE{r});\n')
                program.append(f'{space*12}source(PEs[idx0]) = "aie_kernels/sparse_tensor_PE{r}.cpp";\n')
                program.append(f'{space * 12}if (j == 0) {left_brace}\n')
                program.append(f'{space * 14}connect< window<kNNZ0[{r}]*N_BYTES_UINT8> > (row_fifos_i[idx1], PEs[idx0].in[0]);\n')
                program.append(f'{space * 14}idx1 = idx1 + 1;\n')
                program.append(f'{space * 14}connect< window<kNNZ0[{r}]*N_BYTES_FLOAT> > (row_fifos_d[i], PEs[idx0].in[1]);\n')
                program.append(f'{space * 12}{right_brace}\n')
                program.append(f'{space * 12}else {left_brace}\n')
                program.append(f'{space * 14}connect< window<kNNZ0[{r}]*N_BYTES_UINT8> > (PEs[idx0-1].out[0], PEs[idx0].in[0]);\n')
                program.append(f'{space * 14}connect< window<kNNZ0[{r}]*N_BYTES_FLOAT> > (PEs[idx0-1].out[1], PEs[idx0].in[1]);\n')
                program.append(f'{space * 12}{right_brace}\n')
                program.append(f'{space*10}{right_brace}\n')

            i += 1

    lines1 = []
    connections = []
    idx1 = 0
    for i in range(n_rows_PE):
        for j in range(n_cols_PE):
            idx0 = i * n_cols_PE + j

            if i == n_rows_PE - 1:
                connections.append(f'connect< window<N_SAMPLES_WINDOW_B*N_BYTES_FLOAT> > (PEs[{idx0}].out[2], null_outs[{idx1}]);\n')
                idx1 += 1

            if j == n_cols_PE - 1:
                if tile_indices_sparse[i] == 1:
                    connections.append(f'connect< window<kNNZ0[{i}]*N_BYTES_UINT8> > (PEs[{idx0}].out[0], null_outs[{idx1}]);\n')
                    idx1 += 1
                    connections.append(f'connect< window<kNNZ0[{i}]*N_BYTES_FLOAT> > (PEs[{idx0}].out[1], null_outs[{idx1}]);\n')
                    idx1 += 1
                else:
                    connections.append(f'connect< window<N_SAMPLES_WINDOW_A*N_BYTES_FLOAT> > (PEs[{idx0}].out[1], null_outs[{idx1}]);\n')
                    idx1 += 1

    e_p0 = 0
    for i, l in enumerate(lines):
        if l.startswith('// begin0'):
            lines1.extend(lines[0:i+1])
            lines1.extend(program)

        if l.startswith('// end0'):
            e_p0 = i

        if l.startswith('// begin1'):
            lines1.extend(lines[e_p0:i+1])
            lines1.extend(connections)

        if l.startswith('// end1'):
            lines1.extend(lines[i:])
    return lines1


def gen_top_graph_func(lines, n_rows_PE, n_cols_PE, tile_indices_sparse):
    lines1 = []
    connections0 = []
    connections1 = []
    idx1 = 0
    for i in range(n_rows_PE):
        if tile_indices_sparse[i] == 1:
            connections0.append(f'connect < window<kNNZ0[{i}]*N_BYTES_UINT8> > (ins0[{idx1}], g0.row_fifos_i[{idx1}]);\n')
            idx1 += 1

    for i in range(n_rows_PE):
        if tile_indices_sparse[i] == 1:
            connections0.append(f'connect < window<kNNZ0[{i}]*N_BYTES_FLOAT> > (ins1[{i}], g0.row_fifos_d[{i}]);\n')
        else:
            connections0.append(f'connect < window<N_SAMPLES_WINDOW_A * N_BYTES_FLOAT > > (ins1[{i}], g0.row_fifos_d[{i}]);\n')

    idx1 = 0
    for i in range(n_rows_PE):
        for j in range(n_cols_PE):

            if i == n_rows_PE - 1:
                connections1.append(f'connect < window<N_SAMPLES_WINDOW_B*N_BYTES_FLOAT> > (g0.null_outs[{idx1}], outs1[{idx1}]);\n')
                idx1 += 1

            if j == n_cols_PE - 1:
                if tile_indices_sparse[i] == 1:
                    connections1.append(f'connect < window<kNNZ0[{i}]*N_BYTES_UINT8> > (g0.null_outs[{idx1}], outs1[{idx1}]);\n')
                    idx1 += 1
                    connections1.append(f'connect < window<kNNZ0[{i}]*N_BYTES_FLOAT> > (g0.null_outs[{idx1}], outs1[{idx1}]);\n')
                    idx1 += 1
                else:
                    connections1.append(f'connect < window<N_SAMPLES_WINDOW_A*N_BYTES_FLOAT> > (g0.null_outs[{idx1}], outs1[{idx1}]);\n')
                    idx1 += 1

    e_p0 = 0
    for i, l in enumerate(lines):
        if l.startswith('// begin2'):
            lines1.extend(lines[0:i+1])
            lines1.extend(connections0)

        if l.startswith('// end2'):
            e_p0 = i

        if l.startswith('// begin3'):
            lines1.extend(lines[e_p0:i+1])
            lines1.extend(connections1)

        if l.startswith('// end3'):
            lines1.extend(lines[i:])
    return lines1


def gen_graph_h(src0, src1, n_rows_PE, n_cols_PE, tile_indices_sparse):
    f0 = open(src0, 'r')
    lines0 = f0.readlines()

    lines1 = gen_systolic_graph0_func(lines0, n_rows_PE, n_cols_PE, tile_indices_sparse)
    lines1 = gen_top_graph_func(lines1, n_rows_PE, n_cols_PE, tile_indices_sparse)

    f1 = open(src1, 'w')
    for l in lines1:
        f1.write(l)


def gen_graph_cpp(file_name, n_rows_PE, n_rows_s_PE, n_cols_PE, tile_indices_sparse):
    f = open(file_name, 'w')
    f.write('/*\n')
    f.write('*/\n')
    f.write('#include "graph.h"\n')

    f.write('\n')
    idx0 = 0
    for k in range(n_rows_PE):
        if tile_indices_sparse[k] == 1:
            f.write(f'PLIO *in{idx0} = new PLIO("indices{k}", plio_64_bits, "./data/indices{k}.txt");\n')
            idx0 += 1
    e_p = n_rows_s_PE

    for k in range(n_rows_PE):
        f.write(f'PLIO *in{e_p+k} = new PLIO("row_fifo{k}", plio_64_bits, "./data/row_fifo{k}.txt");\n')
    e_p += n_rows_PE

    for k in range(n_cols_PE):
        f.write(f'PLIO *in{e_p+k} = new PLIO("col_fifo{k}", plio_64_bits, "./data/col_fifo{k}.txt");\n')

    f.write('\n')
    for k in range(n_rows_PE*n_cols_PE):
        f.write(f'PLIO * out{k} = new PLIO("p_out{k}", plio_64_bits, "./data/p_out{k}.txt");\n')
    e_p = n_rows_PE*n_cols_PE

    f.write('\n')
    for k in range(n_rows_PE+n_rows_s_PE+n_cols_PE):
        f.write(f'PLIO * out{e_p+k} = new PLIO("null_out{k}", plio_64_bits, "./data/null_out{k}.txt");\n')

    f.write('\n')
    num_ins = n_rows_PE+n_rows_s_PE+n_cols_PE
    num_outs = n_rows_PE*n_cols_PE+n_rows_PE+n_rows_s_PE+n_cols_PE
    ins = [f'in{i}' for i in range(num_ins)]
    outs = [f'out{j}' for j in range(num_outs)]
    ins.extend(outs)
    ins_str = ', '.join(ins)
    f.write(f'simulation::platform < {num_ins}, {num_outs} > p({ins_str});\n')

    f.write('\n')
    f.write('TopGraph g;\n')

    f.write('\n')
    idx_in_port = 0
    s_p = 10
    for k in range(n_rows_s_PE):
        f.write(f'connect<> net{s_p+k}(p.src[{k}], g.ins{idx_in_port}[{k}]);\n')
    e_p = s_p + n_rows_s_PE
    if n_rows_s_PE > 0:
        idx_in_port += 1

    for k in range(n_rows_PE):
        f.write(f'connect<> net{e_p+k}(p.src[{n_rows_s_PE+k}], g.ins{idx_in_port}[{k}]);\n')
    e_p += n_rows_PE
    idx_in_port += 1

    for k in range(n_cols_PE):
        f.write(f'connect<> net{e_p+k}(p.src[{n_rows_s_PE+n_rows_PE+k}], g.ins{idx_in_port}[{k}]);\n')
    e_p = e_p + n_cols_PE - 1

    f.write('\n')
    s_p = math.ceil(e_p / 10) * 10
    for k in range(n_rows_PE*n_cols_PE):
        f.write(f'connect<> net{s_p+k}(g.outs0[{k}], p.sink[{k}]);\n')
    e_p = s_p + n_rows_PE * n_cols_PE - 1
    if e_p == s_p:
        e_p += 1

    f.write('\n')
    s_p = math.ceil(e_p / 10) * 10
    for k in range(n_rows_PE+n_rows_s_PE+n_cols_PE):
        f.write(f'connect<> net{s_p+k}(g.outs1[{k}], p.sink[{n_rows_PE*n_cols_PE+k}]);\n')

    f.write('\n')
    f.write('#ifdef __AIESIM__\n')
    f.write('   int main(int argc, char ** argv)\n')
    f.write('   {\n')
    f.write('      g.init();\n')
    f.write('      g.run(N_ITERATIONS);\n')
    f.write('      g.end();\n')
    f.write('\n')
    f.write('      return 0;\n')
    f.write('   }\n')
    f.write('#endif\n')


def unroll_loop_sparse0(par, tile_size_b):
    program = []
    left_brace = '{'
    right_brace = '}'
    space = ' '

    par_len = par[2]
    C_B = tile_size_b
    # col_tiles_B = int(C_B / 8)
    col_tiles_B = 1

    # program.append(f'{space * 2}acc{1} = null_v8float();\n')
    program.append(f'{space * 2}acc{1} = null_v32float();\n')
    for i in range(0, par_len * col_tiles_B):
        program.append(f'{space * 2}window_writeincr(p_mat_c, acc{1});\n')

    program.append(f'{space * 0}\n')

    return program


def unroll_loop_sparse1(par, tile_size_a, tile_size_b):
    program = []
    left_brace = '{'
    right_brace = '}'
    space = ' '

    R_A = tile_size_a
    C_A = tile_size_a
    R_B = C_A
    C_B = tile_size_b
    R_C = R_A
    C_C = C_B
    col_tiles_B = int(C_B / 8)

    loop0 = par[2]
    loop1 = par[0]
    if loop0 > 1:
        program.append(f'{space * 2}for (uint8_t i0 = 0; i0 < {loop0}; i0++)\n')
        program.append(f'{space * 2}chess_prepare_for_pipelining\n')
        program.append(f'{space * 2}chess_loop_range({loop0},)\n')
        program.append(f'{space * 2}{left_brace}\n')

    program.append(f'{space * 4}num_reads_a = 0;\n')
    program.append(f'{space * 4}for (int i1 = 0; i1 < {col_tiles_B}; i1++)\n')
    program.append(f'{space * 4}chess_prepare_for_pipelining\n')
    program.append(f'{space * 4}chess_loop_range({col_tiles_B},)\n')
    program.append(f'{space * 4}{left_brace}\n')
    program.append(f'{space * 6}row_idx0_b = 0;\n')
    program.append(f'{space * 6}acc0 = null_v8float();\n')
    program.append(f'{space * 6}window_decr(left0, num_reads_a);\n')
    program.append(f'{space * 6}window_decr(left1, num_reads_a);\n')
    program.append(f'{space * 6}for (int i2 = 0; i2 < {loop1}; i2++)\n')
    program.append(f'{space * 6}chess_prepare_for_pipelining\n')
    program.append(f'{space * 6}chess_loop_range({loop1},)\n')
    program.append(f'{space * 6}chess_flatten_loop\n')
    program.append(f'{space * 6}{left_brace}\n')
    program.append(f'{space * 8}val = window_readincr(left1);\n')
    program.append(f'{space * 8}buf_mat_a = upd_elem(buf_mat_a, 0, val);\n')
    program.append(f'{space * 8}row_idx1_b = window_readincr(left0);\n')
    program.append(f'{space * 8}row_idx_diff0 = row_idx1_b - row_idx0_b;\n')
    program.append(f'{space * 8}row_idx0_b = row_idx1_b;\n')
    program.append(f'{space * 8}window_incr_v8(top, row_idx_diff0);\n')
    program.append(f'{space * 8}buf_mat_b = window_read_v8(top);\n')
    program.append(f'{space * 8}acc0 = fpmac(acc0, xset_w(0, buf_mat_a), 0, 0x00000000, buf_mat_b, 0, 0x76543210);\n')
    program.append(f'{space * 6}{right_brace}\n')
    program.append(f'{space * 6}window_writeincr(p_mat_c, acc0);\n')
    program.append(f'{space * 6}num_reads_a = {loop1};\n')
    program.append(f'{space * 6}row_idx_diff1 = {R_B} - row_idx0_b;\n')
    program.append(f'{space * 6}window_incr_v8(top, row_idx_diff1);\n')
    program.append(f'{space * 4}{right_brace}\n')
    program.append(f'{space * 4}window_decr_v8(top, {R_B * col_tiles_B});\n')

    if loop0 > 1:
        program.append(f'{space * 2}{right_brace}\n')

    program.append(f'{space * 0}\n')

    return program


def unroll_loop_dense(par, tile_size_a, tile_size_b):
    program = []
    left_brace = '{'
    right_brace = '}'
    space = ' '

    R_B = tile_size_a
    col_tiles_A = int(tile_size_a / 8)
    col_tiles_B = int(tile_size_b / 8)

    s = 2 
    max_nnz = [0]
    par_len = par[2]
    for i in range(s, s + par_len):
        program.append(f'{space * 2}v8float buf_mat_a{i} = undef_v8float();\n')
        program.append(f'{space * 2}v8float acc{i} = null_v8float();\n')

    # program.append(f'{space * 2}window_incr_v8(left1, num_incr_a);\n')
    program.append(f'{space * 2}num_reads_a = 0;\n')
    program.append(f'{space * 2}for (int i1 = 0; i1 < {col_tiles_B}; i1++)\n')
    program.append(f'{space * 2}chess_prepare_for_pipelining\n')
    program.append(f'{space * 2}chess_loop_range({col_tiles_B},)\n')
    program.append(f'{space * 2}{left_brace}\n')
    for i in range(s, s + par_len):
        program.append(f'{space * 4}acc{i} = null_v8float();\n')

    program.append(f'{space * 4}window_decr_v8(left1, num_reads_a);\n')
    program.append(f'{space * 4}for (int i2 = 0; i2 < {col_tiles_A}; i2++)\n')
    program.append(f'{space * 4}chess_prepare_for_pipelining\n')
    program.append(f'{space * 4}chess_loop_range({col_tiles_A},)\n')
    program.append(f'{space * 4}{left_brace}\n')
    for i in range(s, s + par_len):
        program.append(f'{space * 6}buf_mat_a{i} = window_readincr_v8(left1);\n')

    program.append(f'{space * 6}for (int i3 = 0; i3 < 8; i3++)\n')
    program.append(f'{space * 6}chess_prepare_for_pipelining\n')
    program.append(f'{space * 6}chess_loop_range(8,)\n')
    program.append(f'{space * 6}chess_flatten_loop\n')
    program.append(f'{space * 6}{left_brace}\n')
    program.append(f'{space * 8}buf_mat_b = window_readincr_v8(top);\n')
    for i in range(s, s + par_len):
        program.append(f'{space * 8}acc{i} = fpmac(acc{i}, xset_w(0, buf_mat_a{i}), i3, 0x00000000, buf_mat_b, 0, 0x76543210);\n')

    program.append(f'{space * 6}{right_brace}\n')
    program.append(f'{space * 4}{right_brace}\n')
    for i in range(s, s + par_len):
        program.append(f'{space * 4}window_writeincr(p_mat_c, acc{i});\n')

    program.append(f'{space * 4}num_reads_a = {col_tiles_A * par_len};\n')
    program.append(f'{space * 2}{right_brace}\n')
    program.append(f'{space * 2}window_decr_v8(top,{R_B * col_tiles_B});\n\n')

    return program


def gen_sparse_PE(dst_dir, tile_size_a, tile_size_b, par_dics, nnzs_tiles, tile_indices_sparse):
    left_brace = '{'
    right_brace = '}'
    space = ' '

    R_A = tile_size_a
    C_A = tile_size_a
    R_B = C_A
    C_B = tile_size_b
    R_C = R_A
    C_C = C_B
    n_reads_top = int((R_B * C_B) / 32)

    base_name = 'sparse_tensor_PE'
    tiles = len(tile_indices_sparse)
    for i in range(tiles):
        if tile_indices_sparse[i] == 1:
            par_dic = par_dics[i]
            nnz_left = nnzs_tiles[i]
            n_reads_left = int(nnz_left / 16)
            lines0 = []
            lines0.append(f'{space * 0}/*\n')
            lines0.append(f'{space * 0}left0: indices\n')
            lines0.append(f'{space * 0}left1: data\n')
            lines0.append(f'{space * 0}top: mat_b\n')
            lines0.append(f'{space * 0}*/\n')
            lines0.append(f'{space * 0}\n')
            lines0.append(f'{space * 0}#include <adf.h>\n')
            lines0.append(f'{space * 0}#include <cstdint>\n')
            lines0.append(f'{space * 0}#include "system_settings.h"\n')
            lines0.append(f'{space * 0}\n')
            lines0.append(f'{space * 0}void SparseTensorPE{i}(\n')
            lines0.append(f'{space * 2}input_window_uint8* __restrict left0,\n')
            lines0.append(f'{space * 2}input_window_float* __restrict left1,\n')
            lines0.append(f'{space * 2}input_window_float* __restrict top,\n')
            lines0.append(f'{space * 2}output_window_uint8* __restrict right0,\n')
            lines0.append(f'{space * 2}output_window_float* __restrict right1,\n')
            lines0.append(f'{space * 2}output_window_float* __restrict down,\n')
            lines0.append(f'{space * 2}output_window_float* __restrict p_mat_c)\n')
            lines0.append(f'{space * 0}{left_brace}\n')
            lines0.append(f'{space * 2}v16uint8 buf_right0 = undef_v16uint8();\n')
            lines0.append(f'{space * 2}v16float buf_right1 = undef_v16float();\n')
            lines0.append(f'{space * 2}v32float buf_down = undef_v32float();\n')
            lines0.append(f'{space * 2}for (uint8_t i0 = 0; i0 < {n_reads_left}; i0++)\n')
            lines0.append(f'{space * 2}chess_prepare_for_pipelining\n')
            lines0.append(f'{space * 2}chess_loop_range({n_reads_left},)\n')
            lines0.append(f'{space * 2}{left_brace}\n')
            lines0.append(f'{space * 4}buf_right0 = window_readincr_v16(left0);\n')
            lines0.append(f'{space * 4}buf_right1 = window_readincr_v16(left1);\n')
            lines0.append(f'{space * 4}window_writeincr(right0, buf_right0);\n')
            lines0.append(f'{space * 4}window_writeincr(right1, buf_right1);\n')
            lines0.append(f'{space * 2}{right_brace}\n')
            lines0.append(f'{space * 2}window_decr_v16(left0, {n_reads_left});\n')
            lines0.append(f'{space * 2}window_decr_v16(left1, {n_reads_left});\n')
            lines0.append(f'{space * 2}for (uint8_t i0 = 0; i0 < {n_reads_top}; i0++)\n')
            lines0.append(f'{space * 2}chess_prepare_for_pipelining\n')
            lines0.append(f'{space * 2}chess_loop_range({n_reads_top},)\n')
            lines0.append(f'{space * 2}{left_brace}\n')
            lines0.append(f'{space * 4}buf_down = window_readincr_v32(top);\n')
            lines0.append(f'{space * 4}window_writeincr(down, buf_down);\n')    
            lines0.append(f'{space * 2}{right_brace}\n')
            lines0.append(f'{space * 2}window_decr_v32(top, {n_reads_top});\n')
            lines0.append(f'{space * 2}v8float buf_mat_a = null_v8float();\n')
            lines0.append(f'{space * 2}v8float buf_mat_b = undef_v8float();\n')
            lines0.append(f'{space * 2}v8float acc0 = null_v8float();\n')
            lines0.append(f'{space * 2}v32float acc1 = null_v32float();\n')
            lines0.append(f'{space * 2}float val = 0.0;\n')
            lines0.append(f'{space * 2}int row_idx0_b = 0;\n')
            lines0.append(f'{space * 2}int row_idx1_b = 0;\n')
            lines0.append(f'{space * 2}int row_idx_diff0 = 0;\n')
            lines0.append(f'{space * 2}int row_idx_diff1 = 0;\n')
            lines0.append(f'{space * 2}int num_reads_a = 0;\n')

            lines0.append(f'{space * 0}\n')

            for par in par_dic.values():
                if par[0] == 0:
                    program = unroll_loop_sparse0(par, tile_size_b)
                elif 0 < par[0] < int(C_A * 1.5):
                    program = unroll_loop_sparse1(par, tile_size_a, tile_size_b)
                else:
                    program = unroll_loop_dense(par, tile_size_a, tile_size_b)

                lines0.extend(program)
            lines0.append(f'{space * 0}{right_brace}\n')
            dst_file = f'{dst_dir}/{base_name}{i}.cpp'
            f1 = open(dst_file, 'w')
            for l in lines0:
                f1.write(l)


if __name__ == "__main__":
    print('finish')
