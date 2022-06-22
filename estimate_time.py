import math

# A * X * W 


def X_W(r_x, c_x, c_w, tile_size_x, tile_size_w):
    # MHz
    freq = 1000
    cycles = 9000

    num_rows_dense_engine = 4
    num_cols_dense_engine = 50

    computed_rows_b = tile_size_x * num_cols_dense_engine
    computed_cols_b = tile_size_w * num_rows_dense_engine

    iterations = math.ceil(r_x / computed_rows_b) * math.ceil(c_x / tile_size_x) * math.ceil(c_w / computed_cols_b)

    time = iterations * (cycles / freq)
    return time


# dense_rectangle
def SpMM_AIE(r_a, c_a, c_b, tile_size_a, tile_size_b):
    # MHz
    freq = 1000
    cycles = 2000

    num_rows_sparse_engine = 4
    num_cols_sparse_engine = 50

    computed_rows_c = tile_size_a * num_cols_sparse_engine
    computed_cols_c = tile_size_b * num_rows_sparse_engine

    iterations = math.ceil(r_a / computed_rows_c) * math.ceil(c_a / tile_size_a) * math.ceil(c_b / computed_cols_c)

    time = iterations * (cycles / freq)
    return time


def calc_tile_nnz(s_v, len0, col_idx, tile_range_before, nnzs_before, nnz_rect, tile_range_after, nnzs_after):
    offset = s_v + len0
    if col_idx < s_v:
        if len(nnzs_before) == 0:
            return

        if tile_range_before[0] <= col_idx < tile_range_before[1]:
            nnzs_before[-1] += 1
        else:
            tile_idx = math.floor(col_idx / len0)
            nnzs_before[tile_idx] += 1

    elif s_v <= col_idx < offset:
        nnz_rect[0] += 1

    else:
        if len(nnzs_after) == 0:
            return

        if tile_range_after[0] <= col_idx < tile_range_after[1]:
            nnzs_after[-1] += 1
        else:
            tile_idx = math.floor((col_idx - offset + 1) / len0)
            nnzs_after[tile_idx] += 1


def SpMM_PL(r_a, c_a, c_b, nnzs_before, nnzs_after):
    num_PEs = 256
    # MHz
    freq = 273
    cycles = 1 + 3 + 1

    total_nnzs = 0
    if len(nnzs_before) > 0:
        for nnz in nnzs_before:
            total_nnzs += nnz

    if len(nnzs_after) > 0:
        for nnz in nnzs_after:
            total_nnzs += nnz

    # us
    time = math.ceil(total_nnzs / num_PEs) * (cycles / freq)
    return time


def A_B(r_a, c_a, c_b, re_root_l, re_graph, tile_size_a, tile_size_b):
    time_A_B = 0
    len_r = len(re_root_l)
    for p in range(len_r):
        s_v = re_root_l[p][0]
        len0 = re_root_l[p][1]
        e_v0 = s_v + len0

        # before rectangle
        # q, rem = divmod(s_v, len0)
        # if rem / len0 > 0.5:
        #     tiles_before = q + 1
        #     tile_range_before = [len0 * q, s_v]
        # else:
        #     tiles_before = q
        #     tile_range_before = [len0 * (q - 1), s_v]

        # # after rectangle
        # offset = s_v + len0
        # q, rem = divmod(c_a - offset, len0)
        # if rem / len0 > 0.5:
        #     tiles_after = q + 1
        #     tile_range_after = [offset + len0 * q, c_a]
        # else:
        #     tiles_after = q
        #     tile_range_after = [offset + len0 * (q - 1), c_a]

        # before rectangle
        if s_v / len0 > 0.1:
            tiles_before = 1
        else:
            tiles_before = 0
        tile_range_before = [0, s_v]

        # after rectangle
        offset = s_v + len0
        if (c_a - offset) / len0 > 0.5:
            tiles_after = 1
        else:
            tiles_after = 0
        tile_range_after = [offset, c_a]

        nnzs_before = [0] * tiles_before
        nnz_rect = [0]
        nnzs_after = [0] * tiles_after

        for v in range(s_v, e_v0):
            neighbours = re_graph[v]
            for neighbour in neighbours:
                calc_tile_nnz(s_v, len0, neighbour, tile_range_before, nnzs_before, nnz_rect, tile_range_after, nnzs_after)

        time_SpMM_AIE = SpMM_AIE(len0, len0, c_b, tile_size_a, tile_size_b)
        time_SpMM_PL = SpMM_PL(r_a, c_a, c_b, nnzs_before, nnzs_after)

        print(f'len0 {len0}')
        print(f'time SpMM_AIE {time_SpMM_AIE} us; time SpMM_PL {time_SpMM_PL} us')

    time_A_B += abs(time_SpMM_AIE - time_SpMM_PL)

    return time_A_B


def estimate_layer_time(r_a, c_a, c_x, c_w, re_root_l, re_graph, tile_size_a, tile_size_b, tile_size_x, tile_size_w):
    r_x = r_a
    r_w = c_x
    c_b = c_w

    time_X_W = X_W(r_x, c_x, c_w, tile_size_x, tile_size_w)
    time_A_B = A_B(r_a, c_a, c_b, re_root_l, re_graph, tile_size_a, tile_size_b)
    print(f'time X_W {time_X_W} us; time A_B {time_A_B} us')
    layer_time = time_X_W + time_A_B
    return layer_time
