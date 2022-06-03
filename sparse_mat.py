import numpy as np
import scipy as sci
from scipy.sparse import csr_matrix
import random
import math


def add_val(s_mat):
    rows, cols = s_mat.shape
    for i in range(rows):
        sub_mat = s_mat[i]
        sub_max = np.amax(sub_mat)
        if sub_max == 0:
            col = random.randint(0, cols - 1)
            s_mat[i][col] = 1.0

    density = np.sum(s_mat) / (rows * cols)
    print(f'density in add_val is {density} !!!')


def gen_random_mat_a(rows, cols, density):
    total_num = rows * cols
    nnz = int(density * total_num)
    z = total_num - nnz
    nnz_arr = np.ones(nnz, dtype=np.float32)
    z_arr = np.zeros(z, dtype=np.float32)
    arr0 = np.concatenate((nnz_arr, z_arr))
    np.random.shuffle(arr0)
    mat_a = np.reshape(arr0, (rows, cols))

    # mat_a = np.random.randint(0, 2, size=(rows, cols))
    # eye_mat = np.eys(rows, cols, dtype=np.float32)
    # f_mat_a = np.dot(mat_a, eye_mat)

    # rng = np.random.default_rng()
    # rvs = sci.stats.poisson(25, loc=10).rvs
    # mat_a = sci.sparse.random(16, 8, density=0.1, random_state=rng, data_rvs=rvs)
    return mat_a


def gen_random_mat_a1(rows, cols, density):
    nnz_per_row = int(cols * density)
    if nnz_per_row == 0:
        nnz_per_row = 1

    arr0 = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        # indices = np.random.randint(0, cols, size=nnz_per_row).tolist()
        indices = np.random.choice(cols, size=nnz_per_row, replace=False)
        for j in indices:
            arr0[i][j] = 1.0

    # sum_c = np.sum(arr0, axis=1)
    # print(f'nnz_per_row is {nnz_per_row} !!!')
    return arr0


def gen_random_mat_a2(nnz_rows, cols):
    rows = len(nnz_rows)
    arr0 = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        nnz_row_i = nnz_rows[i]
        indices = np.random.choice(cols, size=nnz_row_i, replace=False)
        for j in indices:
            arr0[i][j] = 1.0

    nnzs = int(np.sum(arr0))
    print(f'density is {nnzs / (rows * cols)} !!!')
    return [nnzs], arr0


def gen_random_mat_b(rows, cols):
    mat_b = np.random.rand(rows, cols)
    return mat_b


def gen_mat_c(mat_a, mat_b):
    mat_c = np.dot(mat_a, mat_b)
    return mat_c


def gen_p_mat_c(mat_a, mat_b, tile_size_a, tile_size_b, cycle, row_PE, file_name):
    s_r_a = tile_size_a * row_PE
    e_r_a = tile_size_a * (row_PE + 1)
    s_c_a = tile_size_a * cycle
    e_c_a = tile_size_a * (cycle + 1)

    s_r_b = tile_size_a * cycle
    e_r_b = tile_size_a * (cycle + 1)
    s_c_b = 0
    e_c_b = tile_size_b

    p_mat_c = gen_mat_c(mat_a[s_r_a:e_r_a, s_c_a:e_c_a], mat_b[s_r_b:e_r_b, s_c_b:e_c_b])
    write_mat_file1(file_name, 'w', p_mat_c, 1)


def unroll_loop0(file_name, s_mat):
    csr = csr_matrix(s_mat)
    ind_ptr = csr.indptr.tolist()
    len_i = len(ind_ptr) - 1
    left_brace = '{'
    right_brace = '}'

    f = open(file_name, 'w')
    out_loop = 1
    f.write(f'for (unsigned int i = 0; i < {out_loop}; i++)\n')
    f.write('chess_prepare_for_pipelining\n')
    f.write(f'chess_loop_range({out_loop},)\n')
    f.write(f'{left_brace}\n')
    for i in range(len_i):
        nnz_row_i = ind_ptr[i + 1] - ind_ptr[i]

        f.write('  window_decr_v8(matB, row_idx1_b);\n')
        f.write('  row_idx0_b = 0;\n')
        f.write('  acc0 = null_v8float();\n')
        f.write(f'  for (int8_t j = 0; j < {nnz_row_i}; j++)\n')
        f.write('  chess_prepare_for_pipelining\n')
        f.write(f'  chess_loop_range({nnz_row_i},)\n')
        f.write(f'  {left_brace}\n')
        f.write('    val = window_readincr(matA);\n')
        f.write('    buf_mat_a = upd_elem(buf_mat_a, 0, val);\n')
        f.write('    row_idx1_b = window_readincr(indices);\n')
        f.write('    row_idx_diff = row_idx1_b - row_idx0_b;\n')
        f.write('    row_idx0_b = row_idx1_b;\n')
        f.write('    window_incr_v8(matB, row_idx_diff);\n')
        f.write('    buf_mat_b = window_read_v8(matB);\n')
        f.write('    acc0 = fpmac(acc0, buf_mat_a, 0, 0x00000000, buf_mat_b, 0, 0x76543210);\n')
        f.write(f'  {right_brace}\n')
        f.write('  window_writeincr(matC, acc0);\n\n')

    f.write(f'{right_brace}\n')


def unroll_loop1(file_name, s_mat):
    csr = csr_matrix(s_mat)
    ind_ptr = csr.indptr.tolist()
    len_i = len(ind_ptr) - 1
    left_brace = '{'
    right_brace = '}'

    f = open(file_name, 'w')
    out_loop = 1
    f.write(f'for (unsigned int i = 0; i < {out_loop}; i++)\n')
    f.write('chess_prepare_for_pipelining\n')
    f.write(f'chess_loop_range({out_loop},)\n')
    f.write(f'{left_brace}\n')
    for i in range(len_i):
        nnz_row_i = ind_ptr[i + 1] - ind_ptr[i]

        f.write('  window_decr_v8(matB, row_idx1_b);\n')
        f.write('  row_idx0_b = 0;\n')
        f.write('  acc0 = null_v8float();\n')
        for j in range(nnz_row_i):
            f.write('  val = window_readincr(matA);\n')
            f.write('  buf_mat_a = upd_elem(buf_mat_a, 0, val);\n')
            f.write('  row_idx1_b = window_readincr(indices);\n')
            f.write('  row_idx_diff = row_idx1_b - row_idx0_b;\n')
            f.write('  row_idx0_b = row_idx1_b;\n')
            f.write('  window_incr_v8(matB, row_idx_diff);\n')
            f.write('  buf_mat_b = window_read_v8(matB);\n')
            f.write('  acc0 = fpmac(acc0, buf_mat_a, 0, 0x00000000, buf_mat_b, 0, 0x76543210);\n')

        f.write('  window_writeincr(matC, acc0);\n\n')

    f.write(f'{right_brace}\n')


def unroll_loop2(file_name, s_mat):
    csr = csr_matrix(s_mat)
    ind_ptr = csr.indptr.tolist()
    len_i = len(ind_ptr) - 1
    nnz_per_row = []
    for i in range(len_i):
        nnz_row_i = ind_ptr[i + 1] - ind_ptr[i]
        nnz_per_row.append(nnz_row_i)

    num_par = 16
    ave_row_per_par = int(len(nnz_per_row) / num_par)
    par_dic = {}
    idx_par = -1
    max0 = 1024
    max_vals = []
    for i in range(0, len_i, ave_row_per_par):
        max1 = max(nnz_per_row[i:i + ave_row_per_par])
        max_vals.append(max1)
        if -2 <= max1 - max0 <= 0:
            par_dic[idx_par][2] += ave_row_per_par
        else:
            idx_par += 1
            par_dic[idx_par] = [max1, i, ave_row_per_par]
            max0 = max1

    max_str = [str(m) for m in max_vals]
    max_str = ', '.join(max_str)
    print(f'max nnz is: {max_str}')

    left_brace = '{'
    right_brace = '}'
    space = ' '
    f = open(file_name, 'w')
    out_loop = 1
    f.write(f'for (unsigned int i = 0; i < {out_loop}; i++)\n')
    f.write('chess_prepare_for_pipelining\n')
    f.write(f'chess_loop_range({out_loop},)\n')
    f.write(f'{left_brace}\n')
    for p in par_dic.keys():
        p_val = par_dic[p]
        loop0 = p_val[2]
        loop1 = p_val[0]
        f.write(f'{space * 2}for (unsigned int j = 0; j < {loop0}; j++)\n')
        f.write(f'{space * 2}chess_prepare_for_pipelining\n')
        f.write(f'{space * 2}chess_loop_range({loop0},)\n')
        f.write(f'{space * 2}{left_brace}\n')
        f.write(f'{space * 4}window_decr_v8(matB, row_idx1_b);\n')
        f.write(f'{space * 4}row_idx0_b = 0;\n')
        f.write(f'{space * 4}acc0 = null_v8float();\n')
        f.write(f'{space * 4}for (int8_t k = 0; k < {loop1}; k++)\n')
        f.write(f'{space * 4}chess_prepare_for_pipelining\n')
        f.write(f'{space * 4}chess_loop_range({loop1},)\n')
        f.write(f'{space * 4}{left_brace}\n')
        f.write(f'{space * 6}val = window_readincr(matA);\n')
        f.write(f'{space * 6}buf_mat_a = upd_elem(buf_mat_a, 0, val);\n')
        f.write(f'{space * 6}row_idx1_b = window_readincr(indices);\n')
        f.write(f'{space * 6}row_idx_diff = row_idx1_b - row_idx0_b;\n')
        f.write(f'{space * 6}row_idx0_b = row_idx1_b;\n')
        f.write(f'{space * 6}window_incr_v8(matB, row_idx_diff);\n')
        f.write(f'{space * 6}buf_mat_b = window_read_v8(matB);\n')
        f.write(f'{space * 6}acc0 = fpmac(acc0, buf_mat_a, 0, 0x00000000, buf_mat_b, 0, 0x76543210);\n')
        f.write(f'{space * 4}{right_brace}\n')
        f.write(f'{space * 4}window_writeincr(matC, acc0);\n')
        f.write(f'{space * 2}{right_brace}\n\n')

    f.write(f'{right_brace}\n')

    return par_dic


def write_mat_file0(file_name, s_mat, iteration):
    csr = csr_matrix(s_mat)
    data = csr.data
    # indices = csr.indices.tolist()
    indices = csr.indices
    ind_ptr = csr.indptr.tolist()
    len_d = len(data)
    total_num = math.ceil(len_d / 16) * 16
    z = total_num - len_d
    z_arr0 = np.zeros(z, dtype=np.float32)
    z_arr1 = np.zeros(z, dtype=np.int32)
    arr0 = np.concatenate((data, z_arr0))
    arr1 = np.concatenate((indices, z_arr1), dtype=np.int32)

    rem = total_num % 2
    if rem == 1:
        n_len = total_num + 1
    else:
        n_len = total_num

    f0 = open(file_name, 'w')
    for i in range(iteration):
        for j in range(0, n_len, 2):
            if j + 1 < total_num:
                f0.write(f'{arr0[j]}  {arr0[j + 1]}\n')
            else:
                f0.write(f'{arr0[j]}  {0.0}\n')
    f0.close()

    num_per_row = 8
    t, rem3 = divmod(total_num, num_per_row)
    if rem3 != 0:
        n_len = total_num + (num_per_row - rem3)
    else:
        n_len = total_num

    f1 = open('./mat/indices.txt', 'w')
    for i in range(iteration):
        for j in range(0, n_len, num_per_row):
            if j != 0:
                f1.write('\n')

            for k in range(j, j+num_per_row):
                if k < total_num:
                    f1.write(f'{arr1[k]}  ')
                else:
                    f1.write(f'{0}  ')
    f1.close()

    left_brace = '{'
    right_brace = '}'
    print(f'nnz: {csr.nnz}')
    print(f'final nnz: {total_num}')
    print(f'density: {csr.nnz / total_num}')
    # indices_str = [str(ind) for ind in indices]
    # indices_str = ', '.join(indices_str)
    # print(f'int8_t indices[{len(indices)}] = {left_brace}{indices_str}{right_brace};')
    ind_ptr_str = [str(ind) for ind in ind_ptr]
    ind_ptr_str = ', '.join(ind_ptr_str)
    print(f'int ind_ptr[{len(ind_ptr)}] = {left_brace}{ind_ptr_str}{right_brace};')


def write_mat_file1(file_name, w, mat, iteration):
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


def write_mat_file2(file_name, s_mat, par_dic, iteration):
    rows, cols = s_mat.shape
    csr = csr_matrix(s_mat)
    data = csr.data.tolist()
    indices = csr.indices.tolist()
    ind_ptr = csr.indptr.tolist()

    d_arr1 = []
    ind_arr1 = []
    ptr_arr = [0]

    idx_par = 0
    r_e = 0
    max1 = 0
    for i in range(rows):
        if i == r_e:
            # print(f'idx_par: {idx_par}')
            max1 = par_dic[idx_par][0]
            r_s = par_dic[idx_par][1]
            r_e = r_s + par_dic[idx_par][2]
            idx_par += 1

        s = ind_ptr[i]
        e = ind_ptr[i + 1]
        ptr_arr.append(ptr_arr[i] + max1)
        nnz_i = e - s
        rem0 = max1 - nnz_i
        if rem0 == 0:
            d_arr1.extend(data[s:e])
            ind_arr1.extend(indices[s:e])
            continue

        col0 = indices[s]
        col1 = indices[e - 1]
        d_arr0 = []
        ind_arr0 = []
        count_rem = 0
        if col1 - col0 + 1 - nnz_i >= rem0:
            for j in range(col0, col1 + 1):
                ele = s_mat[i][j]
                if count_rem < rem0:
                    d_arr0.append(ele)
                    ind_arr0.append(j)
                    if ele == 0:
                        count_rem += 1
                elif ele > 0:
                    d_arr0.append(ele)
                    ind_arr0.append(j)
        else:
            if col0 >= rem0:
                d_arr0[:] = s_mat[i][(col0 - rem0):col0]
                ind_arr0[:] = range(col0 - rem0, col0)
                d_arr0 += data[s:e]
                ind_arr0 += indices[s:e]
            else:
                d_arr0[:] = s_mat[i][0:col0]
                ind_arr0[:] = range(0, col0)
                rem1 = rem0 - col0
                count_rem = 0
                for j in range(col0, cols):
                    ele = s_mat[i][j]
                    if count_rem < rem1:
                        d_arr0.append(ele)
                        ind_arr0.append(j)
                        if ele == 0:
                            count_rem += 1
                    elif ele > 0:
                        d_arr0.append(ele)
                        ind_arr0.append(j)

        d_arr1 = d_arr1 + d_arr0
        ind_arr1 = ind_arr1 + ind_arr0

    total_num = rows * cols
    len_d = len(d_arr1)
    z = total_num - len_d
    z_arr0 = [0.0] * z
    z_arr1 = [0] * z
    arr0 = d_arr1 + z_arr0
    arr1 = ind_arr1 + z_arr1

    rem2 = total_num % 2
    if rem2 == 1:
        n_len = total_num + 1
    else:
        n_len = total_num

    f0 = open(file_name, 'w')
    f1 = open('./mat/indices.txt', 'w')
    for i in range(iteration):
        for j in range(0, n_len, 2):
            if j + 1 < total_num:
                f0.write(f'{arr0[j]}  {arr0[j + 1]}\n')
                f1.write(f'{arr1[j]}  {arr1[j + 1]}\n')
            else:
                f0.write(f'{arr0[j]}  {0.0}\n')
                f1.write(f'{arr1[j]}  {0.0}\n')
    f0.close()
    f1.close()

    left_brace = '{'
    right_brace = '}'
    print(f'nnz: {csr.nnz}')
    print(f'density: {csr.nnz / total_num}')
    # indices_str = [str(ind) for ind in indices]
    # indices_str = ', '.join(indices_str)
    # print(f'int8_t indices[{len(indices)}] = {left_brace}{indices_str}{right_brace};')
    ind_ptr_str = [str(ind) for ind in ptr_arr]
    ind_ptr_str = ', '.join(ind_ptr_str)
    print(f'int ind_ptr[{len(ptr_arr)}] = {left_brace}{ind_ptr_str}{right_brace};')

    return ptr_arr, ind_arr1, d_arr1


def write_mat_file_ind_d0(s_mat, w, ind_name, data_name):
    csr = csr_matrix(s_mat)
    indices = csr.indices
    ind_ptr = csr.indptr.tolist()
    len_d = len(indices)
    data = np.ones(len_d, dtype=np.float32)
    total_num = math.ceil(len_d / 16) * 16
    z = total_num - len_d
    z_arr0 = np.zeros(z, dtype=np.float32)
    z_arr1 = np.zeros(z, dtype=np.int32)
    arr0 = np.concatenate((data, z_arr0), dtype=np.float32)
    arr1 = np.concatenate((indices, z_arr1), dtype=np.int32)

    num_per_row = 8
    f0 = open(ind_name, w)
    for j in range(0, total_num, num_per_row):
        if j != 0:
            f0.write('\n')
        for k in range(j, j+num_per_row):
            f0.write(f'{arr1[k]}  ')
    f0.close()

    num_per_row = 2
    f1 = open(data_name, w)
    for j in range(0, total_num, num_per_row):
        if j != 0:
            f1.write('\n')
        for k in range(j, j+num_per_row):
            f1.write(f'{arr0[k]}  ')
    f1.close()

    left_brace = '{'
    right_brace = '}'
    print(f'nnz: {csr.nnz}')
    print(f'final nnz: {total_num}')
    # indices_str = [str(ind) for ind in indices]
    # indices_str = ', '.join(indices_str)
    # print(f'int8_t indices[{len(indices)}] = {left_brace}{indices_str}{right_brace};')
    ind_ptr_str = [str(ind) for ind in ind_ptr]
    ind_ptr_str = ', '.join(ind_ptr_str)
    print(f'int ind_ptr[{len(ind_ptr)}] = {left_brace}{ind_ptr_str}{right_brace};')


def write_mat_file_ind_d1(s_mat, par_dic, w, ind_name, data_name):
    csr = csr_matrix(s_mat)
    ind_ptr = csr.indptr.tolist()
    indices0 = csr.indices.tolist()

    len_i = len(ind_ptr) - 1
    idx_par = 0
    r_e = 0
    max0 = 0
    indices1 = []
    data = []
    for i in range(len_i):
        if i == r_e:
            max0 = par_dic[idx_par][0]
            r_s = par_dic[idx_par][1]
            r_e = r_s + par_dic[idx_par][2]
            idx_par += 1

        s_p = ind_ptr[i]
        e_p = ind_ptr[i + 1]
        nnz_row_i = e_p - s_p
        rem = max0 - nnz_row_i
        if nnz_row_i == 0:
            indices1.extend([0] * max0)
            data.extend([0.0] * max0)
            rem = 0
        else:
            indices1.extend(indices0[s_p:e_p])
            data.extend([1.0] * nnz_row_i)

        if rem > 0:
            ind = indices1[-1]
            indices1.extend([ind] * rem)
            data.extend([0.0] * rem)

    len_d = len(data)
    total_num = math.ceil(len_d / 16) * 16
    z = total_num - len_d
    if z > 0:
        indices1.extend([0] * z)
        data.extend([0] * z)

    iteration = 1
    f0 = open(data_name, w)
    for i in range(iteration):
        for j in range(0, total_num, 2):
            f0.write(f'{data[j]}  {data[j + 1]}\n')
    f0.close()

    num_per_row = 8
    f1 = open(ind_name, w)
    if w == 'a':
        f1.write('\n')
    for i in range(iteration):
        for j in range(0, total_num, num_per_row):
            if j != 0:
                f1.write('\n')
            for k in range(j, j + num_per_row):
                f1.write(f'{indices1[k]}  ')
    f1.close()


def write_mat_b_file(file_name, w, mat, iteration):
    rows, cols = mat.shape
    col_tiles = int(cols / 8)
    if col_tiles > 1:
        arr0 = np.zeros((rows*col_tiles, 8))
        for i in range(col_tiles):
            arr0[i*rows:(i+1)*rows, :] = mat[:, i*8:(i+1)*8]
    else:
        arr0 = mat

    write_mat_file1(file_name, w, arr0, iteration)


def compare_mat(a, b):
    dif = np.subtract(a, b)
    dif = np.absolute(dif)
    max_dif = np.amax(dif)
    if max_dif == 0:
        print('No dif !!')
    else:
        print(f'Max dif is {max_dif} !!')


def csr_to_dense(ind_ptr, indices, data, rows, cols):
    ind_ptr = np.array(ind_ptr)
    indices = np.array(indices)
    data = np.array(data)
    csr = csr_matrix((data, indices, ind_ptr), shape=(rows, cols))
    dense = csr.toarray()
    return dense


class MovingAverage(object):
    """Computes and stores the average and current value"""

    def __init__(self, th):
        self.th = th
        self.pre_avg = 0
        self.cur_avg = 0
        self.max = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.pre_avg = 0
        self.cur_avg = 0
        self.max = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.pre_avg = self.cur_avg
        # self.cur_avg = self.sum / self.count
        self.cur_avg = val
        if self.pre_avg == 0:
            self.pre_avg = self.cur_avg

        # if self.pre_avg > 4 and abs(self.cur_avg - self.pre_avg) / self.pre_avg >= self.th:
        # if self.pre_avg > 4 and abs(self.cur_avg - self.pre_avg) >= 4:
        if abs(self.cur_avg - self.pre_avg) >= 4:
            return True
        else:
            if val > self.max:
                self.max = val
            return False


def intra_partition(nnz):
    ave_nnz = MovingAverage(0.15)
    par_dic = {}
    idx_par = 0
    len_d = len(nnz)
    i = 0
    for i in range(len_d):
        nnz_row_i = nnz[i]
        s0 = ave_nnz.update(nnz_row_i)
        if s0:
            par_dic[idx_par] = [ave_nnz.max, i - ave_nnz.count + 1, ave_nnz.count - 1]
            idx_par += 1
            ave_nnz.reset()
            ave_nnz.update(nnz_row_i)

    if not par_dic:
        par_dic[idx_par] = [ave_nnz.max, 0, ave_nnz.count]
    elif par_dic[idx_par - 1][1] + par_dic[idx_par - 1][2] < len_d:
        par_dic[idx_par] = [ave_nnz.max, i - ave_nnz.count + 1, ave_nnz.count]

    return par_dic


def calc_density(par_dic, tile_size):
    total_nnz = 0
    for k, v in par_dic.items():
        nnz = v[0]
        length = v[2]
        total_nnz += nnz * length

    density = total_nnz / (tile_size * tile_size)
    total_nnz = math.ceil(total_nnz / 16) * 16
    return density, total_nnz


def inter_partition(s_mat, tile_size):
    rows, cols = s_mat.shape
    nnz_rows = []
    max_nnz = []
    for i in range(rows):
        nnz_rows.append([])
        for j in range(0, cols, tile_size):
            p_row = s_mat[i, j:j+tile_size]
            nnz_i = np.count_nonzero(p_row)
            nnz_rows[i].append(nnz_i)
        max_nnz.append(max(nnz_rows[i]))

    idx_par = 0
    par_dics = {}
    row_indices_sparse = []
    nnz_tiles = []
    for i in range(0, rows, tile_size):
        par = intra_partition(max_nnz[i:i+tile_size])
        par_dics[idx_par] = par
        idx_par += 1
        density, total_nnz = calc_density(par, tile_size)
        print(f'density in inter_partition is {density} !!!')

        if density > 0.6:
            row_indices_sparse.append(0)
        else:
            row_indices_sparse.append(1)

        nnz_tiles.append(total_nnz)
    return par_dics, row_indices_sparse, nnz_tiles


if __name__ == "__main__":
    r_a = 64
    c_a = 64
    r_b = c_a
    c_b = 16
    r_c = r_a
    c_c = c_b

    mat_a = gen_random_mat_a(r_a, c_a, 0.4)
    add_val(mat_a)

    # mat_a = gen_random_mat_a1(r_a, c_a, 0.4)

    par_dics = inter_partition(mat_a, 16)

    # write_mat_file0('./mat/a_csr.txt', mat_a, 1)
    # write_mat_file1('./mat/a.txt', 'w', mat_a, 1)

    # write_mat_file1('./mat/row_fifo0.txt', 'w', mat_a, 1)
    # write_mat_file_ind_d0('./mat/row_fifo0.txt', mat_a, 1)

    # unroll_loop0('./mat/loop0.txt', mat_a)
    # unroll_loop1('./mat/loop1.txt', mat_a)
    # par_dic = unroll_loop2('./mat/loop2.txt', mat_a)
    # print(par_dic)
    # ptr_arr, ind_arr, d_arr = write_mat_file2('./mat/a_csr.txt', mat_a, par_dic, 1)
    # dense = csr_to_dense(ptr_arr, ind_arr, d_arr, r_a, c_a)
    # compare_mat(dense, mat_a)

    # mat_b = gen_random_mat_b(r_b, c_b)
    # write_mat_file1('./mat/b0.txt', 'w', mat_b, 1)
    # write_mat_file1('./mat/b1.txt', 'w', mat_b, 1)

    # write_mat_b_file('./mat/col_fifo0.txt', 'w', mat_b, 1)
    #
    # mat_c = gen_mat_c(mat_a, mat_b)
    # write_mat_file1('./mat/c.txt', 'w', mat_c, 1)

    print('sparse mat finish !!!')
