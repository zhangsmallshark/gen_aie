
A * X * W 
def X_W(r_x, c_x, r_w, c_w, tile_size_x, tile_size_w):

    num_rows_dense_engine = 4
    num_cols_dense_engine = 50

    computed_rows_b = tile_size_x * num_cols_dense_engine
    computed_cols_b = tile_size_w * num_rows_dense_engine


    iterations_d = math.ceil(c_x / tile_size_x) * int(c_w / computed_cols_b)

    
    total_iterations_d = math.ceil(r_x / computed_rows_b) * iterations_d


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