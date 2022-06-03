/*
*/
#pragma once

// Matrix size
#define R_A 64
#define C_A 64
#define R_B (C_A)
#define C_B 16
#define R_C (R_A)
#define C_C (C_B)

// Window size
#define N_SAMPLES_WINDOW_A (R_A*C_A)
#define N_SAMPLES_WINDOW_B (R_B*C_B)
#define N_SAMPLES_WINDOW_C (R_C*C_C)

#define N_BYTES_UINT8 1
#define N_BYTES_FLOAT 4

// Systolic size
#define N_ROWS_PE 1
#define N_ROWS_S_PE 1
#define N_COLS_PE 1

// Sparse size
constexpr int kNNZ0[N_ROWS_PE] = {1112};
constexpr int kSparse0[N_ROWS_PE] = {1};

#define N_ITERATIONS 1

