/*
left0: indices
left1: data
top: mat_b
*/

#include <adf.h>
#include <cstdint>
#include "system_settings.h"

// begin0
void SparseTensorPE(
// end0
  input_window_uint8* __restrict left0,
  input_window_float* __restrict left1,
  input_window_float* __restrict top,
  output_window_uint8* __restrict right0,
  output_window_float* __restrict right1,
  output_window_float* __restrict down,
  output_window_float* __restrict p_mat_c)
{
  v16uint8 buf_right0 = undef_v16uint8();
  v16float buf_right1 = undef_v16float();
  v32float buf_down = undef_v32float();

  constexpr int kNReadsLeft = kNNZLeft / 16;
  for (uint8_t i0 = 0; i0 < kNReadsLeft; i0++)
  chess_prepare_for_pipelining
  chess_loop_range(kNReadsLeft,)
  {
    buf_right0 = window_readincr_v16(left0);
    buf_right1 = window_readincr_v16(left1);
    window_writeincr(right0, buf_right0);
    window_writeincr(right1, buf_right1);
  }
  window_decr_v16(left0, kNReadsLeft);
  window_decr_v16(left1, kNReadsLeft);

  constexpr int kNReadsTop = (R_B * C_B) / 32;
  for (uint8_t i0 = 0; i0 < kNReadsTop; i0++)
  chess_prepare_for_pipelining
  chess_loop_range(kNReadsTop,)
  {
    buf_down = window_readincr_v32(top);
    window_writeincr(down, buf_down);    
  }
  window_decr_v32(top, kNReadsTop);

  v8float buf_mat_a = null_v8float();
  v8float buf_mat_b = undef_v8float();
  v8float acc0 = null_v8float();

  float val = 0.0;
  int row_idx0_b = 0;
  int row_idx1_b = 0;
  int row_idx_diff0 = 0;
  int row_idx_diff1 = 0;

  constexpr int kColTilesB = C_B / 8;
  int num_reads_a = 0;

// begin1
  for (uint8_t i0 = 0; i0 < R_A; i0++)
  chess_prepare_for_pipelining
  chess_loop_range(R_A,)
  {
    num_reads_a = 0;
    for (int i1 = 0; i1 < kColTilesB; i1++)
    chess_prepare_for_pipelining
    chess_loop_range(kColTilesB,)
    {
      row_idx0_b = 0;
      acc0 = null_v8float();
      window_decr(left0, num_reads_a);
      window_decr(left1, num_reads_a);      
      for (int i2 = 0; i2 < kNNZRowI; i2++)
      chess_prepare_for_pipelining
      chess_loop_range(kNNZRowI,)
      {
        val = window_readincr(left1);
        buf_mat_a = upd_elem(buf_mat_a, 0, val);
        row_idx1_b = window_readincr(left0);
        row_idx_diff0 = row_idx1_b - row_idx0_b;
        row_idx0_b = row_idx1_b;
        window_incr_v8(top, row_idx_diff0);
        buf_mat_b = window_read_v8(top);
        acc0 = fpmac(acc0, xset_w(0, buf_mat_a), 0, 0x00000000, buf_mat_b, 0, 0x76543210);
      }
      window_writeincr(p_mat_c, acc0);
      num_reads_a = kNNZRowI;
      row_idx_diff1 = R_B - row_idx0_b;
      window_incr_v8(top, row_idx_diff1);
    }
    window_decr_v8(top, R_B * kColTilesB);
  }
// end1

}
