/*
*/
#include <adf.h>
#include <cstdint>
#include "system_settings.h"

void TensorPE(
  input_window_float* __restrict left,
  input_window_float* __restrict top,
  output_window_float* __restrict p_mat_c,
  output_window_float* __restrict right,
  output_window_float* __restrict down)
{
  v8float buf_mat_a = undef_v8float();
  v8float buf_mat_b = undef_v8float();
  v8float acc0 = null_v8float();

  v32float buf_right = undef_v32float();
  v32float buf_down = undef_v32float();

  constexpr int kNReadsLeft = (R_A * C_A) / 32;
  for (int i0 = 0; i0 < kNReadsLeft; i0++)
  chess_prepare_for_pipelining
  chess_loop_range(kNReadsLeft,)
  {
    buf_right = window_readincr_v32(left);
    window_writeincr(right, buf_right);
  }
  window_decr_v32(left, kNReadsLeft);

  constexpr int kNReadsTop = (R_B * C_B) / 32;
  for (int i0 = 0; i0 < kNReadsTop; i0++)
  chess_prepare_for_pipelining
  chess_loop_range(kNReadsTop,)
  {
    buf_down = window_readincr_v32(top);
    window_writeincr(down, buf_down);    
  }
  window_decr_v32(top, kNReadsTop);

  constexpr int kColTilesA = C_A / 8;
  constexpr int kColTilesB = C_B / 8;
  int num_reads_a = 0;

  for (uint8_t i0 = 0; i0 < R_A; i0++)
  chess_prepare_for_pipelining
  chess_loop_range(R_A,)
  {
    num_reads_a = 0;
    for (int i1 = 0; i1 < kColTilesB; i1++)
    chess_prepare_for_pipelining
    chess_loop_range(kColTilesB,)
    {
      acc0 = null_v8float();
      window_decr_v8(left, num_reads_a);
      for (int i2 = 0; i2 < kColTilesA; i2++)
      chess_prepare_for_pipelining
      chess_loop_range(kColTilesA,)
      {
        buf_mat_a = window_readincr_v8(left);

        // buf_mat_b = window_readincr_v8(top);
        // acc0 = fpmul(buf_mat_a, 0, 0x00000000, buf_mat_b, 0, 0x76543210);
        for (int i3 = 0; i3 < 8; i3++)
        // chess_prepare_for_pipelining
        // chess_loop_range(8, 8)
        chess_flatten_loop
        {
          buf_mat_b = window_readincr_v8(top);
          acc0 = fpmac(acc0, xset_w(0, buf_mat_a), i3, 0x00000000, buf_mat_b, 0, 0x76543210);
        }
      }

      window_writeincr(p_mat_c, acc0);
      num_reads_a = kColTilesA;
    }
    window_decr_v8(top, R_B * kColTilesB);
  }


}
