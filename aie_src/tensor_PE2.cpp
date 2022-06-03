/*
*/
#include <adf.h>
#include <cstdint>
#include "system_settings.h"

void TensorPE2(
  input_window_float* __restrict left,
  input_window_float* __restrict top,
  output_window_float* __restrict p_mat_c,
  output_window_float* __restrict right,
  output_window_float* __restrict down)
{
  v8float buf_mat_a0 = undef_v8float();
  v8float buf_mat_a1 = undef_v8float();
  v8float buf_mat_b = undef_v8float();
  v8float acc0 = null_v8float();
  v8float acc1 = null_v8float();

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

  constexpr int kRowTilesA = R_A / 2;
  constexpr int kColTilesA = C_A / 8;
  constexpr int kColTilesB = C_B / 8;
  int num_incr_a = 0;
  int num_decr_a = 0;

  for (int i0 = 0; i0 < kRowTilesA; i0++)
  chess_prepare_for_pipelining
  chess_loop_range(kRowTilesA,)
  {
    window_incr_v8(left, num_incr_a);
    num_decr_a = 0;
    for (int i1 = 0; i1 < kColTilesB; i1++)
    chess_prepare_for_pipelining
    chess_loop_range(kColTilesB,)
    {
      acc0 = null_v8float();
      acc1 = null_v8float();
      window_decr_v8(left, num_decr_a);
      for (int i2 = 0; i2 < kColTilesA; i2++)
      chess_prepare_for_pipelining
      chess_loop_range(kColTilesA,)
      {
        window_incr_v8(left, kColTilesA);
        buf_mat_a1 = window_read_v8(left);
        window_decr_v8(left, kColTilesA);
        buf_mat_a0 = window_readincr_v8(left);

        // buf_mat_b = window_readincr_v8(top);
        // acc0 = fpmul(buf_mat_a0, 0, 0x00000000, buf_mat_b, 0, 0x76543210);
        for (int i3 = 0; i3 < 8; i3++)
        chess_prepare_for_pipelining
        chess_loop_range(8, 8)
        chess_flatten_loop
        {
          buf_mat_b = window_readincr_v8(top);
          acc0 = fpmac(acc0, xset_w(0, buf_mat_a0), i3, 0x00000000, buf_mat_b, 0, 0x76543210);
          acc1 = fpmac(acc1, xset_w(0, buf_mat_a1), i3, 0x00000000, buf_mat_b, 0, 0x76543210);
        }
      }

      window_incr_v8(p_mat_c, kColTilesB);
      window_write(p_mat_c, acc1);
      window_decr_v8(p_mat_c, kColTilesB);
      window_writeincr(p_mat_c, acc0);
      num_decr_a = kColTilesA;
    }
    num_incr_a = kColTilesA;
    window_decr_v8(top, R_B * kColTilesB);
  }


}
