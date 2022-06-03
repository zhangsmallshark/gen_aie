/*
*/
#pragma once

void TensorPE(
  input_window_float* __restrict left,
  input_window_float* __restrict top,
  output_window_float* __restrict p_mat_c,
  output_window_float* __restrict right,
  output_window_float* __restrict down);

 void SparseTensorPEV(
  input_window_uint8* __restrict left0,
  input_window_float* __restrict left1,
  input_window_float* __restrict top,
  output_window_uint8* __restrict right0,
  output_window_float* __restrict right1,
  output_window_float* __restrict down,
  output_window_float* __restrict p_mat_c);
