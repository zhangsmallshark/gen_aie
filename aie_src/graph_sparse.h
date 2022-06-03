/*
*/
#include <adf.h>
#include "aie_kernels.h"
#include "system_settings.h"

using namespace adf;

template<int COL, int ROW>
class SystolicGraph0 : public graph
{
private:
  kernel PEs[N_ROWS_PE*N_COLS_PE];

public:
  port<input> row_fifos_i[N_ROWS_S_PE];
  port<input> row_fifos_d[N_ROWS_PE];
  port<input> col_fifos[N_COLS_PE];
  port<output> p_outs[N_ROWS_PE*N_COLS_PE];
  port<output> null_outs[N_ROWS_PE+N_ROWS_S_PE+N_COLS_PE];

  SystolicGraph0()
  {
    int idx0 = 0;
    int idx1 = 0;
    for (int i=0; i<N_ROWS_PE; i++) {
      for (int j=0; j<N_COLS_PE; j++) {

        idx0 = i * N_COLS_PE + j;
        if (kSparse0[i] == 1) {
// begin0
          PEs[idx0] = kernel::create(SparseTensorPE);
          source(PEs[idx0]) = "aie_kernels/sparse_tensor_PE.cpp";

          if (j == 0) {
            connect< window<kNNZ0[0]*N_BYTES_UINT8> > (row_fifos_i[idx1], PEs[idx0].in[0]);
            idx1 = idx1 + 1;
            connect< window<kNNZ0[0]*N_BYTES_FLOAT> > (row_fifos_d[i], PEs[idx0].in[1]);
          }
          else {
            connect< window<kNNZ0[0]*N_BYTES_UINT8> > (PEs[idx0-1].out[0], PEs[idx0].in[0]);
            connect< window<kNNZ0[0]*N_BYTES_FLOAT> > (PEs[idx0-1].out[1], PEs[idx0].in[1]);
          }
// end0
          if (i == 0) {
            connect< window<N_SAMPLES_WINDOW_B*N_BYTES_FLOAT> > (col_fifos[j], PEs[idx0].in[2]);
          }
          else {
            connect< window<N_SAMPLES_WINDOW_B*N_BYTES_FLOAT> > (PEs[(i-1)*N_COLS_PE+j].out[2], PEs[idx0].in[2]);
          }
          
          connect< window<N_SAMPLES_WINDOW_C*N_BYTES_FLOAT> > (PEs[idx0].out[3], p_outs[idx0]);
        }
        else {
          PEs[idx0] = kernel::create(TensorPE);
          source(PEs[idx0]) = "aie_kernels/tensor_PE.cpp";

          if (i == 0) {
            connect< window<N_SAMPLES_WINDOW_B*N_BYTES_FLOAT> > (col_fifos[j], PEs[idx0].in[1]);
          }
          else {
            connect< window<N_SAMPLES_WINDOW_B*N_BYTES_FLOAT> > (PEs[(i-1)*N_COLS_PE+j].out[2], PEs[idx0].in[1]);
          }

          if (j == 0) {
            connect< window<N_SAMPLES_WINDOW_A*N_BYTES_FLOAT> > (row_fifos_d[i], PEs[idx0].in[0]);
          }
          else {
            connect< window<N_SAMPLES_WINDOW_A*N_BYTES_FLOAT> > (PEs[idx0-1].out[1], PEs[idx0].in[0]);
          }

          connect< window<N_SAMPLES_WINDOW_C*N_BYTES_FLOAT> > (PEs[idx0].out[0], p_outs[idx0]);
        }

        // location<kernel>(PEs[idx0]) = tile(COL+j, ROW+i);
        location<kernel>(PEs[idx0]) = tile(COL+i, ROW+j);
        runtime<ratio>(PEs[idx0]) = 0.6;
      }
    }

// begin1
connect< window<N_SAMPLES_WINDOW_B*N_BYTES_FLOAT> > (PEs[0].out[2], null_outs[0]);
connect< window<kNNZ0[0]*N_BYTES_UINT8> > (PEs[0].out[0], null_outs[1]);
connect< window<kNNZ0[0]*N_BYTES_FLOAT> > (PEs[0].out[1], null_outs[2]);
// end1

  }
};


class TopGraph : public graph
{
public:
  port<input> ins0[N_ROWS_S_PE];
  port<input> ins1[N_ROWS_PE];  
  port<input> ins2[N_COLS_PE];
  port<output> outs0[N_ROWS_PE*N_COLS_PE];
  port<output> outs1[N_ROWS_PE+N_ROWS_S_PE+N_COLS_PE];

  SystolicGraph0<25,0> g0;

  TopGraph()
  {

// begin2  
    for (int i=0; i<N_ROWS_S_PE; i++) {
      connect < window<kNNZ0[0]*N_BYTES_UINT8> > (ins0[i], g0.row_fifos_i[i]);
    }

    for (int i=0; i<N_ROWS_PE; i++) {
      if (kSparse0[i] == 1) {
        connect < window<kNNZ0[0]*N_BYTES_FLOAT> > (ins1[i], g0.row_fifos_d[i]);
      }
      else {
        connect < window<N_SAMPLES_WINDOW_A*N_BYTES_FLOAT> > (ins1[i], g0.row_fifos_d[i]);
      }
    }
// end2

    for (int j=0; j<N_COLS_PE; j++) {
      connect < window<N_SAMPLES_WINDOW_B*N_BYTES_FLOAT> > (ins2[j], g0.col_fifos[j]);
    }

    for (int i=0; i<N_ROWS_PE*N_COLS_PE; i++) {
      connect < window<N_SAMPLES_WINDOW_C*N_BYTES_FLOAT> > (g0.p_outs[i], outs0[i]);
    }

// begin3
connect < window<N_SAMPLES_WINDOW_B*N_BYTES_FLOAT> > (g0.null_outs[0], outs1[0]);
connect < window<kNNZ0[0]*N_BYTES_UINT8> > (g0.null_outs[1], outs1[1]);
connect < window<kNNZ0[0]*N_BYTES_FLOAT> > (g0.null_outs[2], outs1[2]);
// end3

  }
};
