/*
*/
#include "graph.h"

PLIO *in0 = new PLIO("indices0", plio_64_bits, "./data/indices0.txt");
PLIO *in1 = new PLIO("row_fifo0", plio_64_bits, "./data/row_fifo0.txt");
PLIO *in2 = new PLIO("col_fifo0", plio_64_bits, "./data/col_fifo0.txt");

PLIO * out0 = new PLIO("p_out0", plio_64_bits, "./data/p_out0.txt");

PLIO * out1 = new PLIO("null_out0", plio_64_bits, "./data/null_out0.txt");
PLIO * out2 = new PLIO("null_out1", plio_64_bits, "./data/null_out1.txt");
PLIO * out3 = new PLIO("null_out2", plio_64_bits, "./data/null_out2.txt");

simulation::platform < 3, 4 > p(in0, in1, in2, out0, out1, out2, out3);

TopGraph g;

connect<> net10(p.src[0], g.ins0[0]);
connect<> net11(p.src[1], g.ins1[0]);
connect<> net12(p.src[2], g.ins2[0]);

connect<> net20(g.outs0[0], p.sink[0]);

connect<> net30(g.outs1[0], p.sink[1]);
connect<> net31(g.outs1[1], p.sink[2]);
connect<> net32(g.outs1[2], p.sink[3]);

#ifdef __AIESIM__
   int main(int argc, char ** argv)
   {
      g.init();
      g.run(N_ITERATIONS);
      g.end();

      return 0;
   }
#endif
