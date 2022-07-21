`include "timescale.v"

(* blackbox = 1 *)
module flexmem_spram_256x32_4we(
	input [7:0] a,
	input [31:0] d,
	output [31:0] q,
	input [3:0] we,
	input clk
);
endmodule // SPBlock_512W64B8W

module eth_spram_256x32(
	// Generic synchronous single-port RAM interface
	clk, rst, ce, we, oe, addr, di, dato
);

  //
  // Generic synchronous single-port RAM interface
  //
  input           clk;  // Clock, rising edge
  input           rst;  // Reset, active high
  input           ce;   // Chip enable input, active high
  input  [3:0]    we;   // Write enable input, active high
  input           oe;   // Output enable input, active high
  input  [7:0]    addr; // address bus inputs
  input  [31:0]   di;   // input data bus
  output [31:0]   dato;   // output data bus

  wire [31:0]     q;
  flexmem_spram_256x32_4we flexmem_spram_256x32_4we
    (
      .a              (addr),
      .we             (ce ? we : 0),
      .clk            (clk),
      .d              (di),
      .q              (q)
      );
  
  assign dato = (oe & ce) ? q : {32{1'bz}};

endmodule
