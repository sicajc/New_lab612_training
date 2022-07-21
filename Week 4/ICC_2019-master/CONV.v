`timescale 1ns/10ps

module  CONV(
            clk,
            reset,
            busy,
            ready,
            iaddr,
            idata,
            cwr,
            caddr_wr,
            cdata_wr,
            crd,
            caddr_rd,
            cdata_rd,
            csel
        );

input						clk;
input						reset         ;
output	reg					busy          ;
input						ready         ;
output	reg [11:0]			iaddr         ;
input	[19:0]				idata         ;
output	 reg				cwr           ;
output	 reg[11:0]  		caddr_wr      ;
output reg signed[19:0] 	cdata_wr      ;
output	 reg				crd           ;
output	reg [11:0]			caddr_rd      ;
input	 [19:0]				cdata_rd      ;
output	 reg [2:0]			csel          ;







endmodule
