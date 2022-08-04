
`timescale 1ns/10ps
module LBP ( clk, reset, gray_addr, gray_req, gray_ready, gray_data, lbp_addr, lbp_valid, lbp_data, finish);
input   	    clk;
input   	    reset;
output  [13:0] 	gray_addr;  //img addr
output         	gray_req;   //high --> get data
input   	    gray_ready; //high --> readt to get data
input   [7:0] 	gray_data;
output  [13:0] 	lbp_addr;
output  	    lbp_valid;  // if done ---> high
output  [7:0] 	lbp_data;
output  	    finish;
//====================================================================








endmodule
