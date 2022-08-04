module JAM (
           input CLK,
           input RST,
           output  [2:0] W,
           output  [2:0] J,
           input [6:0] Cost,
           output  [3:0] MatchCount,
           output  [9:0] MinCost,
           output  Valid );
//Give permutator 8 cycles for calculation at the same time, calculate the minCost of previos permuted result
parameter MAX_COMBINATIONS = 40320;

//States
parameter FIND_PIVOT = 'd0;
parameter SWITCH_PIVOT_MIN = 'd1;
parameter SWITCH_PIVOT_MIN_WB = 'd2;
parameter FLIP = 'd3;
parameter WB  =  'd4;
parameter DONE = 'd5;

//State Register
reg[3:0] currentState,nextState;


endmodule
