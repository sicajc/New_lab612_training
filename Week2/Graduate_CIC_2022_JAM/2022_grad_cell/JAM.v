module JAM (
           input CLK,
           input RST,
           output reg [2:0] W,
           output reg [2:0] J,
           input [6:0] Cost,
           output reg [3:0] MatchCount,
           output reg [9:0] MinCost,
           output reg Valid );
parameter MAX_COMBINATIONS = 40320;

//Work_Calculator_states
parameter  IDLE= 'd0;
parameter  WORK_CAL= 'd1;
parameter  WORK_DONE= 'd2;

//Permutator states
parameter CAL =  'd0;
parameter PERM_DONE = 'd1;
parameter EXHAUST = 'd2;

//State registers
reg[1:0] work_current_state,work_next_state;
reg[1:0] perm_current_state,perm_next_state;

//State indicators
wire work_state_IDLE = work_current_state == IDLE;
wire work_state_WORK_CAL = work_current_state == WORK_CAL;
wire work_state_WORK_DONE = work_current_state == WORK_DONE;
wire perm_state_CAL = perm_current_state ==CAL ;
wire perm_state_PERM_DONE = perm_current_state ==PERM_DONE ;
wire perm_state_EXHAUST = perm_current_state == EXHAUST ;

//Registers
reg[3:0][0:7] job_sequence_reg;

//Counters
reg[2:0] job_cnt;
reg[15:0] work_cal_cnt;
reg[15:0] perm_cnt;
reg[3:0] bff_ptr;

//FLAGS
wire bff_not_empty = bff_ptr != 0;
wire work_cal_done_flag = job_cnt == 'd8;
wire perm_cal_done_flag = work_cal_cnt == MAX_COMBINATIONS;


//Work_Calculator_FSM
always @(posedge CLK or posedge RST)
begin
    work_current_state <= RST ? IDLE : work_next_state;
end

always @(*)
begin
    case(work_current_state)
    IDLE:
        work_next_state = bff_not_empty ? WORK_CAL : IDLE;
    WORK_CAL:
        work_next_state = work_cal_done_flag ? (perm_cal_done_flag ? WORK_DONE : IDLE) : WORK_CAL;
    WORK_DONE:
        work_next_state = IDLE;
    default:
        work_next_state = IDLE;
    endcase
end










endmodule
