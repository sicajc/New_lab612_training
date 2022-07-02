module JAM (
           input CLK,
           input RST,
           output reg [2:0] W,
           output reg [2:0] J,
           input [6:0] Cost,
           output reg [3:0] MatchCount,
           output reg [9:0] MinCost,
           output reg Valid );
//Give permutator 8 cycles for calculation
parameter MAX_COMBINATIONS = 40320;

//Work_Calculator_states
parameter  WORK_CAL=  'd0;
parameter  WORK_DONE= 'd1;

//State registers
reg[1:0] work_current_state,work_next_state;

//State indicators
wire work_state_WORK_CAL = work_current_state == WORK_CAL;
wire work_state_WORK_DONE = work_current_state == WORK_DONE;

//Registers
reg[2:0][0:7] job_sequence_reg;
reg[2:0][0:7] perm_pipe_reg;

//Counters
reg[3:0] job_cnt;
reg[15:0] work_cal_cnt;
reg[3:0] perm_cycle_cnt;

//FLAGS
wire work_cal_done_flag = job_cnt == 'd7;
wire max_comb_reach_flag = work_cal_cnt == MAX_COMBINATIONS;

//wires
wire[2:0][7:0] permutator_o;

//Work_Calculator_FSM
always @(posedge CLK or posedge RST)
begin
    work_current_state <= RST ? WORK_CAL : work_next_state;
end

always @(*)
begin
    case(work_current_state)
    WORK_CAL:
        work_next_state = work_cal_done_flag ? max_comb_reach_flag ? WORK_DONE : WORK_CAL ;
    WORK_DONE:
        work_next_state = WORK_DONE;
    default:
        work_next_state = WORK_CAL;
    endcase
end

always @(posedge clk or posedge RST)
begin
    if(RST)
    begin
        job_cnt <= 'd0;
    end
    else if(work_state_WORK_CAL)
    begin
        job_cnt <= work_cal_done_flag ? 'd0 : (job_cnt + 'd1);
    end
    else
    begin
        job_cnt <= job_cnt;
    end
end

always @(posedge clk or posedge RST)
begin
    if(RST)
    begin
        work_cal_cnt <= 'd0;
    end
    else if(work_cal_done_flag)
    begin
        work_cal_cnt <= work_cal_cnt + 'd1;
    end
    else
    begin
        work_cal_cnt <= work_cal_cnt;
    end
end

always @(posedge clk or posedge RST)
begin
    if(RST)
    begin
        perm_cycle_cnt <= 'd7;
    end
    else if(work_cal_done_flag)
    begin
        perm_cycle_cnt <= perm_cycle_cnt - 'd1;
    end
    else
    begin
        perm_cycle_cnt <= perm_cycle_cnt;
    end
end

integer i;
//Job sequences
always @(posedge clk or posedge RST)
begin
    for(i = 0; i < 8 ;i = i + 1)
    begin
        if(RST)
        begin
            job_sequence_reg[i] <= i;
        end
        else if(work_cal_done_flag)
        begin
            job_sequence_reg[i] <= permutator_o[i];
        end
        else
        begin
            job_sequence_reg[i] <= job_sequence_reg[i];
        end
    end
end

//Permutator
//Stage 1:Find pivot and pivoting number from job sequences

reg[5:0]
always @(*)
begin
    for(i = 0; i < 8 ;i = i+1)
    begin

    end
end







endmodule
