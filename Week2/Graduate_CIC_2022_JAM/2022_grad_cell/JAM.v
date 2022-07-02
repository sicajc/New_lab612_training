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
reg[3:0] pivot_index;
wire[3:0] pivot_value = job_sequence_reg[pivot_index];

reg[6:0] cmpr_gt_bit_stream;
always @(*)
begin
    for(i = 1; i < 8 ;i = i+1)
    begin
        cmpr_gt_bit_stream[i-1] = (job_sequence_reg[i] > job_sequence_reg[i-1]); //This can be shared if we pipelined the result
    end
end

always @(posedge clk or posedge RST)
begin
    if(RST)
    begin
        pivot_index <= 'd0;
    end
    else
    begin
        case(cmpr_gt_bit_stream)
        7'b1xxxxxx:
        begin
            pivot_index<='d7;
        end
        7'b01xxxxx:
        begin
            pivot_index<='d6;
        end
        7'b001xxxx:
        begin
            pivot_index<='d5;
        end
        7'b0001xxx:
        begin
            pivot_index<='d4;
        end
        7'b00001xx:
        begin
            pivot_index<='d3;
        end
        7'b000001x:
        begin
            pivot_index<='d2;
        end
        7'b0000001:
        begin
            pivot_index<='d1;
        end
        default:
        begin
            pivot_index<='d0;
        end
        endcase
    end
end

//Stage 2 cmpr value of indexs greater than pivot index
//Scan through the sequence from index 7~0 for comparison
//Finally swap with the job_sequence pivot value
reg[7:0] value_gt_index_bit_stream;
reg[3:0] scan_ptr;

wire scan_done_flag = scan_ptr == 'd0;
always @(posedge clk or posedge RST)
begin
    if(RST)
    begin
        scan_ptr <= 'd7;
    end
    else if(work_cal_done_flag)
    begin
        scan_ptr <= 'd7;
    end
    else if(scan_done_flag)
    begin
        scan_ptr <= scan_ptr;
    end
    else
    begin
        scan_ptr <= scan_ptr - 'd1;
    end
end

//Stage 3 swaps every number after pivot index
always @(*)
begin
    case(pivot_index)
    'd0: //swap (1,7) (2,6) (3,5)
    begin
        permutator_o[0] = job_sequence_reg[0];
        permutator_o[1] = job_sequence_reg[7];
        permutator_o[2] = job_sequence_reg[6];
        permutator_o[3] = job_sequence_reg[5];
        permutator_o[4] = job_sequence_reg[4];
        permutator_o[5] = job_sequence_reg[3];
        permutator_o[6] = job_sequence_reg[2];
        permutator_o[7] = job_sequence_reg[1];
    end
    'd1: // swap (2,7) (3,6) (4,5)
    begin
        permutator_o[0] = job_sequence_reg[0];
        permutator_o[1] = job_sequence_reg[1];
        permutator_o[2] = job_sequence_reg[7];
        permutator_o[3] = job_sequence_reg[6];
        permutator_o[4] = job_sequence_reg[5];
        permutator_o[5] = job_sequence_reg[4];
        permutator_o[6] = job_sequence_reg[3];
        permutator_o[7] = job_sequence_reg[2];
    end
    'd2: //swap (3,7) (4,6)
    begin
        permutator_o[0] = job_sequence_reg[0];
        permutator_o[1] = job_sequence_reg[1];
        permutator_o[2] = job_sequence_reg[2];
        permutator_o[3] = job_sequence_reg[7];
        permutator_o[4] = job_sequence_reg[6];
        permutator_o[5] = job_sequence_reg[5];
        permutator_o[6] = job_sequence_reg[4];
        permutator_o[7] = job_sequence_reg[3];
    end
    'd3: //swap (4,7) (5,6)
    begin
        permutator_o[0] = job_sequence_reg[0];
        permutator_o[1] = job_sequence_reg[1];
        permutator_o[2] = job_sequence_reg[2];
        permutator_o[3] = job_sequence_reg[3];
        permutator_o[4] = job_sequence_reg[7];
        permutator_o[5] = job_sequence_reg[6];
        permutator_o[6] = job_sequence_reg[5];
        permutator_o[7] = job_sequence_reg[4];
    end
    'd4: //swap (5,7)
    begin
        permutator_o[0] = job_sequence_reg[0];
        permutator_o[1] = job_sequence_reg[1];
        permutator_o[2] = job_sequence_reg[2];
        permutator_o[3] = job_sequence_reg[3];
        permutator_o[4] = job_sequence_reg[4];
        permutator_o[5] = job_sequence_reg[7];
        permutator_o[6] = job_sequence_reg[6];
        permutator_o[7] = job_sequence_reg[5];
    end
    'd5: //swap (6,7)
    begin
        permutator_o[0] = job_sequence_reg[0];
        permutator_o[1] = job_sequence_reg[1];
        permutator_o[2] = job_sequence_reg[2];
        permutator_o[3] = job_sequence_reg[3];
        permutator_o[4] = job_sequence_reg[4];
        permutator_o[5] = job_sequence_reg[5];
        permutator_o[6] = job_sequence_reg[7];
        permutator_o[7] = job_sequence_reg[6];
    end
    default:
    begin
        for(i=0;i<8 ;i = i+1)
        begin
            permutator_o[i] = job_sequence_reg[i];
        end
    end
    endcase
end







endmodule
