module JAM (
           input CLK,
           input RST,
           output  [2:0] W,
           output  [2:0] J,
           input [6:0] Cost,
           output reg [3:0] MatchCount,
           output  reg[9:0] MinCost,
           output  Valid );
//Give permutator 8 cycles for calculation at the same time, calculate the minCost of previos permuted result
parameter MAX_COMBINATIONS = 40320;

//CNTs
reg[2:0] workIndexCnt;
reg[11:0] permutationCnt;

//PTRs
reg[2:0] headPTR;
reg[2:0] tailPTR;
reg[2:0] pivot;

wire[2:0] headValue = J_tempSequenceReg[headPTR];
wire[2:0] tailValue = J_tempSequenceReg[tailPTR];
wire[2:0] pivotValue = J_SequenceReg[pivot];

//States
wire STATE_FIND_PIVOT_SET_TAILHEAD = workIndexCnt == 'd0;
wire STATE_SWITCH_PIVOT            = workIndexCnt >= 'd1 || workIndexCnt <= 'd4;
wire STATE_SWITCH_PIVOT_SWAP       = workIndexCnt == 'd5;
wire STATE_FLIP                    = workIndexCnt == 'd6;
wire STATE_WB                      = workIndexCnt == 'd7;
wire STATE_DONE                    = (permutationCnt == MAX_COMBINATIONS - 1);

//State Register
reg[3:0] currentState,nextState;

//Flags
wire switch_gt_MINdone_flag = STATE_SWITCH_PIVOT_SWAP;
wire PTR_met_flag = (headPTR +'d1) == tailPTR || headPTR == tailPTR;
wire MIN_COST_CalculationDone_flag = STATE_WB;

//-------------------------------CONTROL_PATH------------------------------//
always @(posedge CLK)
begin
    workIndexCnt <= RST ? 'd0 : (STATE_DONE ? workIndexCnt : (MIN_COST_CalculationDone_flag ?  'd0 : workIndexCnt + 'd1));
end

//Permutation CNT
always @(posedge CLK)
begin
    permutationCnt <= RST ? 'd0 : MIN_COST_CalculationDone_flag ? permutationCnt + 'd1 : permutationCnt;
end


//PTRS
//head ptr
always @(posedge CLK )
begin
    if(RST)
    begin
        headPTR <= 'd0;
    end
    else if(STATE_FIND_PIVOT_SET_TAILHEAD)
    begin
        headPTR <= pivot + 'd1;
    end
    else if(STATE_SWITCH_PIVOT)
    begin
        headPTR <= PTR_met_flag ?  headPTR :  headPTR + 'd1;
    end
    else
    begin
        headPTR <= headPTR;
    end
end

//tailPTR
always @(posedge CLK)
begin
    if(RST)
    begin
        tailPTR <= 'd0;
    end
    else if(STATE_FIND_PIVOT_SET_TAILHEAD)
    begin
        tailPTR <= 'd7;
    end
    else if(STATE_SWITCH_PIVOT)
    begin
        tailPTR <= PTR_met_flag ?  tailPTR :  tailPTR - 'd1;
    end
    else
    begin
        tailPTR <= tailPTR;
    end
end

//I/O
assign W = workIndexCnt;
assign J = J_SequenceReg[workIndexCnt];
assign Valid = STATE_DONE;


//---------------------CP------------------------//
//Registers
reg[2:0] J_SequenceReg;
reg[2:0] J_tempSequenceReg;
reg[9:0] localCostReg_rd;


//J_sequnceReg
integer i;
always @(posedge CLK)
begin
    for(i=0;i<8;i=i+1)
    begin
        if(RST)
        begin
            J_SequenceReg[i] <= i;
        end
        else if(STATE_WB)
        begin
            J_SequenceReg[i] <= J_tempSequenceReg[i];
        end
        else
        begin
            J_SequenceReg[i] <= J_SequenceReg[i];
        end
    end
end

wire tempValueUnChanged = (tempValueReg == 'd8);

//J_tempSequenceReg
always @(posedge CLK)
begin
    for(i=0;i<8;i=i+1)
    begin
        if(RST)
        begin
            J_tempSequenceReg[i] <= i;
        end
        else if(STATE_SWITCH_PIVOT_SWAP)
        begin
            if(tempValueUnChanged)
            begin
                J_tempSequenceReg[i] <= J_tempSequenceReg[i];
            end
            else
            begin
                J_tempSequenceReg[pivot] <= tempValueReg;
                J_tempSequenceReg[tempIndexReg] <= pivotValue;
            end
        end
        else if(STATE_FLIP)
        begin
            case(pivot)
                'd0:
                begin
                    J_tempSequenceReg[1] <= J_tempSequenceReg[7] ;
                    J_tempSequenceReg[7] <= J_tempSequenceReg[1] ;

                    J_tempSequenceReg[2] <= J_tempSequenceReg[6] ;
                    J_tempSequenceReg[6] <= J_tempSequenceReg[2] ;

                    J_tempSequenceReg[3] <= J_tempSequenceReg[5] ;
                    J_tempSequenceReg[5] <= J_tempSequenceReg[3] ;
                end
                'd1:
                begin
                    J_tempSequenceReg[2] <= J_tempSequenceReg[7] ;
                    J_tempSequenceReg[7] <= J_tempSequenceReg[2] ;

                    J_tempSequenceReg[3] <= J_tempSequenceReg[6] ;
                    J_tempSequenceReg[6] <= J_tempSequenceReg[3] ;

                    J_tempSequenceReg[4] <= J_tempSequenceReg[5] ;
                    J_tempSequenceReg[5] <= J_tempSequenceReg[4] ;
                end
                'd2:
                begin
                    J_tempSequenceReg[3] <= J_tempSequenceReg[7] ;
                    J_tempSequenceReg[7] <= J_tempSequenceReg[3] ;

                    J_tempSequenceReg[4] <= J_tempSequenceReg[6] ;
                    J_tempSequenceReg[6] <= J_tempSequenceReg[4] ;
                end
                'd3:
                begin
                    J_tempSequenceReg[4] <= J_tempSequenceReg[7] ;
                    J_tempSequenceReg[7] <= J_tempSequenceReg[4] ;

                    J_tempSequenceReg[5] <= J_tempSequenceReg[6] ;
                    J_tempSequenceReg[6] <= J_tempSequenceReg[5] ;
                end
                'd4:
                begin
                    J_tempSequenceReg[5] <= J_tempSequenceReg[7] ;
                    J_tempSequenceReg[7] <= J_tempSequenceReg[5] ;
                end
                'd5:
                begin
                    J_tempSequenceReg[6] <= J_tempSequenceReg[7] ;
                    J_tempSequenceReg[7] <= J_tempSequenceReg[6] ;
                end
                default:
                begin
                    J_tempSequenceReg[i] <= J_tempSequenceReg[i];
                end
            endcase
        end
        else
        begin
            J_tempSequenceReg[i] <= J_tempSequenceReg[i];
        end
    end
end


//localCostReg
wire[9:0] localCost = localCostReg_rd + Cost;
wire[9:0] localCostReg_wr  = MIN_COST_CalculationDone_flag ? 'd0 : localCost;

always @(posedge CLK )
begin
    localCostReg_rd <= RST ? 'd0 : localCostReg_wr;
end


//MinCost,MatchCount
wire localCost_eq_MinCost_flag = (localCost == MinCost);

wire[3:0] MatchCount_localCost_eq_action = (localCost_eq_MinCost_flag ? MatchCount + 'd1 :MatchCount) ;
wire[3:0] MatchCount_localCost_lt_action = (localCost_lt_MinCost_flag ? 'd1 :  MatchCount_localCost_eq_action) ;

wire localCost_lt_MinCost_flag = (localCost < MinCost);
wire[9:0] LocalCost_lt_action = ( localCost_lt_MinCost_flag ? localCost : MinCost) ;

always @(posedge CLK)
begin
    MinCost <= RST ? 'd1023 : (MIN_COST_CalculationDone_flag ? LocalCost_lt_action : MinCost);
end

always @(posedge CLK)
begin
    MatchCount <= RST ? 'd0 : (MIN_COST_CalculationDone_flag ? MatchCount_localCost_lt_action: MatchCount);
end

//Find pivot
reg[6:0] FindPivotBitstream;
always @(*)
begin
    for(i=0;i<7;i=i+1)
    begin
        FindPivotBitstream[i] = (J_SequenceReg[i] < J_SequenceReg[i+1]);
    end
end

always @(*)
begin
    casex(FindPivotBitstream)
        7'b1xxxxxx:
            pivot = 'd7;
        7'b01xxxxx:
            pivot = 'd6;
        7'b001xxxx:
            pivot = 'd5;
        7'b0001xxx:
            pivot = 'd4;
        7'b00001xx:
            pivot = 'd3;
        7'b000001x:
            pivot = 'd2;
        7'b0000001:
            pivot = 'd1;
        default:
            pivot = 'd0;
    endcase
end

//Switch registers.
reg[3:0] MinMaxComparedValue;
reg[3:0] MinMaxComparedIndex;
reg[3:0] tempValueReg;
reg[3:0] tempIndexReg;

wire head_gt_pivot_flag = headValue > pivotValue;
wire tail_gt_pivot_flag = tailValue > pivotValue;
wire head_lt_tail_flag  = headValue < tailValue ;
wire headOrTail_lt_temp_flag = MinMaxComparedValue < tempValueReg;

always @(*)
begin
    case({head_gt_pivot_flag,tail_gt_pivot_flag})
        2'b11:
        begin
            MinMaxComparedValue = head_lt_tail_flag ? headValue : tailValue;
            MinMaxComparedIndex = head_lt_tail_flag ? headPTR : tailPTR;
        end
        2'b10:
        begin
            MinMaxComparedValue = headValue;
            MinMaxComparedIndex = headPTR;
        end
        2'b01:
        begin
            MinMaxComparedValue = tailValue;
            MinMaxComparedIndex = tailPTR;
        end
        default:
        begin
            MinMaxComparedValue = pivotValue;
            MinMaxComparedIndex = pivot;
        end
    endcase
end

always @(posedge CLK)
begin
    if(RST || STATE_FIND_PIVOT_SET_TAILHEAD)
    begin
        tempValueReg <= 'd8;
    end
    else if(STATE_SWITCH_PIVOT)
    begin
        tempValueReg <=  headOrTail_lt_temp_flag ? MinMaxComparedValue : tempValueReg;
    end
    else
    begin
        tempValueReg <= tempValueReg;
    end
end

always @(posedge CLK)
begin
    if(RST || STATE_FIND_PIVOT_SET_TAILHEAD)
    begin
        tempIndexReg <= 'd8;
    end
    else if(STATE_SWITCH_PIVOT)
    begin
        tempIndexReg <=  headOrTail_lt_temp_flag ? MinMaxComparedIndex : tempIndexReg;
    end
    else
    begin
        tempIndexReg <= tempIndexReg;
    end
end


endmodule
