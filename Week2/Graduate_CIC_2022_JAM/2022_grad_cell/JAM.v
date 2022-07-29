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
parameter FLIP = 'd2;
parameter WB  =  'd3;
parameter DONE = 'd4;

//State Register
reg[2:0] currentState,nextState;

//State indicator
wire STATE_FIND_PIVOT               =     (  FIND_PIVOT         == currentState); //2 Cycles with 4 comparators
wire STATE_SWITCH_PIVOT_MIN         =     (  SWITCH_PIVOT_MIN   == currentState); //4
wire STATE_FLIP                     =     (  FLIP               == currentState); //1
wire STATE_WB                       =     (  WB                 == currentState); //1
wire STATE_DONE                     =     (  DONE               == currentState);

//CNTS
reg[2:0] workIndexCnt;
reg[1:0] localStatesCnt;
reg[14:0] permutationCnt;

//Flags
wire findPivotDone_flag   =   STATE_FIND_PIVOT ?       (localStatesCnt == 'd1) : 'dz;
wire switchPivotDone_flag =   STATE_SWITCH_PIVOT_MIN ? (localStatesCnt == 'd4) : 'dz;
wire allPermutationFound_flag = ( permutationCnt == (MAX_COMBINATIONS-1) );
wire minCostCalculationDone_flag = (workIndexCnt == 'd7);
wire sequenceTraversed_flag = STATE_SWITCH_PIVOT_MIN ? (headPTR == tailPTR) ||  (tailPTR == (headPTR+'d1)) : 'dz;
wire switchPivotSet_flag = STATE_SWITCH_PIVOT_MIN && (localStatesCnt == 'd0);


//PTRS
reg[2:0] headPTR;
reg[2:0] tailPTR;

//Registers
reg[2:0][0:7] jobSequenceReg;
reg[2:0][0:7] jobTempSequenceReg;
reg[9:0] minCostReg_o;
reg[3:0] matchCountReg_o;
reg[9:0] localMincostReg;
reg[3:0] pivotIndexReg;


//CTR
always @(posedge CLK)
begin
    currentState <= RST ? STATE_FIND_PIVOT : nextState;
end

always @(*)
begin
    case(currentState)
        FIND_PIVOT:
            nextState = allPermutationFound_flag ? DONE : (findPivotDone_flag ? SWITCH_PIVOT_MIN : FIND_PIVOT);
        SWITCH_PIVOT_MIN:
            nextState = switchPivotDone_flag ? FLIP : SWITCH_PIVOT_MIN;
        FLIP:
            nextState = WB;
        WB:
            nextState = FIND_PIVOT;
        default:
            nextState = FIND_PIVOT;
    endcase
end

//workIndexCnt
always @(posedge CLK)
begin
    workIndexCnt <= RST ? 'd0 : (minCostCalculationDone_flag ? 'd0 : workIndexCnt + 'd1);
end

//localStatesCnt
always @(posedge CLK)
begin
    if(RST)
    begin
        localStatesCnt <= 'd0;
    end
    else if(findPivotDone_flag || switchPivotDone_flag)
    begin
        localStatesCnt <= 'd0;
    end
    else if(STATE_FIND_PIVOT || STATE_SWITCH_PIVOT_MIN)
    begin
        localStatesCnt <= localStatesCnt + 'd1;
    end
    else
    begin
        localStatesCnt <= localStatesCnt;
    end
end

//permutationCnt
always @(posedge CLK )
begin
    permutationCnt <= RST ? 'd0 : (WB ? permutationCnt + 'd1 : permutationCnt);
end

//PTRs
always @(posedge CLK)
begin
    if(RST)
    begin
        headPTR <= 'd0;
        tailPTR <= 'd0;
    end
    else if(STATE_SWITCH_PIVOT_MIN)
    begin
        headPTR <= switchPivotSet_flag ? pivotIndexReg + 'd1 : (sequenceTraversed_flag ? headPTR : headPTR + 'd1);
        tailPTR <= switchPivotSet_flag ? tailPTR : (sequenceTraversed_flag ? tailPTR : tailPTR - 'd1);
    end
    else
    begin
        headPTR <= headPTR;
        tailPTR <= tailPTR;
    end
end

//Registers
//job sequences
integer  i;
always @(posedge CLK)
begin
    for(i = 0 ; i < 8; i = i + 1)
    begin
        if(RST)
        begin
            jobSequenceReg[i] <= i;
        end
        else if(STATE_WB)
        begin
            jobSequenceReg[i] <= jobTempSequenceReg[i];
        end
        else
        begin
            jobSequenceReg[i] <= jobSequenceReg[i];
        end
    end
end

//jobTempSequenceReg
wire[2:0] pivotValue = jobTempSequenceReg[pivotIndexReg];
wire[2:0] headPTRValue = jobTempSequenceReg[headPTR];
wire[2:0] tailPTRValue = jobTempSequenceReg[tailPTR];

wire compareValue_gt_flag;
wire headPTRValue_gt_flag;

always @(posedge CLK)
begin
    for(i = 0 ; i < 8; i = i + 1)
    begin
        if(RST)
        begin
            jobTempSequenceReg[i] <= i;
        end
        else if(STATE_FIND_PIVOT)
        begin
            jobTempSequenceReg[pivotIndexReg] <= compareValue_gt_flag ? (headPTRValue_gt_flag ? headPTRValue: tailPTRValue) : pivotValue;
        end
        else if(STATE_FLIP)
        begin
            case(pivotIndexReg)
                'd0:
                begin
                    jobTempSequenceReg[1] <= jobTempSequenceReg[7] ;
                    jobTempSequenceReg[7] <= jobTempSequenceReg[1] ;

                    jobTempSequenceReg[2] <= jobTempSequenceReg[6] ;
                    jobTempSequenceReg[6] <= jobTempSequenceReg[2] ;

                    jobTempSequenceReg[3] <= jobTempSequenceReg[5] ;
                    jobTempSequenceReg[5] <= jobTempSequenceReg[3] ;
                end
                'd1:
                begin
                    jobTempSequenceReg[2] <= jobTempSequenceReg[7] ;
                    jobTempSequenceReg[7] <= jobTempSequenceReg[2] ;

                    jobTempSequenceReg[3] <= jobTempSequenceReg[6] ;
                    jobTempSequenceReg[6] <= jobTempSequenceReg[3] ;

                    jobTempSequenceReg[4] <= jobTempSequenceReg[5] ;
                    jobTempSequenceReg[5] <= jobTempSequenceReg[4] ;
                end
                'd2:
                begin
                    jobTempSequenceReg[3] <= jobTempSequenceReg[7] ;
                    jobTempSequenceReg[7] <= jobTempSequenceReg[3] ;

                    jobTempSequenceReg[4] <= jobTempSequenceReg[6] ;
                    jobTempSequenceReg[6] <= jobTempSequenceReg[4] ;
                end
                'd3:
                begin
                    jobTempSequenceReg[4] <= jobTempSequenceReg[7] ;
                    jobTempSequenceReg[7] <= jobTempSequenceReg[4] ;

                    jobTempSequenceReg[5] <= jobTempSequenceReg[6] ;
                    jobTempSequenceReg[6] <= jobTempSequenceReg[5] ;
                end
                'd4:
                begin
                    jobTempSequenceReg[5] <= jobTempSequenceReg[7] ;
                    jobTempSequenceReg[7] <= jobTempSequenceReg[5] ;
                end
                'd5:
                begin
                    jobTempSequenceReg[6] <= jobTempSequenceReg[7] ;
                    jobTempSequenceReg[7] <= jobTempSequenceReg[6] ;
                end
                default:
                begin
                    jobTempSequenceReg[i] <= jobTempSequenceReg[i];
                end
            endcase
        end
        else
        begin
            jobTempSequenceReg[i] <= jobTempSequenceReg[i];
        end
    end
end
//localMincostReg
always @(posedge CLK)
begin
    if(RST)
    begin
        localMincostReg <= 'd0;
    end
    else if(minCostCalculationDone_flag)
    begin
        localMincostReg <= 'd0;
    end
    else
    begin
        localMincostReg <= localMincostReg + Cost;
    end
end

//minCostReg_o
always @(posedge CLK)
begin
    if(RST)
    begin
        minCostReg_o <= 'd1023;
    end
    else if(minCostCalculationDone_flag)
    begin
        minCostReg_o <= 'd1023;
    end
    else
    begin
        minCostReg_o <= minCostReg_o + jobSequenceReg[workIndexCnt];
    end
end

//matchCountReg_o
wire newMinCost_eq_localMinCost_flag = minCostCalculationDone_flag ? (minCostReg_o == localMincostReg) : 'dz;
wire newMinCost_gt_localMinCost_flag = minCostCalculationDone_flag ? (minCostReg_o < localMincostReg) :'dz ;
always @(posedge CLK)
begin
    if(RST)
    begin
        matchCountReg_o <= 'd0;
    end
    else if(minCostCalculationDone_flag)
    begin
        matchCountReg_o <=newMinCost_eq_localMinCost_flag ?  matchCountReg_o + 'd1 :
                        (newMinCost_gt_localMinCost_flag ? 'd0 : matchCountReg_o);
    end
    else
    begin
        matchCountReg_o <= matchCountReg_o;
    end
end

//PivotIndexReg
always @(posedge CLK)
begin
    if(RST)
    begin
        pivotIndexReg <= 'd0;
    end
    else if(STATE_WB)
    begin
        pivotIndexReg <= pivotIndexReg;
    end
    else if(STATE_FIND_PIVOT)
    begin
        pivotIndexReg <= pivotIndex_wr;
    end
    else
    begin
        pivotIndexReg <= pivotIndexReg;
    end
end
//I/O
assign W = workIndexCnt;
assign J = jobSequenceReg[workIndexCnt];
assign MatchCount = matchCountReg_o;
assign MinCost    = minCostReg_o;
assign Valid = STATE_DONE;

//-------------------------------DP----------------------------//
//DP without considerations of comparators
//Find pivot
reg[2:0] pivotIndex_wr;
reg[6:0] findPivotLocationBitStream;
always @(*)
begin
    for(i=0;i<=7 ; i = i + 1)
    begin
        findPivotLocationBitStream[i] = (jobSequenceReg[i] < jobSequenceReg[i+1]);
    end
end

always @(*)
begin
    case(findPivotLocationBitStream)
    7'b1xxxxxx: pivotIndex_wr = 'd6;
    7'b01xxxxx: pivotIndex_wr = 'd5;
    7'b001xxxx: pivotIndex_wr = 'd4;
    7'b0001xxx: pivotIndex_wr = 'd3;
    7'b00001xx: pivotIndex_wr = 'd2;
    7'b000001x: pivotIndex_wr = 'd1;
    7'b0000001: pivotIndex_wr = 'd0;
    default:
    begin
        pivotIndex_wr = 'd0;
    end
    endcase
end





endmodule
