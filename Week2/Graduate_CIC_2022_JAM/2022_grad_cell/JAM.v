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
parameter DONE = 'd3;

//State Register
reg[2:0] currentState,nextState;

//State indicator
wire STATE_FIND_PIVOT               =     (  FIND_PIVOT         == currentState);
wire STATE_SWITCH_PIVOT_MIN         =     (  SWITCH_PIVOT_MIN   == currentState);
wire STATE_FLIP                     =     (  FLIP               == currentState);
wire STATE_DONE                     =     (  DONE               == currentState);

//CNTS
reg[2:0] workIndexCnt;
reg[1:0] localStatesCnt;
reg[14:0] permutationCnt;

//Flags
wire findPivotDone_flag   =   STATE_FIND_PIVOT ?       (localStatesCnt == 'd2) : 'dz;
wire switchPivotDone_flag =   STATE_SWITCH_PIVOT_MIN ? (localStatesCnt == 'd3) : 'dz;
wire allPermutationFound_flag = ( permutationCnt == (MAX_COMBINATIONS-1) );
wire minCostCalculationDone_flag = (workIndexCnt == 'd7);
wire sequenceTraversed_flag = STATE_SWITCH_PIVOT_MIN ? (headPTR == tailPTR) ||  (tailPTR == (headPTR+'d1)) : 'dz;

reg pivotSwitched_flag;


//PTRS
reg[2:0] headPTR;
reg[2:0] tailPTR;

//Registers
reg[2:0][0:7] jobSequenceReg;
reg[2:0][0:7] jobTempSequenceReg;
reg[9:0] minCostReg;
reg[3:0] matchCountReg;
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
    permutationCnt <= RST ? 'd0 : (STATE_FLIP ? permutationCnt + 'd1 : permutationCnt);
end

//PTRs
always @(posedge CLK)
begin
    if(RST || STATE_FIND_PIVOT)
    begin
        headPTR <= pivotIndexReg + 'd1;
        tailPTR <= 'd7;
    end
    else if(STATE_SWITCH_PIVOT_MIN)
    begin
        headPTR <= sequenceTraversed_flag ? headPTR : headPTR + 'd1;
        tailPTR <= 'd0;
    end
    else
    begin
        headPTR <= 'd0;
        tailPTR <= 'd0;
    end
end

always @(posedge CLK)
begin

end






endmodule
