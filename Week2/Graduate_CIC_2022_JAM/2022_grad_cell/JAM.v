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

//State indicator
wire STATE_FIND_PIVOT               =     (  FIND_PIVOT         == currentState); //2 Cycles
wire STATE_SWITCH_PIVOT_MIN         =     (  SWITCH_PIVOT_MIN   == currentState); //4
wire STATE_SWITCH_PIVOT_MIN_WB      =     ( SWITCH_PIVOT_MIN_WB == currentState); //1
wire STATE_FLIP                     =     (  FLIP               == currentState); //1
//No need to write back, combine it into WB mode
wire STATE_WB                       =     (  WB                 == currentState); //1
wire STATE_DONE                     =     (  DONE               == currentState);

//CNTS
reg[3:0] workIndexCnt;
reg[2:0] localStatesCnt;
reg[15:0] permutationCnt;

//PTRS
reg[2:0] headPTR;
reg[2:0] tailPTR;

//Flags
wire findPivotDone_flag             = STATE_FIND_PIVOT ?  (localStatesCnt == 'd1) : 'd0;
wire switchPivotDone_flag           = STATE_SWITCH_PIVOT_MIN ? (localStatesCnt == 'd3) : 'd0 ;
wire allPermutationFound_flag       = ( permutationCnt == (MAX_COMBINATIONS-1) );
wire minCostCalculationDone_flag    = (workIndexCnt == 'd8);
wire sequenceTraversed_flag         = (headPTR == tailPTR) ||  (tailPTR == (headPTR+'d1)) ;
wire switchPivotSet_flag            = STATE_SWITCH_PIVOT_MIN && (localStatesCnt == 'd0);
wire pivotValueNotChanged           = (tempMinValueReg == 'd8);

//Registers
reg[2:0] jobSequenceReg[0:7];
reg[2:0] jobTempSequenceReg[0:7];
reg[9:0] minCostReg_o;
reg[3:0] matchCountReg_o;
reg[3:0] pivotIndexReg;
reg[3:0] tempMinValueReg;
reg[3:0] tempPivotIndexReg;


//CTR
always @(posedge CLK)
begin
    currentState <= RST ? FIND_PIVOT : nextState;
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
            nextState = minCostCalculationDone_flag ? FIND_PIVOT : WB;
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
    permutationCnt <= RST ? 'd0 : (minCostCalculationDone_flag ? permutationCnt + 'd1 : permutationCnt);
end

//PTRs
always @(posedge CLK)
begin
    if(RST || STATE_WB)
    begin
        headPTR <= 'd0;
        tailPTR <= 'd7;
    end
    else if(STATE_FIND_PIVOT)
    begin
        headPTR <= pivotIndexReg + 'd1;
        tailPTR <= 'd7;
    end
    else if(STATE_SWITCH_PIVOT_MIN)
    begin
        headPTR <= (sequenceTraversed_flag ? headPTR : headPTR + 'd1);
        tailPTR <= (sequenceTraversed_flag ? tailPTR : tailPTR - 'd1);
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

wire compareTailPTRValue_gt_pivot_flag = tailPTRValue > pivotValue;
wire compareHeadPTRValue_gt_pivot_flag = headPTRValue > pivotValue;

always @(posedge CLK)
begin
    for(i = 0 ; i < 8; i = i + 1)
    begin
        if(RST)
        begin
            jobTempSequenceReg[i] <= i;
        end
        else if(switchPivotDone_flag & !pivotValueNotChanged)
        begin
            jobTempSequenceReg[pivotIndexReg]     <= tempMinValueReg;
            jobTempSequenceReg[tempPivotIndexReg] <= pivotValue;
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
//localMincostReg_rd
reg[9:0] localMincostReg_wr,localMincostReg_rd;

always @(posedge CLK)
begin
    localMincostReg_rd <=  RST ? 'd0 : localMincostReg_wr;
end

always @(*)
begin
    if(RST)
    begin
        localMincostReg_wr = 'd0;
    end
    else if(minCostCalculationDone_flag)
    begin
        localMincostReg_wr = 'd0;
    end
    else
    begin
        localMincostReg_wr = localMincostReg_rd + Cost;
    end
end

wire startMinCostCalculation_flag = (workIndexCnt == 'd0);
//minCostReg_o
always @(posedge CLK)
begin
    if(RST)
    begin
        minCostReg_o <= 'd1023;
    end
    else if(minCostCalculationDone_flag)
    begin
        minCostReg_o <= localMinCost_lt_MinCost_flag ? localMincostReg_rd : minCostReg_o ;
    end
    else
    begin
        minCostReg_o <= minCostReg_o;
    end
end

//matchCountReg_o
wire MinCost_eq_localMinCost_flag = minCostCalculationDone_flag ? (minCostReg_o == localMincostReg_rd) : 'dz;
wire localMinCost_lt_MinCost_flag = minCostCalculationDone_flag ? (localMincostReg_rd < minCostReg_o) : 'dz;


always @(posedge CLK)
begin
    if(RST)
    begin
        matchCountReg_o <= 'd0;
    end
    else if(minCostCalculationDone_flag)
    begin
        matchCountReg_o <= MinCost_eq_localMinCost_flag ?  matchCountReg_o + 'd1 :
                        (localMinCost_lt_MinCost_flag ? 'd0 : matchCountReg_o);
    end
    else
    begin
        matchCountReg_o <= matchCountReg_o;
    end
end

reg[2:0] pivotIndex_wr;
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
assign J = minCostCalculationDone_flag ?  'd0 : jobSequenceReg[workIndexCnt];
assign MatchCount = matchCountReg_o;
assign MinCost    = minCostReg_o;
assign Valid = STATE_DONE;

//-------------------------------DP----------------------------//
//DP without considerations of comparators
//Find pivot

reg[6:0] findPivotLocationBitStream;
always @(*)
begin
    for(i=0;i<7 ; i = i + 1)
    begin
        findPivotLocationBitStream[i] = (jobSequenceReg[i] < jobSequenceReg[i+1]);
    end
end

always @(*)
begin
    casex(findPivotLocationBitStream)
        7'b1xxxxxx:
            pivotIndex_wr = 'd6;
        7'b01xxxxx:
            pivotIndex_wr = 'd5;
        7'b001xxxx:
            pivotIndex_wr = 'd4;
        7'b0001xxx:
            pivotIndex_wr = 'd3;
        7'b00001xx:
            pivotIndex_wr = 'd2;
        7'b000001x:
            pivotIndex_wr = 'd1;
        7'b0000001:
            pivotIndex_wr = 'd0;
        default:
        begin
            pivotIndex_wr = 'd0;
        end
    endcase
end



//tempMinValueReg,tempPivotIndexReg
always @(posedge CLK)
begin
    if(RST)
    begin
        tempMinValueReg <= 'd8;
        tempPivotIndexReg <= 'd8;
    end
    else if(STATE_FIND_PIVOT)
    begin
        tempMinValueReg <= 'd8;
        tempPivotIndexReg <= pivotIndexReg;
    end
    else if(STATE_SWITCH_PIVOT_MIN)
    begin
        tempMinValueReg <= tempMinValue_wr;
        tempPivotIndexReg <= tempIndex_wr;
    end
    else
    begin
        tempMinValueReg <= tempMinValueReg;
        tempPivotIndexReg <= tempPivotIndexReg;
    end
end

wire compareTailValue_lt_temp_flag = tailPTRValue < tempMinValueReg;
wire compareHeadValue_lt_temp_flag = headPTRValue < tempMinValueReg;
wire headPTRValue_lt_tailPTRValue_flag = headPTRValue < tailPTRValue;
reg[3:0] tempMinValue_wr,tempIndex_wr;

//tempMinValue_wr
always @(*)
begin
    case({compareTailPTRValue_gt_pivot_flag,compareHeadPTRValue_gt_pivot_flag})
        2'b11:
        begin
            if(headPTRValue_lt_tailPTRValue_flag)
            begin
                if(compareHeadValue_lt_temp_flag )
                begin
                    tempMinValue_wr = headPTRValue;
                    tempIndex_wr      = headPTR;
                end
                else
                begin
                    tempMinValue_wr = tempMinValueReg;
                    tempIndex_wr      = tempPivotIndexReg;
                end
            end
            else
            begin
                if(compareTailValue_lt_temp_flag)
                begin
                    tempMinValue_wr = tailPTRValue;
                    tempIndex_wr      = tailPTR;
                end
                else
                begin
                    tempMinValue_wr = tempMinValueReg;
                    tempIndex_wr      = tempPivotIndexReg;
                end
            end
        end
        2'b10:
        begin
            if(compareTailValue_lt_temp_flag)
            begin
                tempMinValue_wr   = tailPTRValue;
                tempIndex_wr      = tailPTR;
            end
            else
            begin
                tempMinValue_wr   = tempMinValueReg;
                tempIndex_wr      = tempPivotIndexReg;
            end
        end
        2'b01:
        begin
            if(compareHeadValue_lt_temp_flag)
            begin
                tempMinValue_wr = tailPTRValue;
                tempIndex_wr      = tailPTR;
            end
            else
            begin
                tempMinValue_wr = tempMinValueReg;
                tempIndex_wr      = tempPivotIndexReg;
            end
        end
        default:
        begin
            tempMinValue_wr = tempMinValueReg;
            tempIndex_wr      = tempPivotIndexReg;
        end
    endcase
end
endmodule
