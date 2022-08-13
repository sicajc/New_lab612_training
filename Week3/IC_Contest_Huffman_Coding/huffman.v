module huffman(clk, reset, gray_valid, gray_data, CNT_valid, CNT1, CNT2, CNT3, CNT4, CNT5, CNT6,
               code_valid, HC1, HC2, HC3, HC4, HC5, HC6, M1, M2, M3, M4, M5, M6);

input clk;
input reset;
input gray_valid;
input [7:0] gray_data;
output CNT_valid;
output [7:0] CNT1, CNT2, CNT3, CNT4, CNT5, CNT6;
output code_valid;
output [7:0] HC1, HC2, HC3, HC4, HC5, HC6;
output [7:0] M1, M2, M3, M4, M5, M6;

reg [7:0] CNT1, CNT2, CNT3, CNT4, CNT5, CNT6;
reg CNT_valid;

reg code_valid;
reg [7:0] HC1, HC2, HC3, HC4, HC5, HC6;
reg [7:0] M1, M2, M3, M4, M5, M6;


reg[2:0] currentState;
reg[2:0] nextState;

reg[9:0] frquencyTableReg[0:10];
reg[4:0] orderTableReg[0:5];
reg[9:0] sortedResult[10:0];

reg[4:0] tailPTR;
reg[4:0] mergedNode;

//States
parameter IDLE = 0;
parameter RD_SYMBOL = 1;
parameter SORT  = 2;
parameter MERGE = 3;
parameter SPLIT_ENCODE = 4;
parameter DONE = 4;

wire STATE_IDLE = IDLE == currentState;
wire STATE_RD_SYMBOL = RD_SYMBOL == currentState;
wire STATE_SORT = SORT == currentState;
wire STATE_MERGE = MERGE == currentState;
wire STATE_SPLIT_ENCODE = SPLIT_ENCODE == currentState;
wire STATE_DONE = DONE == currentState;
//0.Create Frequency table while reading data, FrequencyTable Consist of
//Frequency of A1,A2,A3,A4,A5,A6,N0,N1,N2,N3,N4
integer idx;
always @(posedge clk or posedge reset)
begin
    for(idx= 0  ; idx < 11 ; idx = idx + 1)
    begin
        if(reset)
        begin
            orderTableReg[idx] <= idx+1 ;
        end
        else if(STATE_SORT)
        begin
            orderTableReg[idx] <= sortedResult[idx];
        end
        else if(STATE_MERGE)
        begin
            orderTableReg[tailPTR] <= mergedNode;
            orderTableReg[nextElementPTR] <= 'd0;
        end
        else
        begin
            orderTableReg[idx] <= orderTableReg[idx];
        end
    end
end




//0.Create an OrderTable to store sorted result, update this table everytime after sorting, with highest being at index 0, Unused
//0.Initialise OrderTable with A1,A2,A3,A4,A5,A6
wire[4:0] nextElementPTR = tailPTR + 'd1;

always @(posedge clk or posedge reset)
begin
    for(idx= 0  ; idx < 6 ; idx = idx + 1)
    begin
        if(reset)
        begin
            orderTableReg[idx] <= idx+1 ;
        end
        else if(STATE_SORT)
        begin
            orderTableReg[idx] <= sortedResult[idx];
        end
        else if(STATE_MERGE)
        begin
            orderTableReg[tailPTR] <= mergedNode;
            orderTableReg[nextElementPTR] <= 'd0;
        end
        else
        begin
            orderTableReg[idx] <= orderTableReg[idx];
        end
    end
end

//1.0 Sort A1,A2,A3,A4,A5,A6 According to frequency
//Slot assign 0 in freuqnecy
//6 values sorter using Bitonic sorter

// two_stages_8_bs #(.N(10))
// sorter(.rst_n(reset),.clk(clk),.a(orderTableReg[]),.b(orderTableReg[]),.c(orderTableReg[]),.d(orderTableReg[]),.e(orderTableReg[]),.f(orderTableReg[]),.g(0),.h(0),
// .idx(orderTableSortedResult[0]),.j(orderTableSortedResult[1]),.k(orderTableSortedResult[2]),.l(orderTableSortedResult[3]),.m(orderTableSortedResult[4]),.n(orderTableSortedResult[5]),.o(orderTableSortedResult[6]),.p(orderTableSortedResult[]));


//1.1Merging the lowest 2 key, removes them from sortTable then insert the new key
//At the same time, add left child(lower one) and right child(larger one) to index 0 of huffman tree
//Repeat the process 1.0 ~ 1.1 until huffman tree is full

//2.0 Splitting and assign.
//Initialise Encoded Register(ER) and SubEncode Register(SER) also Mask(M)
//Determine if the left child and right child are nodes, if both nodes, store the right child into stack for later traversal,Since we traverse from left
//child first
//Start the traversal from left child to right child. If the traversd node is symbol, output the symbol's code from SER AND ER
//If the it is another node,later shift the cnt to the node after the traversal of this level is done.
//Done if we encoded all 6 symbols.


























































































































































































































































































































































































































































































































































































endmodule
