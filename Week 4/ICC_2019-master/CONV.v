`timescale 1ns/10ps

module  CONV(
            clk,
            reset,
            busy,
            ready,
            iaddr,
            idata,
            cwr,
            caddr_wr,
            cdata_wr,
            crd,
            caddr_rd,
            cdata_rd,
            csel
        );

input						clk;
input						reset         ;
output						busy          ;
input						ready         ;
output [11:0]			iaddr;
input	[19:0]				idata         ;
output	 reg				cwr           ;
output	 reg[11:0]  		caddr_wr          ;
output reg signed[19:0] 	cdata_wr          ;
output	 reg				crd           ;
output	reg [11:0]			caddr_rd      ;
input	 [19:0]				cdata_rd      ;
output	 reg [2:0]			csel          ;

//Flatten WB needs to be added

//Control path
reg[3:0] mainCTR_current_state,mainCTR_next_state;
reg[10:0] sharedCnt;

reg[6:0] imgColPTR_r;
reg[6:0] imgRowPTR_r;

//CONSTANTS
parameter KERNAL_WIDTH = 3;
parameter KERNAL_MID   = 1;
parameter IMAGE_WIDTH_HEIGHT = 64;
parameter PIXELS_OF_KERNAL = 'd8;
parameter MP_KERNAL_SIZE = 'd3;
parameter IMG_RD_DONE = 4096;
parameter DATA_WIDTH = 20;
parameter ADDR_WIDTH = 12;
parameter BIAS = 20'hF7295;

//STATES
parameter L0_ZEROPAD_CONV = 'd0;
parameter L0_K0_BIAS_RELU = 'd1;
parameter L0_K1_BIAS_RELU = 'd2;
parameter L0_K0_WB = 'd8;
parameter L0_K1_WB = 'd9;

parameter L1_K0_MAXPOOLING_COMPARE_MAX = 'd3;
parameter L1_K0_MAXPOOLING_WB = 'd4;
parameter L1_K1_MAXPOOLING_COMPARE_MAX = 'd5;
parameter L1_K1_MAXPOOLING_WB = 'd6;

parameter L2_K0_RD = 'd12;
parameter L2_K1_RD = 'd14;
parameter L2_K0_WB = 'd11;
parameter L2_K1_WB = 'd13;

parameter DONE = 'd7;

//KERNAL0 VALUES
parameter KERNAL0_00 = 20'h0a89e;
parameter KERNAL0_01 = 20'h092d5;
parameter KERNAL0_02 = 20'h06d43;
parameter KERNAL0_10 = 20'h01004;
parameter KERNAL0_11 = 20'hf8f71;
parameter KERNAL0_12 = 20'hf6e54;
parameter KERNAL0_20 = 20'hfa6d7;
parameter KERNAL0_21 = 20'hfc834;
parameter KERNAL0_22 = 20'hfac19;

//KERNAL1 VALUES
parameter KERNAL1_00 = 20'hfd855;
parameter KERNAL1_01 = 20'h02992;
parameter KERNAL1_02 = 20'hfc994;
parameter KERNAL1_10 = 20'h050fd;
parameter KERNAL1_11 = 20'h02f20;
parameter KERNAL1_12 = 20'h0202d;
parameter KERNAL1_20 = 20'h03bd7;
parameter KERNAL1_21 = 20'hfd369;
parameter KERNAL1_22 = 20'h05e68;

//STATES INDICATORS
wire STATE_L0_ZEROPAD_CONV                  =  mainCTR_current_state == L0_ZEROPAD_CONV;
wire STATE_L0_K0_BIAS_RELU                  =  mainCTR_current_state == L0_K0_BIAS_RELU;
wire STATE_L0_K1_BIAS_RELU                  =  mainCTR_current_state == L0_K1_BIAS_RELU;
wire STATE_L0_K0_WB                         =  mainCTR_current_state == L0_K0_WB;
wire STATE_L0_K1_WB                         =   mainCTR_current_state == L0_K1_WB;

wire STATE_L1_K0_MAXPOOLING_COMPARE_MAX    = mainCTR_current_state == L1_K0_MAXPOOLING_COMPARE_MAX;
wire STATE_L1_K1_MAXPOOLING_COMPARE_MAX    = mainCTR_current_state == L1_K1_MAXPOOLING_COMPARE_MAX;
wire STATE_L1_K0_MAXPOOLING_WB             =  mainCTR_current_state == L1_K0_MAXPOOLING_WB;
wire STATE_L1_K1_MAXPOOLING_WB             =  mainCTR_current_state == L1_K1_MAXPOOLING_WB;

wire STATE_L2_K0_RD                         = mainCTR_current_state == L2_K0_RD;
wire STATE_L2_K1_RD                         = mainCTR_current_state == L2_K1_RD;
wire STATE_L2_K0_WB                         = mainCTR_current_state == L2_K0_WB;
wire STATE_L2_K1_WB                         = mainCTR_current_state == L2_K1_WB;

wire STATE_DONE                             =  mainCTR_current_state == DONE;


//FLAGS
wire L0_LocalZeroPadConv_DoneFlag = STATE_L0_ZEROPAD_CONV ? (sharedCnt == PIXELS_OF_KERNAL) : 'd0;
wire L0_DoneFlag = ((imgRowPTR_r == IMAGE_WIDTH_HEIGHT) && (imgColPTR_r == IMAGE_WIDTH_HEIGHT)) ;
wire L0_imgRightBoundReach_Flag = STATE_L0_K1_BIAS_RELU ? (imgRowPTR_r == IMAGE_WIDTH_HEIGHT) : 'd0;

wire L1_K0_MaxPoolingCompare_DoneFlag = STATE_L1_K0_MAXPOOLING_COMPARE_MAX ? (sharedCnt == MP_KERNAL_SIZE) :'d0;
wire L1_K1_MaxPoolingCompare_DoneFlag = STATE_L1_K1_MAXPOOLING_COMPARE_MAX ? (sharedCnt == MP_KERNAL_SIZE) :'d0;
wire L1_imgRightBoundReach_Flag = STATE_L1_K1_MAXPOOLING_WB ? (imgRowPTR_r == IMAGE_WIDTH_HEIGHT - 1) :'d0;
wire L1_DoneFlag = ((imgRowPTR_r == IMAGE_WIDTH_HEIGHT-1) && (imgColPTR_r == IMAGE_WIDTH_HEIGHT - 1));

//Needs 1 cnt only uses sharedCnt
wire L2_K0_flatten_DoneFlag = (sharedCnt == IMG_RD_DONE);
wire L2_flatten_DoneFlag = (sharedCnt == IMG_RD_DONE);


//MAIN_CTR
always @(posedge clk or posedge reset)
begin
    mainCTR_current_state <= reset ? L0_ZEROPAD_CONV : mainCTR_next_state;
end

always @(*)
begin
    case(mainCTR_current_state)
        L0_ZEROPAD_CONV:
        begin
            mainCTR_next_state = L0_LocalZeroPadConv_DoneFlag ? L0_K0_BIAS_RELU : L0_ZEROPAD_CONV;
        end
        L0_K0_BIAS_RELU:
        begin
            mainCTR_next_state = L0_K0_WB;
        end
        L0_K0_WB:
        begin
            mainCTR_next_state = L0_K1_BIAS_RELU;
        end
        L0_K1_BIAS_RELU:
        begin
            mainCTR_next_state = L0_K1_WB;
        end
        L0_K1_WB:
        begin
            mainCTR_next_state = L0_DoneFlag ? L1_K0_MAXPOOLING_COMPARE_MAX : L0_ZEROPAD_CONV;
        end
        L1_K0_MAXPOOLING_COMPARE_MAX:
        begin
            mainCTR_next_state = L1_K0_MaxPoolingCompare_DoneFlag ? L1_K0_MAXPOOLING_WB : L1_K0_MAXPOOLING_COMPARE_MAX;
        end
        L1_K0_MAXPOOLING_WB:
        begin
            mainCTR_next_state = L1_K1_MAXPOOLING_COMPARE_MAX;
        end
        L1_K1_MAXPOOLING_COMPARE_MAX:
        begin
            mainCTR_next_state = L1_K1_MaxPoolingCompare_DoneFlag ? L1_K0_MAXPOOLING_WB : L1_K1_MAXPOOLING_COMPARE_MAX;
        end
        L1_K1_MAXPOOLING_WB:
        begin
            mainCTR_next_state = L1_DoneFlag ? L2_K0_RD : L1_K0_MAXPOOLING_COMPARE_MAX;
        end
        L2_K0_RD:
        begin
            mainCTR_current_state = L2_K0_WB;
        end
        L2_K0_WB:
        begin
            mainCTR_current_state = L2_K0_flatten_DoneFlag ? L2_K1_RD : L2_K0_RD;
        end
        L2_K1_RD:
        begin
            mainCTR_current_state = L2_K1_WB;
        end
        L2_K1_WB:
        begin
            mainCTR_current_state = L2_flatten_DoneFlag ? DONE : L2_K1_RD;
        end
        DONE:
        begin
            mainCTR_next_state = DONE;
        end
        default:
        begin
            mainCTR_next_state = L0_ZEROPAD_CONV;
        end
    endcase
end

//Shared counter
always @(posedge clk or posedge reset)
begin
    if(reset)
    begin
        sharedCnt <= reset ? 'd0 : sharedCnt_w;
    end
end

wire sharedCnt_w = (L0_LocalZeroPadConv_DoneFlag
                    || L1_K0_MaxPoolingCompare_DoneFlag || L1_K1_MaxPoolingCompare_DoneFlag || L2_K0_flatten_DoneFlag ||
                    L2_flatten_DoneFlag) ? 'd0 : (sharedCnt + 'd1);

//IMG_PTR
reg[6:0] imgColPTR_w;
reg[6:0] imgRowPTR_w;

always @(posedge clk or posedge reset)
begin
    imgColPTR_r <= reset || L1_DoneFlag ? 'd1 : L0_DoneFlag ?  'd0 : imgColPTR_w;
    imgRowPTR_r <= reset || L1_DoneFlag ? 'd1 : L0_DoneFlag ?  'd0 : imgRowPTR_w;
end

always @(*)
begin
    case(mainCTR_current_state)
        L0_K1_BIAS_RELU:
        begin
            imgColPTR_w  = L0_imgRightBoundReach_Flag ? 'd1 : imgColPTR_r + 'd1;
            imgRowPTR_w  = L0_imgRightBoundReach_Flag ? (imgRowPTR_r + 'd1) : imgRowPTR_r;
        end
        L1_K1_MAXPOOLING_WB:
        begin
            imgColPTR_w  = L1_imgRightBoundReach_Flag ? 'd0 : imgColPTR_r + 'd1;
            imgRowPTR_w  = L1_imgRightBoundReach_Flag ? (imgRowPTR_r + 'd1) : imgRowPTR_r;
        end
        default:
        begin
            imgColPTR_w  = imgColPTR_r;
            imgRowPTR_w  = imgRowPTR_r;
        end
    endcase
end

reg[11:0] STATE_L0_K0_BIAS_RELU ? conv_K0Result_r : conv_K1_Result_r;

wire[11:0] addrZeroPad =  (imgColPTR_r - 'd1) + (imgRowPTR_r - 'd1) * IMAGE_WIDTH_HEIGHT;
wire[11:0] addrMaxPooling = imgColPTR_r + imgRowPTR_r * IMAGE_WIDTH_HEIGHT;

parameter NO_ACCESS= 3'b000;
parameter L0_MEM0_ACCESS= 3'b001;
parameter L0_MEM1_ACCESS= 3'b010;
parameter L1_MEM0_ACCESS= 3'b011;
parameter L1_MEM1_ACCESS= 3'b100;
parameter L2_MEM_ACCESS= 3'b101;

reg[19:0] maxPooling_result_r;
reg[19:0] flattenTemp_r;
reg[19:0] flattenWriteAddr;

//Read write bus control
always @(*)
begin
    case(mainCTR_current_state)
        L0_K0_WB,L0_K1_WB:
        begin
            cdata_wr = STATE_L0_K0_BIAS_RELU ? conv_K0Result_r : conv_K1_Result_r;
            caddr_wr = addrZeroPad;
        end
        L1_K1_MAXPOOLING_WB,L1_K0_MAXPOOLING_WB:
        begin
            cdata_wr = maxPooling_result_r;
            caddr_wr = addrMaxPooling;
        end
        L2_K0_WB,L2_K1_WB:
        begin
            cdata_wr = flattenTemp_r;
            caddr_wr = flattenWriteAddr;
        end
        default:
        begin
            cdata_wr = 'd0;
            caddr_wr = 'd0;
        end
    endcase
end

//Memory access csel,crd,cwr
always @(*)
begin
    case(mainCTR_current_state)
        L0_K0_WB,L1_K0_MAXPOOLING_COMPARE_MAX:
        begin
            csel = L0_MEM0_ACCESS;
        end
        L0_K1_WB,L1_K1_MAXPOOLING_COMPARE_MAX:
        begin
            csel = L0_MEM1_ACCESS;
        end
        L1_K0_MAXPOOLING_WB:
        begin
            csel = L1_MEM0_ACCESS;
        end
        L1_K1_MAXPOOLING_WB:
        begin
            csel = L1_MEM1_ACCESS;
        end
        L2_K0_WB,L2_K1_WB:
        begin
            csel = L2_MEM_ACCESS;
        end
        default:
        begin
            csel = NO_ACCESS;
        end
    endcase
end

always @(*)
begin
    case(mainCTR_current_state)
        L0_K0_WB,L0_K1_WB,L1_K0_MAXPOOLING_WB,L1_K1_MAXPOOLING_WB,L2_K0_WB,L2_K1_WB:
        begin
            {crd,cwr} = 2'b01;
        end
        L1_K0_MAXPOOLING_COMPARE_MAX,L1_K1_MAXPOOLING_COMPARE_MAX,L2_K0_RD,L2_K1_RD:
        begin
            {crd,cwr} = 2'b10;
        end
        default:
        begin
            {crd,cwr} = 2'b00;
        end
    endcase
end

//Busy
assign busy = ((~STATE_DONE) ^ (~reset) ) ? 1 : 0 ;

//DATAPATH
reg signed[DATA_WIDTH-1:0] sharedReg1_r;
reg signed[DATA_WIDTH-1:0] sharedReg2_r;
reg signed[DATA_WIDTH-1:0] sharedReg1_wr;
reg signed[DATA_WIDTH-1:0] sharedReg2_wr;

wire signed[DATA_WIDTH-1:0] K0_convResult;
wire signed[DATA_WIDTH-1:0] K1_convResult;

//2 Multipliers for ZeroPadConv, uses input2 as kernal value input
wire signed[DATA_WIDTH*2 - 1 : 0] serialMultiplier1_o = multiplier1_Input1 * multiplier2_Input2;
wire signed[DATA_WIDTH*2 - 1 : 0] serialMultiplier2_o = multiplier2_Input1 * multiplier2_Input2;

wire signed[DATA_WIDTH - 1 : 0]multiplier1_Input1;
reg signed[DATA_WIDTH - 1 : 0]multiplier1_Input2;
wire signed[DATA_WIDTH - 1 : 0]multiplier2_Input1;
reg signed[DATA_WIDTH - 1 : 0]multiplier2_Input2;

//1 signed adder for BIAS_ReLU
wire signed[DATA_WIDTH-1:0] signedAdder_o = signedAdder_Input1 + BIAS;
wire signed[DATA_WIDTH-1:0] signedAdder_Input1;

assign signedAdder_Input1 = STATE_L0_K0_BIAS_RELU ? conv_K0Result_r : conv_K1_Result_r;

//1 Comparator for BIAS_ReLU & Maxpooling
wire signed compare_gt = comparatorInput1 > comparatorInput2 ;
reg[DATA_WIDTH-1:0] comparatorInput1;
reg[DATA_WIDTH-1:0] comparatorInput2;

wire compare_gt_zero = STATE_L0_K0_BIAS_RELU || STATE_L0_K1_BIAS_RELU ? 'd0 : compare_gt;
wire compare_gt_tempMax = STATE_L1_K0_MAXPOOLING_COMPARE_MAX || STATE_L1_K1_MAXPOOLING_COMPARE_MAX ? 'd0 : compare_gt;

//sharedReg1_r,sharedReg2_r
always @(posedge clk or posedge reset)
begin
    sharedReg1_r <= reset ? 'd0 : sharedReg1_wr;
    sharedReg2_r <= reset ? 'd0 : sharedReg2_wr;
end

//sharedReg1_wr
always @(*)
begin
    case(mainCTR_current_state)
        L0_ZEROPAD_CONV:
            //Serial multiplier
        begin
            sharedReg1_wr = serialMultiplier1_o + sharedReg1_r;
        end
        L0_K0_BIAS_RELU:
        begin
            sharedReg1_wr = compare_gt_zero ? signedAdder_o : 'd0;
        end
        default:
        begin
            sharedReg1_wr = sharedReg1_r;
        end
    endcase
end

//sharedReg2_wr
always @(*)
begin
    case(mainCTR_current_state)
        L0_ZEROPAD_CONV:
        begin
            sharedReg2_wr = serialMultiplier2_o + sharedReg2_r;
        end
        default:
        begin
            sharedReg2_wr = sharedReg2_r;
        end
    endcase
end

//Multipler2 Inputs
always @(*)
begin
    if(STATE_L0_ZEROPAD_CONV)
    begin
        case(sharedCnt)
            'd0:
            begin
                multiplier1_Input2 = KERNAL0_00;
                multiplier2_Input2 = KERNAL1_00;
            end
            'd1:
            begin
                multiplier1_Input2 = KERNAL0_01;
                multiplier2_Input2 = KERNAL1_01;
            end
            'd2:
            begin
                multiplier1_Input2 = KERNAL0_02;
                multiplier2_Input2 = KERNAL1_02;
            end
            'd3:
            begin
                multiplier1_Input2 = KERNAL0_10;
                multiplier2_Input2 = KERNAL1_10;
            end
            'd4:
            begin
                multiplier1_Input2 = KERNAL0_11;
                multiplier2_Input2 = KERNAL1_11;
            end
            'd5:
            begin
                multiplier1_Input2 = KERNAL0_12;
                multiplier2_Input2 = KERNAL1_12;
            end
            'd6:
            begin
                multiplier1_Input2 = KERNAL0_20;
                multiplier2_Input2 = KERNAL1_20;
            end
            'd7:
            begin
                multiplier1_Input2 = KERNAL0_21;
                multiplier2_Input2 = KERNAL1_21;
            end
            'd8:
            begin
                multiplier1_Input2 = KERNAL0_22;
                multiplier2_Input2 = KERNAL1_22;
            end
            default:
            begin
                multiplier1_Input2 = 'd10;
                multiplier2_Input2 = 'd10;
            end
        endcase
    end
    else
    begin
        multiplier1_Input2 = 'd0;
        multiplier2_Input2 = 'd0;
    end
end

assign multiplier1_Input1 = STATE_L0_ZEROPAD_CONV ? idata : 'd0;
assign multiplier2_Input1 = STATE_L0_ZEROPAD_CONV ? idata : 'd0;

parameter ZERO = 0;
//comparator Inputs
always @(*)
begin
    case(mainCTR_current_state)
        L0_K0_BIAS_RELU,L0_K1_BIAS_RELU:
        begin
            comparatorInput1 = STATE_L0_K0_BIAS_RELU ? conv_K0Result_r : conv_K1_Result_r;
            comparatorInput2 = ZERO;
        end
        L1_K0_MAXPOOLING_COMPARE_MAX,L1_K1_MAXPOOLING_COMPARE_MAX:
        begin
            comparatorInput1 = cdata_rd;
            comparatorInput2 = tempMax_r;
        end
        default:
        begin
            comparatorInput1 = 1;
            comparatorInput2 = 0;
        end
    endcase
end

//Convolution results
assign K0_convResult   =  STATE_L0_ZEROPAD_CONV || STATE_L0_K0_BIAS_RELU ? sharedReg1_r :'d0;
assign K1_convResult   =  STATE_L0_ZEROPAD_CONV || STATE_L0_K1_BIAS_RELU ? sharedReg2_r :'d0;


endmodule
