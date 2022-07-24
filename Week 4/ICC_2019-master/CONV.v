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
output	reg					busy          ;
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


//Control path
reg[3:0] mainCTR_current_state,mainCTR_next_state;
reg[10:0] sharedCnt;

reg[PTR_LENGTH-1:0] imgColPTR_rd;
reg[PTR_LENGTH-1:0] imgRowPTR_rd;

//CONSTANTS
parameter KERNAL_WIDTH = 3;
parameter KERNAL_MID   = 1;
parameter IMAGE_WIDTH_HEIGHT = 64;
parameter PIXELS_OF_KERNAL = 'd8;
parameter MP_KERNAL_SIZE = 'd3;
parameter MP_IMG_RD_DONE = 1023;
parameter DATA_WIDTH = 20;
parameter ADDR_WIDTH = 12;
parameter BIAS = 20'hF7295;
parameter PTR_LENGTH = 7;

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
wire STATE_L0_K1_WB                         =  mainCTR_current_state == L0_K1_WB;

wire STATE_L1_K0_MAXPOOLING_COMPARE_MAX    =  mainCTR_current_state == L1_K0_MAXPOOLING_COMPARE_MAX;
wire STATE_L1_K1_MAXPOOLING_COMPARE_MAX    =  mainCTR_current_state == L1_K1_MAXPOOLING_COMPARE_MAX;
wire STATE_L1_K0_MAXPOOLING_WB             =  mainCTR_current_state == L1_K0_MAXPOOLING_WB;
wire STATE_L1_K1_MAXPOOLING_WB             =  mainCTR_current_state == L1_K1_MAXPOOLING_WB;

wire STATE_L2_K0_RD                         = mainCTR_current_state == L2_K0_RD;
wire STATE_L2_K1_RD                         = mainCTR_current_state == L2_K1_RD;
wire STATE_L2_K0_WB                         = mainCTR_current_state == L2_K0_WB;
wire STATE_L2_K1_WB                         = mainCTR_current_state == L2_K1_WB;

wire STATE_DONE                             = mainCTR_current_state == DONE;

//FLAGS
wire L0_LocalZeroPadConv_DoneFlag = STATE_L0_ZEROPAD_CONV ? (sharedCnt == PIXELS_OF_KERNAL) : 'd0;
wire L0_DoneFlag = ((imgRowPTR_rd == IMAGE_WIDTH_HEIGHT) && (imgColPTR_rd == IMAGE_WIDTH_HEIGHT)) ;
wire L0_imgRightBoundReach_Flag = STATE_L0_K1_BIAS_RELU ? (imgRowPTR_rd == IMAGE_WIDTH_HEIGHT) : 'd0;

wire L1_K0_MaxPoolingCompare_DoneFlag = STATE_L1_K0_MAXPOOLING_COMPARE_MAX ? (sharedCnt == MP_KERNAL_SIZE) :'d0;
wire L1_K1_MaxPoolingCompare_DoneFlag = STATE_L1_K1_MAXPOOLING_COMPARE_MAX ? (sharedCnt == MP_KERNAL_SIZE) :'d0;
wire L1_imgRightBoundReach_Flag = STATE_L1_K1_MAXPOOLING_WB ? (imgRowPTR_rd == IMAGE_WIDTH_HEIGHT - 1) :'d0;
wire L1_DoneFlag = ((imgRowPTR_rd == IMAGE_WIDTH_HEIGHT-1) && (imgColPTR_rd == IMAGE_WIDTH_HEIGHT - 1));

//Needs 1 cnt only uses sharedCnt
wire L2_K0_flatten_DoneFlag = (sharedCnt == MP_IMG_RD_DONE);
wire L2_K1_flatten_DoneFlag = (sharedCnt == MP_IMG_RD_DONE);


//----------------------------------CONTROL_PATH--------------------------------//
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
            mainCTR_current_state = L2_K1_flatten_DoneFlag ? DONE : L2_K1_RD;
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
        sharedCnt <= 'd0;
    end
    else
    begin
        case(mainCTR_current_state)
            L0_ZEROPAD_CONV:
            begin
                sharedCnt <= ~ready ? sharedCnt + 'd1 : sharedCnt;
            end
            L1_K0_MAXPOOLING_COMPARE_MAX,L1_K1_MAXPOOLING_COMPARE_MAX:
            begin
                sharedCnt <= sharedCnt + 'd1;
            end
            default:
            begin
                sharedCnt <= 'd0;
            end
        endcase
    end
end

//IMG_PTR
reg[PTR_LENGTH-1:0] imgColPTR_wr;
reg[PTR_LENGTH-1:0] imgRowPTR_wr;

always @(posedge clk or posedge reset)
begin
    imgColPTR_rd <= reset || L1_DoneFlag ? 'd1 : L0_DoneFlag ?  'd0 : imgColPTR_wr;
    imgRowPTR_rd <= reset || L1_DoneFlag ? 'd1 : L0_DoneFlag ?  'd0 : imgRowPTR_wr;
end

always @(*)
begin
    case(mainCTR_current_state)
        L0_K1_BIAS_RELU:
        begin
            imgColPTR_wr  = L0_imgRightBoundReach_Flag ? 'd1 : imgColPTR_rd + 'd1;
            imgRowPTR_wr  = L0_imgRightBoundReach_Flag ? (imgRowPTR_rd + 'd1) : imgRowPTR_rd;
        end
        L1_K1_MAXPOOLING_WB:
        begin
            imgColPTR_wr  = L1_imgRightBoundReach_Flag ? 'd0 : imgColPTR_rd + 'd1;
            imgRowPTR_wr  = L1_imgRightBoundReach_Flag ? (imgRowPTR_rd + 'd1) : imgRowPTR_rd;
        end
        default:
        begin
            imgColPTR_wr  = imgColPTR_rd;
            imgRowPTR_wr  = imgRowPTR_rd;
        end
    endcase
end

wire[11:0] maxPoolingWriteAddr = imgColPTR_rd + imgRowPTR_rd * IMAGE_WIDTH_HEIGHT;

parameter NO_ACCESS= 3'b000;
parameter L0_MEM0_ACCESS= 3'b001;
parameter L0_MEM1_ACCESS= 3'b010;
parameter L1_MEM0_ACCESS= 3'b011;
parameter L1_MEM1_ACCESS= 3'b100;
parameter L2_MEM_ACCESS= 3'b101;

reg[19:0] maxPooling_result_rd;
reg[19:0] flattenTemp_rd;
reg[19:0] flattenWriteAddr;

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
        L1_K0_MAXPOOLING_WB,L2_K0_RD:
        begin
            csel = L1_MEM0_ACCESS;
        end
        L1_K1_MAXPOOLING_WB,L2_K1_RD:
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
always @(*)
begin
    if(reset)
    begin
        busy = 0;
    end
    else if(STATE_DONE)
    begin
        busy = 0;
    end
    else
    begin
        busy = 1;
    end
end

//------------------------------------DATAPATH--------------------------------//
//When sharing components, first declare the components you want to share
//Second, Declare the input wr and output rd of the components
//Lastly declare the varaiable as wires you instantiate the components then connect them together
//2 registers needed
reg signed[DATA_WIDTH-1:0] sharedReg1_rd;
reg signed[DATA_WIDTH-1:0] sharedReg2_rd;
reg signed[DATA_WIDTH-1:0] sharedReg1_wr;
reg signed[DATA_WIDTH-1:0] sharedReg2_wr;
reg sharedReg1Ld;
reg sharedReg2Ld;

//Ld control for registers
always @(*)
begin
    case(mainCTR_current_state)
        L0_ZEROPAD_CONV:
        begin
            {sharedReg1Ld,sharedReg2Ld} = 2'b11;
        end
        L0_K0_BIAS_RELU,L0_K1_BIAS_RELU,L1_K0_MAXPOOLING_COMPARE_MAX,L1_K1_MAXPOOLING_COMPARE_MAX,L2_K0_RD,L2_K1_RD:
        begin
            {sharedReg1Ld,sharedReg2Ld} = 2'b10;
        end
        default:
        begin
            {sharedReg1Ld,sharedReg2Ld} = 2'b00;
        end
    endcase
end

always @(posedge clk or posedge reset)
begin
    sharedReg1_rd <= reset ? 'd0 : sharedReg1Ld ? sharedReg1_wr : sharedReg1_rd;
    sharedReg2_rd <= reset ? 'd0 : sharedReg2Ld ? sharedReg2_wr : sharedReg2_rd;
end

//2 Multipliers for ZeroPadConv, uses input2 as kernal value input
wire signed[DATA_WIDTH*2 - 1 : 0] serialMultiplier1_o = multiplier1_input1 * multiplier1_input2;
wire signed[DATA_WIDTH*2 - 1 : 0] serialMultiplier2_o = multiplier2_input1 * multiplier2_input2;

reg signed[DATA_WIDTH - 1 : 0] multiplier1_input1;
wire signed[DATA_WIDTH - 1 : 0]  multiplier1_input2;

reg signed[DATA_WIDTH - 1 : 0] multiplier2_input1;
wire signed[DATA_WIDTH - 1 : 0]  multiplier2_input2;

//1 signed adder for BIAS_ReLU
wire signed[DATA_WIDTH-1:0] signedAdder_o = signedAdder_input1 + BIAS;
reg signed[DATA_WIDTH-1:0] signedAdder_input1;



//1 Comparator for BIAS_ReLU & Maxpooling
wire compare_gt = comparatorInput1 > comparatorInput2 ;
reg signed[DATA_WIDTH-1:0] comparatorInput1;
reg signed[DATA_WIDTH-1:0] comparatorInput2;

//L0
//L0 K0 K1 ZeroPadConv
wire[DATA_WIDTH-1:0] conv_K0_Result_wr;
wire[DATA_WIDTH-1:0] conv_K1_Result_wr;
wire[DATA_WIDTH-1:0] conv_K0_Result_rd;
wire[DATA_WIDTH-1:0] conv_K1_Result_rd;

reg[PTR_LENGTH-1:0] offsetColPTR;
reg[PTR_LENGTH-1:0] offsetRowPTR;

wire zeroPad = offsetColPTR == 'd0 || offsetColPTR == 'd65 || offsetRowPTR == 'd0 || offsetRowPTR == 'd65 ;

wire[11:0] zeroPadWriteAddr =  (imgColPTR_rd - 'd1) + (imgRowPTR_rd - 'd1) * IMAGE_WIDTH_HEIGHT;
wire[11:0] zeroPadReadAddr  = zeroPad ? 'd0 : (offsetColPTR - 'd1) + (offsetRowPTR - 'd1) * IMAGE_WIDTH_HEIGHT;

assign iaddr = STATE_L0_ZEROPAD_CONV ? zeroPadReadAddr : 'dz;

wire[DATA_WIDTH-1:0] conv_pixel = zeroPad ? 'd0 : idata;
always @(*)
begin
    if(STATE_L0_ZEROPAD_CONV)
    begin
        case(sharedCnt)
            'd0:
            begin
                offsetColPTR = (imgColPTR_rd - 'd1);
                offsetRowPTR = (imgRowPTR_rd - 'd1);
            end
            'd1:
            begin
                offsetColPTR = (imgColPTR_rd);
                offsetRowPTR = (imgRowPTR_rd - 'd1);
            end
            'd2:
            begin
                offsetColPTR = imgColPTR_rd + 'd1;
                offsetRowPTR = imgRowPTR_rd - 'd1;
            end
            'd3:
            begin
                offsetColPTR = imgColPTR_rd - 'd1;
                offsetRowPTR = imgRowPTR_rd ;
            end
            'd4:
            begin
                offsetColPTR = imgColPTR_rd;
                offsetRowPTR = imgRowPTR_rd;
            end
            'd5:
            begin
                offsetColPTR = imgColPTR_rd + 'd1;
                offsetRowPTR = imgRowPTR_rd;
            end
            'd6:
            begin
                offsetColPTR = imgColPTR_rd - 'd1;
                offsetRowPTR = imgRowPTR_rd + 'd1;
            end
            'd7:
            begin
                offsetColPTR = imgColPTR_rd;
                offsetRowPTR = imgRowPTR_rd + 'd1;
            end
            'd8:
            begin
                offsetColPTR = imgColPTR_rd + 'd1;
                offsetRowPTR = imgRowPTR_rd + 'd1;
            end
            default:
            begin
                offsetColPTR = imgColPTR_rd;
                offsetRowPTR = imgRowPTR_rd;
            end
        endcase
    end
    else if(STATE_L1_K0_MAXPOOLING_COMPARE_MAX || STATE_L1_K1_MAXPOOLING_COMPARE_MAX)
    begin
        case(sharedCnt)
            'd0:
            begin
                offsetColPTR = imgColPTR_rd;
                offsetRowPTR = imgRowPTR_rd;
            end
            'd1:
            begin
                offsetColPTR = (imgColPTR_rd + 'd1);
                offsetRowPTR = imgRowPTR_rd;
            end
            'd2:
            begin
                offsetColPTR = imgColPTR_rd;
                offsetRowPTR = imgRowPTR_rd + 'd1;
            end
            'd3:
            begin
                offsetColPTR = imgColPTR_rd + 'd1;
                offsetRowPTR = imgRowPTR_rd + 'd1;
            end
            default:
            begin
                offsetColPTR = imgColPTR_rd;
                offsetRowPTR = imgRowPTR_rd;
            end
        endcase
    end
    else
    begin
        offsetColPTR = imgColPTR_rd;
        offsetRowPTR = imgRowPTR_rd;
    end
end



always @(*)
begin
    if(STATE_L0_ZEROPAD_CONV)
    begin
        sharedReg1_wr = conv_K0_Result_wr;
        sharedReg2_wr = conv_K1_Result_wr;
    end
    else
    begin
        sharedReg1_wr = 'dz;
        sharedReg2_wr = 'dz;
    end
end

//Giving KERNAL VALUES to multipliers
always @(*)
begin
    if(STATE_L0_ZEROPAD_CONV)
    begin
        case(sharedCnt)
            'd0:
            begin
                multiplier1_input1 = KERNAL0_00;
                multiplier2_input1 = KERNAL1_00;
            end
            'd1:
            begin
                multiplier1_input1 = KERNAL0_01;
                multiplier2_input1 = KERNAL1_01;
            end

            'd2:
            begin
                multiplier1_input1 = KERNAL0_02;
                multiplier2_input1 = KERNAL1_02;
            end

            'd3:
            begin
                multiplier1_input1 = KERNAL0_10;
                multiplier2_input1 = KERNAL1_10;
            end

            'd4:
            begin
                multiplier1_input1 = KERNAL0_11;
                multiplier2_input1 = KERNAL1_11;
            end

            'd5:
            begin
                multiplier1_input1 = KERNAL0_12;
                multiplier2_input1 = KERNAL1_12;
            end

            'd6:
            begin
                multiplier1_input1 = KERNAL0_20;
                multiplier2_input1 = KERNAL1_20;
            end

            'd7:
            begin
                multiplier1_input1 = KERNAL0_21;
                multiplier2_input1 = KERNAL1_21;
            end

            'd8:
            begin
                multiplier1_input1 = KERNAL0_22;
                multiplier2_input1 = KERNAL1_22;
            end
            default:
            begin
                multiplier1_input1 = 'd0;
                multiplier2_input1 = 'd0;
            end
        endcase
    end
    else
    begin
        multiplier1_input1 = 'd0;
        multiplier2_input1 = 'd0;
    end
end

assign multiplier1_input2 = STATE_L0_ZEROPAD_CONV ? conv_pixel  : 'd0;
assign multiplier2_input2 = STATE_L0_ZEROPAD_CONV ? conv_pixel :  'd0;

assign conv_K0_Result_wr = STATE_L0_ZEROPAD_CONV ?  serialMultiplier1_o[34:15] + conv_K0_Result_rd : 'dz;
assign conv_K1_Result_wr = STATE_L0_ZEROPAD_CONV ?  serialMultiplier2_o[34:15] + conv_K1_Result_rd : 'dz;

assign conv_K0_Result_rd = sharedReg1_rd;
assign conv_K1_Result_rd = sharedReg2_rd;

//K0 K1 BiaS ReLU
wire[DATA_WIDTH-1:0] ReLU_result_wr;
wire compare_gt_ZERO = compare_gt;
wire[DATA_WIDTH-1:0] biased_result = signedAdder_o;

parameter ZERO = 'd0;
always @(*)
begin
    if(STATE_L0_K0_BIAS_RELU)
    begin
        signedAdder_input1 = conv_K0_Result_rd;
    end
    else if(STATE_L0_K1_BIAS_RELU)
    begin
        signedAdder_input1 = conv_K1_Result_rd;
    end
    else
    begin
        signedAdder_input1 = 'd0;
    end
end

always @(*)
begin
    if(STATE_L0_K0_BIAS_RELU || STATE_L0_K1_BIAS_RELU)
    begin
        comparatorInput1 = signedAdder_o;
        comparatorInput2 = ZERO;
    end
    else
    begin
        comparatorInput1 = 'dz;
        comparatorInput2 = 'dz;
    end
end

assign ReLU_result_wr = compare_gt_ZERO ? signedAdder_o : 'd0;

always @(*)
begin
    if(STATE_L0_K0_BIAS_RELU||STATE_L0_K1_BIAS_RELU)
    begin
        sharedReg1_wr = ReLU_result_wr;
    end
    else
    begin
        sharedReg1_wr = 'dz;
    end
end

//K0 K1 WB
wire[DATA_WIDTH-1:0] L0_Result_rd = STATE_L0_K0_WB ? conv_K0_Result_rd : (STATE_L0_K1_WB ? conv_K1_Result_rd : 'd0);

always @(*)
begin
    if(STATE_L0_K0_WB||STATE_L0_K1_WB)
    begin
        cdata_wr = L0_Result_rd;
        caddr_wr = zeroPadWriteAddr;
    end
    else
    begin
        cdata_wr = 'dz;
        caddr_wr = 'dz;
    end
end

//L1
//K0 K1 MP
wire[DATA_WIDTH-1:0] tempMax_rd = sharedReg1_rd;

always @(*)
begin
    if(STATE_L1_K0_MAXPOOLING_COMPARE_MAX || STATE_L1_K1_MAXPOOLING_COMPARE_MAX)
    begin
        comparatorInput1    = cdata_rd;
        comparatorInput2    = tempMax_rd;
    end
    else
    begin
        comparatorInput1    = 'dz;
        comparatorInput2    = 'dz;
    end
end

wire[DATA_WIDTH-1:0] tempMax_wr = compare_gt ? cdata_rd : tempMax_rd;

always @(*)
begin
    if(STATE_L1_K0_MAXPOOLING_COMPARE_MAX || STATE_L1_K1_MAXPOOLING_COMPARE_MAX)
    begin
        sharedReg1_wr = tempMax_wr;
    end
    else
    begin
        sharedReg1_wr = 'dz;
    end
end

//L2
//K0,K1 RD WB
wire[DATA_WIDTH-1:0] K0_K1_temp_rd = sharedReg1_rd;

always @(*)
begin
    if(STATE_L2_K0_RD || STATE_L2_K1_RD)
    begin
        sharedReg1_wr = cdata_rd;
    end
    else
    begin
        sharedReg1_wr = 'dz;
    end
end

always @(*)
begin
    if(STATE_L2_K0_WB || STATE_L2_K1_WB)
    begin
        cdata_wr = K0_K1_temp_rd;
    end
    else
    begin
        cdata_wr = 'dz;
    end
end

endmodule
