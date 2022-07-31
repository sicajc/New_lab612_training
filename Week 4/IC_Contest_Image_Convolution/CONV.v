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
reg[3:0] currentState,nextState;
reg[10:0] sharedCnt;
reg[10:0] flatten_MemAddrCnt;

reg[PTR_LENGTH-1:0] imgColPTR;
reg[PTR_LENGTH-1:0] imgRowPTR;

reg[DATA_WIDTH-1:0] kernal1_Value,kernal0_Value;
//Registers
reg[DATA_WIDTH-1:0] R1_rd;
reg[DATA_WIDTH-1:0] R2_rd;
reg[DATA_WIDTH-1:0] R1_wr;
reg[DATA_WIDTH-1:0] R2_wr;

//PTRs
reg[PTR_LENGTH-1:0] imgColOffsetPTR;
reg[PTR_LENGTH-1:0] imgRowOffsetPTR;

//IO
wire[DATA_WIDTH-1:0] MP_PixelAddr_o = (imgColOffsetPTR) + (imgRowOffsetPTR) * IMAGE_WIDTH_HEIGHT;
wire[DATA_WIDTH-1:0] flatten_PixelAddr_o = sharedCnt;
wire[ADDR_WIDTH-1:0] ConvLocalOffsetPixelAddr_o = (imgColOffsetPTR - 1) + (imgRowOffsetPTR - 1) * IMAGE_WIDTH_HEIGHT;

wire[ADDR_WIDTH-1:0] ConvLocalOffsetPixelValue_i;
wire[DATA_WIDTH-1:0] MP_PixelValue_i;
wire[DATA_WIDTH-1:0] flatten_PixelValue_i;

//K0 Variables
wire[DATA_WIDTH-1:0] K0_K1_localConvolutionResult_rd = R1_rd;
wire[ADDR_WIDTH-1:0] K0_K1_LocalWBAddr_o            = (imgColPTR-1) + (imgRowPTR-1) * IMAGE_WIDTH_HEIGHT;

wire[DATA_WIDTH-1:0] K0_K1_MP_Result_rd              = R1_rd;
wire[ADDR_WIDTH-1:0] K0_K1_MP_WBAddr_o              = imgColPTR + imgRowPTR * IMAGE_WIDTH_HEIGHT;

wire[DATA_WIDTH-1:0] K0_Flatten_Result_rd            = R1_rd;
wire[ADDR_WIDTH-1:0] K0_Flatten_WBAddr_o            = flatten_MemAddrCnt;

wire[DATA_WIDTH-1:0] K0_K1_ReLUResult_rd;

//K1 Varaibles
wire[DATA_WIDTH-1:0] K1_Flatten_Result_rd            = R1_rd;
wire[ADDR_WIDTH-1:0] K1_Flatten_WBAddr_o             = flatten_MemAddrCnt;

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
parameter PTR_LENGTH = 8;

//STATES
parameter L0_ZEROPAD_CONV = 'd0;
parameter L0_K0_BIAS_RELU = 'd1;
parameter L0_K1_BIAS_RELU = 'd2;
parameter L0_K0_WB = 'd3;
parameter L0_K1_WB = 'd4;

parameter L1_K0_MAXPOOLING_COMPARE_MAX = 'd5;
parameter L1_K0_MAXPOOLING_WB = 'd6;
parameter L1_K1_MAXPOOLING_COMPARE_MAX = 'd7;
parameter L1_K1_MAXPOOLING_WB = 'd8;

parameter L2_K0_RD = 'd9;
parameter L2_K1_RD = 'd10;
parameter L2_K0_WB = 'd11;
parameter L2_K1_WB = 'd12;

parameter DONE = 'd13;

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
wire STATE_L0_ZEROPAD_CONV                  =  currentState == L0_ZEROPAD_CONV;
wire STATE_L0_K0_BIAS_RELU                  =  currentState == L0_K0_BIAS_RELU;
wire STATE_L0_K1_BIAS_RELU                  =  currentState == L0_K1_BIAS_RELU;
wire STATE_L0_K0_WB                         =  currentState == L0_K0_WB;
wire STATE_L0_K1_WB                         =  currentState == L0_K1_WB;

wire STATE_L1_K0_MAXPOOLING_COMPARE_MAX    =  currentState == L1_K0_MAXPOOLING_COMPARE_MAX;
wire STATE_L1_K1_MAXPOOLING_COMPARE_MAX    =  currentState == L1_K1_MAXPOOLING_COMPARE_MAX;
wire STATE_L1_K0_MAXPOOLING_WB             =  currentState == L1_K0_MAXPOOLING_WB;
wire STATE_L1_K1_MAXPOOLING_WB             =  currentState == L1_K1_MAXPOOLING_WB;

wire STATE_L2_K0_RD                         = currentState == L2_K0_RD;
wire STATE_L2_K1_RD                         = currentState == L2_K1_RD;
wire STATE_L2_K0_WB                         = currentState == L2_K0_WB;
wire STATE_L2_K1_WB                         = currentState == L2_K1_WB;

wire STATE_DONE                             = currentState == DONE;

//FLAGS
wire L0LocalZeroPadConvDone_flag     = (sharedCnt == PIXELS_OF_KERNAL);
wire L0_DoneFlag                      = ((imgRowPTR == IMAGE_WIDTH_HEIGHT) && (imgColPTR == IMAGE_WIDTH_HEIGHT));
wire L0_imgRightBoundReach_Flag       = (imgColPTR == IMAGE_WIDTH_HEIGHT);

wire L1_K0_MaxPoolingCompareDone_flag = (sharedCnt == MP_KERNAL_SIZE) ;
wire L1_K1_MaxPoolingCompareDone_flag = (sharedCnt == MP_KERNAL_SIZE) ;
wire L1_imgRightBoundReach_Flag       = (imgColPTR == IMAGE_WIDTH_HEIGHT - 2) ;
wire L1_DoneFlag                      = (K0_K1_MP_WBAddr_o == 'd1023) ;

wire L2_K0_flatten_DoneFlag           =(sharedCnt == MP_IMG_RD_DONE) ;
wire L2_K1_flatten_DoneFlag           =(sharedCnt == MP_IMG_RD_DONE) ;

wire L0_mode  = STATE_L0_ZEROPAD_CONV;
wire L1_mode  = STATE_L1_K0_MAXPOOLING_COMPARE_MAX || STATE_L1_K1_MAXPOOLING_COMPARE_MAX;

//----------------------------------CONTROL_PATH--------------------------------//
always @(posedge clk or posedge reset)
begin
    currentState <= reset ? L0_ZEROPAD_CONV : nextState;
end

always @(*)
begin
    case(currentState)
        L0_ZEROPAD_CONV:
        begin
            nextState = L0LocalZeroPadConvDone_flag ? L0_K0_BIAS_RELU : L0_ZEROPAD_CONV;
        end
        L0_K0_BIAS_RELU:
        begin
            nextState = L0_K0_WB;
        end
        L0_K0_WB:
        begin
            nextState = L0_K1_BIAS_RELU;
        end
        L0_K1_BIAS_RELU:
        begin
            nextState = L0_K1_WB;
        end
        L0_K1_WB:
        begin
            nextState = L0_DoneFlag ? L1_K0_MAXPOOLING_COMPARE_MAX : L0_ZEROPAD_CONV;
        end
        L1_K0_MAXPOOLING_COMPARE_MAX:
        begin
            nextState = L1_K0_MaxPoolingCompareDone_flag ? L1_K0_MAXPOOLING_WB : L1_K0_MAXPOOLING_COMPARE_MAX;
        end
        L1_K0_MAXPOOLING_WB:
        begin
            nextState = L1_K1_MAXPOOLING_COMPARE_MAX;
        end
        L1_K1_MAXPOOLING_COMPARE_MAX:
        begin
            nextState = L1_K1_MaxPoolingCompareDone_flag ? L1_K1_MAXPOOLING_WB : L1_K1_MAXPOOLING_COMPARE_MAX;
        end
        L1_K1_MAXPOOLING_WB:
        begin
            nextState = L1_DoneFlag ? L2_K0_RD : L1_K0_MAXPOOLING_COMPARE_MAX;
        end
        L2_K0_RD:
        begin
            nextState = L2_K0_flatten_DoneFlag ? L2_K1_RD : L2_K0_WB;
        end
        L2_K0_WB:
        begin
            nextState = L2_K0_RD;
        end
        L2_K1_RD:
        begin
            nextState = L2_K1_flatten_DoneFlag ? DONE: L2_K1_WB;
        end
        L2_K1_WB:
        begin
            nextState = L2_K1_RD;
        end
        DONE:
        begin
            nextState = DONE;
        end
        default:
        begin
            nextState = L0_ZEROPAD_CONV;
        end
    endcase
end

//sharedCnt
always @(posedge clk or posedge reset)
begin
    if(reset)
    begin
        sharedCnt <= 'd0;
    end
    else
    begin
        case(currentState)
            L0_ZEROPAD_CONV:
            begin
                sharedCnt <= L0LocalZeroPadConvDone_flag ?  'd0 : sharedCnt + 'd1;
            end
            L1_K0_MAXPOOLING_COMPARE_MAX,L1_K1_MAXPOOLING_COMPARE_MAX:
            begin
                sharedCnt <= L1_K0_MaxPoolingCompareDone_flag || L1_K1_MaxPoolingCompareDone_flag ? 'd0 : sharedCnt + 'd1;
            end
            L2_K0_RD,L2_K1_RD:
            begin
                sharedCnt <= sharedCnt + 'd1;
            end
            L2_K1_WB,L2_K1_WB:
            begin
                sharedCnt <= sharedCnt + 'd2;
            end
            default:
            begin
                sharedCnt <= 'd0;
            end
        endcase
    end
end
//flatten MemoryAddrCNT
always @(posedge clk or posedge reset)
begin
    if(reset)
    begin
        flatten_MemAddrCnt <= 'd0;
    end
    else if(STATE_L2_K0_WB)
    begin
        flatten_MemAddrCnt <=  L2_K0_flatten_DoneFlag ? 'd1: flatten_MemAddrCnt + 'd2;
    end
    else if(STATE_L2_K1_WB)
    begin
        flatten_MemAddrCnt <=  L2_K1_flatten_DoneFlag? 'd0: flatten_MemAddrCnt + 'd2;
    end
    else
    begin
        flatten_MemAddrCnt <= flatten_MemAddrCnt;
    end
end

//PTRs
always @(posedge clk or posedge reset)
begin
    if(reset)
    begin
        imgColPTR <= 'd0;
        imgRowPTR  <= 'd0;
    end
    else if(STATE_L0_ZEROPAD_CONV)
    begin
        imgColPTR <= (L0_DoneFlag || L0_imgRightBoundReach_Flag) ? 'd0 : imgColPTR + 'd1;
        imgRowPTR <= L0_DoneFlag ? 'd0 : (L0_imgRightBoundReach_Flag ? imgRowPTR + 'd1 : imgRowPTR);
    end
    else if(STATE_L1_K1_MAXPOOLING_WB)
    begin
        imgColPTR <= L1_imgRightBoundReach_Flag ? 'd0 : imgColPTR + 'd2;
        imgRowPTR <= L1_imgRightBoundReach_Flag ? imgRowPTR + 'd2 : imgRowPTR;
    end
    else
    begin
        imgColPTR  <= imgColPTR;
        imgRowPTR  <= imgRowPTR;
    end
end

//offset pointers

always @(*)
begin
    if(L0_mode)
    begin
        case(sharedCnt)
            'd0:
            begin
                imgColOffsetPTR = imgColPTR - 'd1;
                imgRowOffsetPTR = imgRowPTR - 'd1;
            end
            'd1:
            begin
                imgColOffsetPTR = imgColPTR;
                imgRowOffsetPTR = imgRowPTR - 'd1;
            end
            'd2:
            begin
                imgColOffsetPTR = imgColPTR + 'd1;
                imgRowOffsetPTR = imgRowPTR - 'd1;
            end
            'd3:
            begin
                imgColOffsetPTR = imgColPTR - 'd1;
                imgRowOffsetPTR = imgRowPTR;
            end
            'd4:
            begin
                imgColOffsetPTR = imgColPTR;
                imgRowOffsetPTR = imgRowPTR;
            end
            'd5:
            begin
                imgColOffsetPTR = imgColPTR + 'd1;
                imgRowOffsetPTR = imgRowPTR;
            end
            'd6:
            begin
                imgColOffsetPTR = imgColPTR - 'd1;
                imgRowOffsetPTR = imgRowPTR + 'd1;
            end
            'd7:
            begin
                imgColOffsetPTR = imgColPTR;
                imgRowOffsetPTR = imgRowPTR + 'd1;
            end
            'd8:
            begin
                imgColOffsetPTR = imgColPTR + 'd1;
                imgRowOffsetPTR = imgRowPTR + 'd1;
            end
            default:
            begin
                imgColOffsetPTR = imgColPTR ;
                imgRowOffsetPTR = imgRowPTR ;
            end
        endcase
    end
    else if(L1_mode)
    begin
        case(sharedCnt)
            'd0:
            begin
                imgColOffsetPTR = imgColPTR;
                imgRowOffsetPTR = imgRowPTR;
            end
            'd1:
            begin
                imgColOffsetPTR = imgColPTR + 'd1;
                imgRowOffsetPTR = imgRowPTR;
            end
            'd2:
            begin
                imgColOffsetPTR = imgColPTR;
                imgRowOffsetPTR = imgRowPTR + 'd1;
            end
            'd3:
            begin
                imgColOffsetPTR = imgColPTR + 'd1;
                imgRowOffsetPTR = imgRowPTR + 'd1;
            end
            default:
            begin
                imgColOffsetPTR = imgColPTR;
                imgRowOffsetPTR = imgRowPTR;
            end
        endcase
    end
    else
    begin
        imgColOffsetPTR = imgColPTR;
        imgRowOffsetPTR = imgRowPTR;
    end
end

//I/O
parameter NO_ACCESS= 3'b000;
parameter L0_MEM0_ACCESS= 3'b001;
parameter L0_MEM1_ACCESS= 3'b010;
parameter L1_MEM0_ACCESS= 3'b011;
parameter L1_MEM1_ACCESS= 3'b100;
parameter L2_MEM_ACCESS= 3'b101;

always @(*)
begin
    case(currentState)
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
    case(currentState)
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

//cdata_wr and caddr_wr
assign K0_K1_ReLUResult_rd  = R1_rd;

always @(*)
begin
    case(currentState)
        L0_K0_WB,L0_K1_WB:
        begin
            cdata_wr = K0_K1_ReLUResult_rd;
            caddr_wr = K0_K1_LocalWBAddr_o;
        end
        L1_K0_MAXPOOLING_WB,L1_K1_MAXPOOLING_WB:
        begin
            cdata_wr = K0_K1_MP_Result_rd;
            caddr_wr = K0_K1_MP_WBAddr_o;
        end
        L2_K0_WB:
        begin
            cdata_wr = K0_Flatten_Result_rd;
            caddr_wr = K0_Flatten_WBAddr_o;
        end
        L2_K1_WB:
        begin
            cdata_wr = K1_Flatten_Result_rd;
            caddr_wr = K1_Flatten_WBAddr_o;
        end
        default:
        begin
            cdata_wr = 'd0;
            caddr_wr = 'd0;
        end
    endcase
end

//caddr_rd
always @(*)
begin
    case(currentState)
        L1_K0_MAXPOOLING_COMPARE_MAX,L1_K1_MAXPOOLING_COMPARE_MAX:
        begin
            caddr_rd = MP_PixelAddr_o;
        end
        L2_K0_RD,L2_K1_RD:
        begin
            caddr_rd = flatten_PixelAddr_o;
        end
        default:
        begin
            caddr_rd = 'd0;
        end
    endcase

end
//iaddr,idata,cdata_rd
wire Zeropadding = (imgColOffsetPTR == 0 || imgColOffsetPTR == 65 || imgRowOffsetPTR == 0 || imgRowOffsetPTR == 65);

assign iaddr = ConvLocalOffsetPixelAddr_o;
assign ConvLocalOffsetPixelValue_i = Zeropadding ? 'd0 : idata;

assign MP_PixelValue_i      = cdata_rd;
assign flatten_PixelValue_i = cdata_rd;

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
//From now on, the register compoenents you use must be declared with rd and wr
//Lastly declare the varaiable as wires you instantiate the components then connect them together
//Can only use 2 register only

always @(posedge clk or posedge reset)
begin
    R1_rd <= reset ? 'd0 : R1_wr;
    R2_rd <= reset ? 'd0 : R2_wr;
end

wire[DATA_WIDTH-1:0] ReLU_Result_wr;

//R1_wr
always @(*)
begin
    case(currentState)
        L0_ZEROPAD_CONV:
        begin
            R1_wr = R1_rd + kernal0_Value * ConvLocalOffsetPixelValue_i;
        end
        L0_K0_BIAS_RELU,L0_K1_BIAS_RELU:
        begin
            R1_wr = ReLU_Result_wr;
        end
        L1_K0_MAXPOOLING_COMPARE_MAX,L1_K1_MAXPOOLING_COMPARE_MAX:
        begin
            R1_wr = MP_LocalMax_wr;
        end
        L2_K0_RD,L2_K1_RD:
        begin
            R1_wr = flatten_PixelValue_i;
        end
        default:
        begin
            R1_wr = R1_rd;
        end
    endcase
end

//R2_Wr
always @(*)
begin
    case(currentState)
        L0_ZEROPAD_CONV:
        begin
            R2_wr = R2_rd + kernal1_Value * ConvLocalOffsetPixelValue_i;
        end
        default:
        begin
            R2_wr = R2_rd;
        end
    endcase
end

//kernal0_Value
always @(*)
begin
    case(sharedCnt)
        'd0:
            kernal0_Value = KERNAL0_00;
        'd1:
            kernal0_Value = KERNAL0_01;
        'd2:
            kernal0_Value = KERNAL0_02;
        'd3:
            kernal0_Value = KERNAL0_10;
        'd4:
            kernal0_Value = KERNAL0_11;
        'd5:
            kernal0_Value = KERNAL0_12;
        'd6:
            kernal0_Value = KERNAL0_20;
        'd7:
            kernal0_Value = KERNAL0_21;
        'd8:
            kernal0_Value = KERNAL0_22;
        default:
            kernal0_Value = 'd0;
    endcase
end

//kernal1_Value
always @(*)
begin
    case(sharedCnt)
        'd0:
            kernal1_Value = KERNAL1_00;
        'd1:
            kernal1_Value = KERNAL1_01;
        'd2:
            kernal1_Value = KERNAL1_02;
        'd3:
            kernal1_Value = KERNAL1_10;
        'd4:
            kernal1_Value = KERNAL1_11;
        'd5:
            kernal1_Value = KERNAL1_12;
        'd6:
            kernal1_Value = KERNAL1_20;
        'd7:
            kernal1_Value = KERNAL1_21;
        'd8:
            kernal1_Value = KERNAL1_22;
        default:
            kernal1_Value = 'd0;
    endcase
end

//ReLU bias datapath
wire[DATA_WIDTH-1:0] biasedResult;

assign ReLU_Result_wr = (biasedResult > 0) ? biasedResult : 'd0;
assign biasedResult   = K0_K1_localConvolutionResult_rd + BIAS;

//L1
//K0 K1 MP
wire[DATA_WIDTH-1:0] MP_LocalMax_wr = (MP_PixelValue_i > MP_LocalMax_rd) ? MP_PixelValue_i : MP_LocalMax_rd;
assign MP_LocalMax_rd = R1_rd;

endmodule
