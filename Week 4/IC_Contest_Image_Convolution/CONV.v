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
wire L0LocalZeroPadConvDone_flag     = STATE_L0_ZEROPAD_CONV ? (sharedCnt == PIXELS_OF_KERNAL) : 'dz;
wire L0_DoneFlag                      = STATE_L0_K1_WB ? ((imgRowPTR == IMAGE_WIDTH_HEIGHT) && (imgColPTR == IMAGE_WIDTH_HEIGHT)) : 'dz;
wire L0_imgRightBoundReach_Flag       = STATE_L0_K1_WB ? (imgColPTR == IMAGE_WIDTH_HEIGHT) : 'dz;

wire L1_K0_MaxPoolingCompareDone_flag = STATE_L1_K0_MAXPOOLING_COMPARE_MAX ? (sharedCnt == MP_KERNAL_SIZE) :'dz; L1_K1_MaxPoolingCompareDone_flag
wire L1_K1_MaxPoolingCompareDone_flag = STATE_L1_K1_MAXPOOLING_COMPARE_MAX ? (sharedCnt == MP_KERNAL_SIZE) :'dz;
wire L1_imgRightBoundReach_Flag       = STATE_L1_K1_MAXPOOLING_WB ? (imgColPTR == IMAGE_WIDTH_HEIGHT - 2) :'dz;
wire L1_DoneFlag                      = STATE_L1_K1_MAXPOOLING_WB ? (maxPoolingWriteAddr == 'd1023) : 'dz;

wire L2_K0_flatten_DoneFlag           = STATE_L2_K0_RD ?  (sharedCnt == MP_IMG_RD_DONE) : 'dz;
wire L2_K1_flatten_DoneFlag           = STATE_L2_K1_RD ? (sharedCnt == MP_IMG_RD_DONE)  : 'dz;

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
            nextState = L1_K0_MaxPoolingCompareDone_flag ? L1_K0_MAXPOOLING_WB : L1_K0_MAXPOOLING_COMPARE_MAX; L1_K1_MaxPoolingCompareDone_flag
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

parameter NO_ACCESS= 3'b000;
parameter L0_MEM0_ACCESS= 3'b001;
parameter L0_MEM1_ACCESS= 3'b010;
parameter L1_MEM0_ACCESS= 3'b011;
parameter L1_MEM1_ACCESS= 3'b100;
parameter L2_MEM_ACCESS= 3'b101;

//I/O
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
wire[DATA_WIDTH-1:0] K0_K1_localConvolutionResult_rd = R1_rd;
wire[ADDR_WIDTH-1:0] K0_K1_LocalWBAddr            = (imgColPTR-1) + (imgRowPTR-1) * IMAGE_WIDTH_HEIGHT;

wire[DATA_WIDTH-1:0] K0_K1_MP_Result_rd              = R1_rd;
wire[ADDR_WIDTH-1:0] K0_K1_MP_WBAddr              = imgColPTR + imgRowPTR * IMAGE_WIDTH_HEIGHT;

wire[DATA_WIDTH-1:0] K0_Flatten_Result_rd            = R1_rd;
wire[ADDR_WIDTH-1:0] K0_Flatten_WBAddr            = flatten_MemAddrCnt;

wire[DATA_WIDTH-1:0] K1_Flatten_Result_rd            = R1_rd;
wire[ADDR_WIDTH-1:0] K1_Flatten_WBAddr            = flatten_MemAddrCnt;

always @(*)
begin
    case(currentState)
        L0_K0_WB,L0_K1_WB:
        begin
            cdata_wr = K0_K1_localConvolutionResult_rd;
            caddr_wr = K0_K1_LocalWBAddr;
        end
        L1_K0_MAXPOOLING_WB,L1_K1_MAXPOOLING_WB:
        begin
            cdata_wr = K0_K1_MP_Result_rd;
            caddr_wr = K0_K1_MP_WBAddr;
        end
        L2_K0_WB:
        begin
            cdata_wr = K0_Flatten_Result_rd;
            caddr_wr = K0_Flatten_WBAddr;
        end
        L2_K1_WB:
        begin
            cdata_wr = K1_Flatten_Result_rd;
            caddr_wr = K1_Flatten_WBAddr;
        end
        default:
        begin
            cdata_wr = 'd0;
            caddr_wr = 'd0;
        end
    endcase
end

//caddr_rd
wire[DATA_WIDTH-1:0] MP_PixelAddr_i;
wire[DATA_WIDTH-1:0] flatten_PixelAddr_i;

always @(*)
begin
    case(currentState)
        L1_K0_MAXPOOLING_COMPARE_MAX,L1_K1_MAXPOOLING_COMPARE_MAX:
        begin
            caddr_rd = MP_PixelAddr_i;
        end
        L2_K0_RD,L2_K1_RD:
        begin
            caddr_rd = flatten_PixelAddr_i;
        end
        default:
        begin
            caddr_rd = 'd0;
        end
    endcase

end
//iaddr,idata,cdata_rd
wire[ADDR_WIDTH-1:0] ConvLocalOffsetPixelAddr_o;
wire[ADDR_WIDTH-1:0] ConvLocalOffsetPixelValue_i;
wire[DATA_WIDTH-1:0] MP_PixelValue_i;
wire[DATA_WIDTH-1:0] flatten_PixelValue_i;

assign iaddr = ConvLocalOffsetPixelAddr_o;
assign ConvLocalOffsetPixelValue_i = idata;

assign cdata_rd= MP_PixelValue_i;
assign cdata_rd= flatten_PixelValue_i;

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
reg[DATA_WIDTH-1:0] R1_rd;
reg[DATA_WIDTH-1:0] R2_rd;
reg[DATA_WIDTH-1:0] R1_wr;
reg[DATA_WIDTH-1:0] R2_wr;
always @(posedge clk or posedge reset)
begin
    R1_rd <= reset ? 'd0 : R1_wr;
    R2_rd <= reset ? 'd0 : R2_wr;
end

//R1_wr
always @(*)
begin
    case(currentState)
    L0_ZEROPAD_CONV:
    begin
        R1_wr = R1_rd + kernal1_Value * ConvLocalOffsetPixelValue_i;
    end


end

//1 signed adder for BIAS_ReLU

//1 Comparator for BIAS_ReLU & Maxpooling

//L0
//L0 K0 K1 ZeroPadConv

//Giving KERNAL VALUES to multiplier

//K0 K1 BiaS ReLU

//K0 K1 WB


//L1
//K0 K1 MP

//L2
//K0,K1 RD WB

endmodule
