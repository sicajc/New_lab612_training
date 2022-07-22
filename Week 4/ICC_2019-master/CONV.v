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
reg[3:0] sharedCnt;

reg[6:0] imgColPTR_r;
reg[6:0] imgRowPTR_r;
reg[11:0] mem0_cnt,mem1_cnt;

//CONSTANTS
parameter KERNAL_WIDTH = 3;
parameter KERNAL_MID   = 1;
parameter IMAGE_WIDTH_HEIGHT = 64;
parameter PIXELS_OF_KERNAL = 'd8;
parameter MP_KERNAL_SIZE = 'd3;
parameter IMG_RD_DONE = 4096;

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

parameter L2_K0_FLATTEN = 'd10;
parameter L2_K0_WB = 'd11;
parameter L2_K1_FLATTEN = 'd12;
parameter L2_K1_WB = 'd13;

parameter DONE = 'd7;

//STATES INDICATORS
wire STATE_L0_ZEROPAD_CONV                  =  mainCTR_current_state == L0_ZEROPAD_CONV;
wire STATE_L0_K0_BIAS_RELU               =  mainCTR_current_state == L0_K0_BIAS_RELU;
wire STATE_L0_K1_BIAS_RELU               =  mainCTR_current_state == L0_K1_BIAS_RELU;
wire STATE_L0_K0_WB                         =   mainCTR_current_state == L0_K0_WB;
wire STATE_L0_K1_WB                         =   mainCTR_current_state == L0_K1_WB;

wire STATE_L1_K0_MAXPOOLING_COMPARE_MAX    = mainCTR_current_state == L1_K0_MAXPOOLING_COMPARE_MAX;
wire STATE_L1_K1_MAXPOOLING_COMPARE_MAX    = mainCTR_current_state == L1_K1_MAXPOOLING_COMPARE_MAX;
wire STATE_L1_K0_MAXPOOLING_WB             =  mainCTR_current_state == L1_K0_MAXPOOLING_WB;
wire STATE_L1_K1_MAXPOOLING_WB             =  mainCTR_current_state == L1_K1_MAXPOOLING_WB;

wire STATE_L2_K0_WB                         = mainCTR_current_state == L2_K0_WB;
wire STATE_L2_K1_WB                         = mainCTR_current_state == L2_K1_WB;
wire STATE_L2_K1_FLATTEN                          = mainCTR_current_state == L2_K1_FLATTEN;

wire STATE_DONE                             =  mainCTR_current_state == DONE;


//FLAGS
wire L0_LocalZeroPadConv_DoneFlag = STATE_L0_ZEROPAD_CONV ? (sharedCnt == PIXELS_OF_KERNAL) : 'd0;
wire L0_DoneFlag = ((imgRowPTR_r == IMAGE_WIDTH_HEIGHT) && (imgColPTR_r == IMAGE_WIDTH_HEIGHT)) ;
wire L0_imgRightBoundReach_Flag = STATE_L0_K1_BIAS_RELU ? 'd0 : (imgRowPTR_r == IMAGE_WIDTH_HEIGHT);

wire L1_K0_MaxPoolingCompare_DoneFlag = STATE_L1_K0_MAXPOOLING_COMPARE_MAX ? (sharedCnt == MP_KERNAL_SIZE) :'d0;
wire L1_K1_MaxPoolingCompare_DoneFlag = STATE_L1_K1_MAXPOOLING_COMPARE_MAX ? (sharedCnt == MP_KERNAL_SIZE) :'d0;
wire L1_DoneFlag = ((imgRowPTR_r == IMAGE_WIDTH_HEIGHT-1) && (imgColPTR_r == IMAGE_WIDTH_HEIGHT - 1));
wire L1_imgRightBoundReach_Flag = STATE_L1_K1_MAXPOOLING_WB ? 'd0 : (imgRowPTR_r == IMAGE_WIDTH_HEIGHT - 1);

wire L2_flatten_DoneFlag = (mem0_cnt == IMG_RD_DONE) && (mem1_cnt == IMG_RD_DONE);


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
            mainCTR_next_state = L1_DoneFlag ?  L2_K1_FLATTEN : L1_K0_MAXPOOLING_COMPARE_MAX;
        end
        L2_K1_FLATTEN:
        begin
           mainCTR_next_state = L2_flatten_DoneFlag ? DONE : L2_K1_FLATTEN;
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
|| L1_K0_MaxPoolingCompare_DoneFlag || L1_K1_MaxPoolingCompare_DoneFlag) ? 'd0 : (sharedCnt + 'd1);


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

reg[11:0] conv_result_r;

wire[11:0] addrZeroPad =  (imgColPTR_r - 'd1) + (imgRowPTR_r - 'd1) * IMAGE_WIDTH_HEIGHT;
wire[11:0] addrMaxPooling = imgColPTR_r + imgRowPTR_r * IMAGE_WIDTH_HEIGHT;

parameter NO_ACCESS= 3'b000;
parameter L0_MEM0_ACCESS= 3'b001;
parameter L0_MEM1_ACCESS= 3'b010;
parameter L1_MEM0_ACCESS= 3'b011;
parameter L1_MEM1_ACCESS= 3'b100;
parameter L2_MEM_ACCESS= 3'b101;

reg[19:0] maxPooling_result_r;

//Read write bus control
always @(*)
begin
    case(mainCTR_current_state)
        L0_K0_WB,L0_K1_WB:
        begin
            cdata_wr = conv_result_r;
            caddr_wr = addrZeroPad;
        end
        L1_K1_MAXPOOLING_WB,L1_K0_MAXPOOLING_WB:
        begin
            cdata_wr = maxPooling_result_r;
            caddr_wr = addrMaxPooling;
        end
        default:
        begin
            cdata_wr = 0;
            caddr_wr = 0;
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
        L2_K1_FLATTEN:
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
        L0_K0_WB,L0_K1_WB,L1_K0_MAXPOOLING_WB,L1_K1_MAXPOOLING_WB:
        begin
            {crd,cwr} = 2'b01;
        end
        L1_K0_MAXPOOLING_COMPARE_MAX,L1_K1_MAXPOOLING_COMPARE_MAX:
        begin
            {crd,cwr} = 2'b10;
        end
        L2_K1_FLATTEN:
        begin

        end
        default:
        begin
        end
    endcase
end





















endmodule
