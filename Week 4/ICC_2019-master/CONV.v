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
output	reg [11:0]			iaddr         ;
input	[19:0]				idata         ;
output	 reg				cwr           ;
output	 reg[11:0]  		caddr_wr      ;
output reg signed[19:0] 	cdata_wr      ;
output	 reg				crd           ;
output	reg [11:0]			caddr_rd      ;
input	 [19:0]				cdata_rd      ;
output	 reg [2:0]			csel          ;

reg[3:0] mainCTR_current_state,mainCTR_next_state;
reg[3:0] sharedCnt;

reg[6:0] imgColPTR_r;
reg[6:0] imgRowPTR_r;


parameter KERNAL_WIDTH = 3;
parameter KERNAL_MID   = 1;
parameter IMAGE_WIDTH_HEIGHT = 64;
parameter PIXELS_OF_KERNAL = 'd8;
parameter MP_KERNAL_SIZE = 'd3;

parameter L0_ZEROPAD_CONV = 'd0;
parameter L0_K0_BIAS_RELU_WB = 'd1;
parameter L0_K1_BIAS_RELU_WB = 'd2;
parameter L12_K0_MAXPOOLING_COMPARE_MAX = 'd3;
parameter L12_K0_MAXPOOLING_WB = 'd4;
parameter L12_K1_MAXPOOLING_COMPARE_MAX = 'd5;
parameter L12_K1_MAXPOOLING_WB = 'd6;
parameter DONE = 'd7;

wire STATE_L0_ZEROPAD_CONV                  =  mainCTR_current_state == 'd0;
wire STATE_L0_K0_BIAS_RELU_WB               =  mainCTR_current_state == 'd1;
wire STATE_L0_K1_BIAS_RELU_WB               =  mainCTR_current_state == 'd2;
wire STATE_L12_K0_MAXPOOLING_COMPARE_MAX    = mainCTR_current_state == 'd3;
wire STATE_L12_K0_MAXPOOLING_WB             =  mainCTR_current_state == 'd4;
wire STATE_L12_K1_MAXPOOLING_COMPARE_MAX    = mainCTR_current_state == 'd5;
wire STATE_L12_K1_MAXPOOLING_WB             =  mainCTR_current_state == 'd6;
wire STATE_DONE                             =  mainCTR_current_state == 'd7;

wire L0_LocalZeroPadConv_DoneFlag = STATE_L0_ZEROPAD_CONV ? (sharedCnt == PIXELS_OF_KERNAL) : 'd0;

wire L0_K0_ReLUWb_DoneFlag = STATE_L0_K0_BIAS_RELU_WB ? (sharedCnt == 'd1) : 'd0;

wire L0_K1_ReLUWb_DoneFlag = STATE_L0_K1_BIAS_RELU_WB ? (sharedCnt == 'd1) : 'd0;

wire L0_DoneFlag = ((imgRowPTR_r == IMAGE_WIDTH_HEIGHT) && (imgColPTR_r == IMAGE_WIDTH_HEIGHT)) ;

wire L12_K0_MaxPoolingCompare_DoneFlag = STATE_L12_K0_MAXPOOLING_COMPARE_MAX ? (sharedCnt == MP_KERNAL_SIZE) :'d0;

wire L12_K1_MaxPoolingCompare_DoneFlag = STATE_L12_K1_MAXPOOLING_COMPARE_MAX ? (sharedCnt == MP_KERNAL_SIZE) :'d0;

wire L12_DoneFlag = ((imgRowPTR_r == IMAGE_WIDTH_HEIGHT-1) && (imgColPTR_r == IMAGE_WIDTH_HEIGHT - 1));

wire L0_imgRightBoundReach_Flag = STATE_L0_K1_BIAS_RELU_WB ? 'd0 : (imgRowPTR_r == IMAGE_WIDTH_HEIGHT);

wire L12_imgRightBoundReach_Flag = STATE_L12_K1_MAXPOOLING_WB ? 'd0 : (imgRowPTR_r == IMAGE_WIDTH_HEIGHT - 1);

always @(posedge clk or posedge reset)
begin
    mainCTR_current_state <= reset ? L0_ZEROPAD_CONV : mainCTR_next_state;
end

always @(*)
begin
    case(mainCTR_current_state)
        L0_ZEROPAD_CONV:
        begin
            mainCTR_next_state = L0_LocalZeroPadConv_DoneFlag ? L0_K0_BIAS_RELU_WB : L0_ZEROPAD_CONV;
        end
        L0_K0_BIAS_RELU_WB:
        begin
            mainCTR_next_state = L0_K0_ReLUWb_DoneFlag ? L0_K1_BIAS_RELU_WB : L0_K0_BIAS_RELU_WB;
        end
        L0_K1_BIAS_RELU_WB:
        begin
            mainCTR_next_state = L0_K1_ReLUWb_DoneFlag ?
            (L0_DoneFlag ? L12_K0_MAXPOOLING_COMPARE_MAX : L0_ZEROPAD_CONV) : L0_K1_BIAS_RELU_WB;
        end
        L12_K0_MAXPOOLING_COMPARE_MAX:
        begin
            mainCTR_next_state = L12_K0_MaxPoolingCompare_DoneFlag ? L12_K0_MAXPOOLING_WB : L12_K0_MAXPOOLING_COMPARE_MAX;
        end
        L12_K0_MAXPOOLING_WB:
        begin
            mainCTR_next_state = L12_K1_MAXPOOLING_COMPARE_MAX;
        end
        L12_K1_MAXPOOLING_COMPARE_MAX:
        begin
            mainCTR_next_state = L12_K1_MaxPoolingCompare_DoneFlag ? L12_K0_MAXPOOLING_WB : L12_K1_MAXPOOLING_COMPARE_MAX;
        end
        L12_K1_MAXPOOLING_WB:
        begin
            mainCTR_next_state = L12_DoneFlag ? L12_K0_MAXPOOLING_COMPARE_MAX : DONE;
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

always @(posedge clk or posedge reset)
begin
    if(reset)
    begin
        sharedCnt <= reset ? 'd0 : sharedCnt_w;
    end
end

wire sharedCnt_w = (L0_LocalZeroPadConv_DoneFlag || L0_K0_BIAS_RELU_WB || L0_K1_BIAS_RELU_WB
                    || L12_K0_MaxPoolingCompare_DoneFlag || L12_K1_MaxPoolingCompare_DoneFlag) ? 'd0 : (sharedCnt + 'd1);

always @(posedge clk or posedge reset)
begin
    imgColPTR_r <= reset ? 'd0 : imgColPTR_w;
    imgRowPTR_r <= reset ? 'd0 : imgRowPTR_w;
end

wire imgColPTR_w = ;
wire imgRowPTR_w = ;












endmodule
