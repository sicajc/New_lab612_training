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
output	reg				    busy          ;
input						ready         ;
output [11:0]			iaddr;
input	[19:0]				idata         ;
output	 				cwr           ;
output	 reg[11:0]  		caddr_wr          ;
output  reg signed[19:0] 	cdata_wr          ;
output	 				crd           ;
output	reg [11:0]			caddr_rd      ;
input	 [19:0]				cdata_rd      ;
output	 reg [2:0]			csel          ;

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
parameter L1_MEM_SIZE = 1024;

//Control path
reg[3:0] currentState,nextState;
reg ModeReg;

//STATES
parameter CONV = 'd0;
parameter ReLU = 'd1;
parameter L0_WB = 'd2;
parameter MAXPOOLING = 'd3;
parameter L1_WB = 'd4;
parameter FLATTEN = 'd5;
parameter L2_WB = 'd6;
parameter DONE = 'd7;

//STATE INDICATORS
wire STATE_CONV = currentState == CONV;
wire STATE_ReLU = currentState == ReLU;
wire STATE_L0_WB = currentState == L0_WB;

wire STATE_MAXPOOLING = currentState == MAXPOOLING;
wire STATE_L1_WB = currentState == L1_WB;

wire STATE_FLATTEN = currentState == FLATTEN;
wire STATE_L2_WB = currentState == L2_WB;
wire STATE_DONE = currentState == DONE;

wire K0_mode = (ModeReg == 'd0);
wire K1_mode = (ModeReg == 'd1);

wire L0 = STATE_CONV || STATE_ReLU || STATE_L0_WB;
wire L1 = STATE_MAXPOOLING || STATE_L1_WB;
wire L2 = STATE_FLATTEN || STATE_L2_WB;

//CNT
reg[10:0] sharedCnt1;
reg[10:0] sharedCnt2;

//PTR
reg[PTR_LENGTH-1:0] imgColPTR,imgRowPTR;
reg[PTR_LENGTH-1:0] imgOffsetColPTR,imgOffsetRowPTR;


//Flags
wire ZeroPad_flag = imgOffsetColPTR == 'd0 || imgOffsetRowPTR == 'd0 || imgOffsetRowPTR == 'd65 || imgOffsetColPTR == 'd65 ;
wire LocalConvDone_flag = (sharedCnt1 == 'd8);
wire CONV_ImgRightBoundReach_flag = imgOffsetColPTR == IMAGE_WIDTH_HEIGHT;
wire ImgBottomBoundReach_flag = imgOffsetRowPTR == IMAGE_WIDTH_HEIGHT;
wire ConvDone_flag = (CONV_ImgRightBoundReach_flag && ImgBottomBoundReach_flag);
wire L0_Done_flag = K1_mode && ConvDone_flag;


wire MP_ImgRightBoundReach_flag = (imgColPTR == IMAGE_WIDTH_HEIGHT - 1);
wire MP_ImgBottomBoundReach_flag = (imgRowPTR == IMAGE_WIDTH_HEIGHT - 1);
wire LocalMaxPoolingDone_flag = (sharedCnt1 == 'd3);
wire MaxPoolingDone_flag = MP_ImgBottomBoundReach_flag && MP_ImgRightBoundReach_flag;
wire L1_Done_flag = K1_mode && MaxPoolingDone_flag;


wire FlattenDone_flag = (sharedCnt1 == L1_MEM_SIZE - 1);
wire L2_Done_flag = K1_mode && FlattenDone_flag;

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

//MEM_ACCESS
parameter NO_ACCESS= 3'b000;
parameter L0_MEM0_ACCESS= 3'b001;
parameter L0_MEM1_ACCESS= 3'b010;
parameter L1_MEM0_ACCESS= 3'b011;
parameter L1_MEM1_ACCESS= 3'b100;
parameter L2_MEM_ACCESS= 3'b101;

//---------------------------------CONTROL_PATH---------------------------------//
//MAIN CTR
always @(posedge clk or negedge reset)
begin
    currentState <= reset ? CONV : nextState;
end

always @(*)
begin
    case(currentState)
        CONV:
            nextState = LocalConvDone_flag ? ReLU : CONV;
        ReLU:
            nextState = L0_WB;
        L0_WB:
            nextState = L0_Done_flag ? MAXPOOLING : CONV;
        MAXPOOLING:
            nextState = MaxPoolingDone_flag ? L1_WB : MAXPOOLING;
        L1_WB:
            nextState = L1_Done_flag ? FLATTEN : MAXPOOLING;
        FLATTEN:
            nextState = L2_WB;
        L2_WB:
            nextState = L2_Done_flag ? DONE : FLATTEN;
        DONE:
            nextState = DONE;
    endcase
end

//IMGColPTR,IMGROWPTR
wire[PTR_LENGTH-1:0] L0_ColPTR_action = (ConvDone_flag ? 'd1 : (CONV_ImgRightBoundReach_flag ? 'd1 : imgColPTR + 'd1) );
wire[PTR_LENGTH-1:0] L0_RowPTR_action = (ConvDone_flag ? 'd1 : (CONV_ImgRightBoundReach_flag ? imgRowPTR + 'd1 : imgRowPTR));

wire[PTR_LENGTH-1:0] L1_ColPTR_action = (MaxPoolingDone_flag ? 'd0 : MP_ImgRightBoundReach_flag ? 'd0 : imgColPTR + 'd2);
wire[PTR_LENGTH-1:0] L1_RowPTR_action = (MaxPoolingDone_flag ? 'd0 : MP_ImgBottomBoundReach_flag ? 'd0 : imgRowPTR + 'd2);
always @(posedge clk or negedge reset)
begin
    if(reset)
    begin
        imgColPTR <= 'd1;
        imgRowPTR <= 'd1;
    end
    else if(STATE_L0_WB)
    begin
        imgColPTR <= L0_Done_flag ? 'd0 : L0_ColPTR_action;
        imgRowPTR <= L0_Done_flag ? 'd0 : L0_RowPTR_action;
    end
    else if(STATE_L1_WB)
    begin
        imgColPTR <= L1_Done_flag ? 'd0 : L1_ColPTR_action;
        imgRowPTR <= L1_Done_flag ? 'd0 : L1_RowPTR_action;
    end
    else
    begin
        imgColPTR <=  imgColPTR;
        imgRowPTR <=  imgRowPTR ;
    end
end

//IMGoffsetPTR
always @(*)
begin
    if(STATE_CONV)
    begin
        case(sharedCnt1)
            'd0:
            begin
                imgOffsetColPTR = imgColPTR -'d1;
                imgOffsetRowPTR = imgRowPTR -'d1;
            end
            'd1:
            begin
                imgOffsetColPTR = imgColPTR;
                imgOffsetRowPTR = imgRowPTR - 'd1;
            end
            'd2:
            begin
                imgOffsetColPTR = imgColPTR + 'd1;
                imgOffsetRowPTR = imgRowPTR - 'd1;
            end
            'd3:
            begin
                imgOffsetColPTR = imgColPTR - 'd1;
                imgOffsetRowPTR = imgRowPTR;
            end
            'd4:
            begin
                imgOffsetColPTR = imgColPTR;
                imgOffsetRowPTR = imgRowPTR;
            end
            'd5:
            begin
                imgOffsetColPTR = imgColPTR+'d1;
                imgOffsetRowPTR = imgRowPTR;
            end
            'd6:
            begin
                imgOffsetColPTR = imgColPTR - 'd1;
                imgOffsetRowPTR = imgRowPTR + 'd1;
            end
            'd7:
            begin
                imgOffsetColPTR = imgColPTR;
                imgOffsetRowPTR = imgRowPTR + 'd1;
            end
            'd8:
            begin
                imgOffsetColPTR = imgColPTR + 'd1;
                imgOffsetRowPTR = imgRowPTR + 'd1;
            end
        endcase
    end
    else if(STATE_MAXPOOLING)
    begin
        case(sharedCnt1)
            'd0:
            begin
                imgOffsetColPTR = imgColPTR;
                imgOffsetRowPTR = imgRowPTR;
            end
            'd1:
            begin
                imgOffsetColPTR = imgColPTR+'d1;
                imgOffsetRowPTR = imgRowPTR;
            end
            'd2:
            begin
                imgOffsetColPTR = imgColPTR;
                imgOffsetRowPTR = imgRowPTR+'d1;
            end
            'd3:
            begin
                imgOffsetColPTR = imgColPTR + 'd1;
                imgOffsetRowPTR = imgRowPTR + 'd1;
            end
            default:
            begin
                imgOffsetColPTR = imgColPTR;
                imgOffsetRowPTR = imgRowPTR;
            end
        endcase
    end
    else
    begin

        begin
            imgOffsetColPTR = imgColPTR;
            imgOffsetRowPTR = imgRowPTR;
        end
    end
end

//sharedCnt1
always @(posedge clk or negedge reset)
begin
    if(reset)
    begin
        sharedCnt1 <= 'd0;
    end
    else if(STATE_CONV)
    begin
        sharedCnt1 <=  LocalConvDone_flag ?  'd0 : sharedCnt1 + 'd1;
    end
    else if(STATE_MAXPOOLING)
    begin
        sharedCnt1 <= LocalMaxPoolingDone_flag ? 'd0 : sharedCnt1 + 'd1;
    end
    else if(STATE_FLATTEN)
    begin
        sharedCnt1 <= FlattenDone_flag ? 'd0 : sharedCnt1 + 'd1;
    end
    else
    begin
        sharedCnt1 <= sharedCnt1;
    end
end

//sharedCnt2
always @(posedge clk or negedge reset)
begin
    if(reset)
    begin
        sharedCnt2 <= 'd0;
    end
    else if(STATE_FLATTEN)
    begin
        sharedCnt2 <= FlattenDone_flag ? 'd1 : sharedCnt2 + 'd2;
    end
    else
    begin
        sharedCnt2 <= sharedCnt2;
    end
end

//---------------------------------I/O--------------------------------//
wire[DATA_WIDTH-1:0] imgPixel_i;
wire[ADDR_WIDTH-1:0] convPixeladdr_o;
wire[ADDR_WIDTH-1:0] L0_WB_addr_o;

wire[DATA_WIDTH-1:0] MP_PixelValue_i;
wire[ADDR_WIDTH-1:0] MP_PixelAddr_o;
wire[ADDR_WIDTH-1:0] L1_WB_addr_o;

wire[ADDR_WIDTH-1:0] flatten_PixelAddr_o;
wire[DATA_WIDTH-1:0] flatten_PixelValue_i;
wire[ADDR_WIDTH-1:0] L2_WB_addr_o;

//busy
always @(posedge clk)
begin
    if(reset || STATE_DONE)
    begin
        busy <= 'd0;
    end
    else
    begin
        busy <= 'd1;
    end
end

//iaddr,idata
assign iaddr = convPixeladdr_o;
assign imgPixel_i = idata;


wire RdMem = STATE_FLATTEN || STATE_MAXPOOLING;
assign crd = RdMem ? 1 : 0;

//cdata_rd
assign MP_PixelValue_i = cdata_rd;
assign flatten_PixelValue_i = cdata_rd;
//caddr_rd
always @(*)
begin
    if(STATE_MAXPOOLING)
    begin
        caddr_rd = MP_PixelAddr_o;
    end
    else if(STATE_FLATTEN)
    begin
        caddr_rd = flatten_PixelAddr_o;
    end
    else
    begin
        caddr_rd = 'd0;
    end
end


//cdata_wr,caddr_wr
always @(*)
begin
    if(STATE_L0_WB)
    begin
        cdata_wr = ReLU_Result_rd;
        caddr_wr = L0_WB_addr_o;
    end
    else if(STATE_L1_WB)
    begin
        cdata_wr = MP_Result_rd;
        caddr_wr = L1_WB_addr_o;
    end
    else if(STATE_L2_WB)
    begin
        cdata_wr = Flatten_temp_rd;
        caddr_wr = L2_WB_addr_o;
    end
    else
    begin
        cdata_wr = 'd0;
        caddr_wr = 'd0;
    end
end

//csel
always @(*)
begin
    if(STATE_L0_WB || STATE_MAXPOOLING)
    begin
        csel = K0_mode ? L0_MEM0_ACCESS : L0_MEM1_ACCESS;
    end
    else if(STATE_L1_WB || STATE_FLATTEN)
    begin
        csel = K0_mode ? L1_MEM0_ACCESS : L1_MEM1_ACCESS;
    end
    else if(STATE_L2_WB)
    begin
        csel = L2_MEM_ACCESS ;
    end
    else
    begin
        csel = NO_ACCESS;
    end
end

//cwr
wire WriteMem = STATE_L0_WB || STATE_L1_WB || STATE_L2_WB;
assign cwr = WriteMem ? 1 : 0;

//----------------------------------DATAPATH------------------------------------//
//Using 1 multiplier, 1 adder, 1 comparator and 1 register
//Regiseter
reg[DATA_WIDTH-1:0] R1_rd;
reg[DATA_WIDTH-1:0] R1_wr;

wire signed[DATA_WIDTH-1:0] SerialMultiplierIN1;
wire signed[DATA_WIDTH-1:0] SerialMultiplierIN2;
reg signed[DATA_WIDTH-1:0] kernal_input;
wire signed[2*DATA_WIDTH-1:0] SerialMultiplierOUT = SerialMultiplierIN1 * SerialMultiplierIN2;


reg signed[DATA_WIDTH-1:0] AdderIN1;
reg signed[DATA_WIDTH-1:0] AdderIN2;
wire signed[DATA_WIDTH-1:0] AdderOUT = AdderIN1 + AdderIN2;

reg signed[DATA_WIDTH-1:0] ComparatorIN1;
reg signed[DATA_WIDTH-1:0] ComparatorIN2;
wire signed[DATA_WIDTH-1:0] Comparator_gt = ComparatorIN1 > ComparatorIN2;

wire[DATA_WIDTH-1:0] ReLU_Result_wr = Comparator_gt ? biased_Result : 'd0;
wire[DATA_WIDTH-1:0] biased_Result = AdderOUT;
wire[DATA_WIDTH-1:0] MP_Result_wr  = Comparator_gt ? MP_PixelValue_i : MP_Result_rd;

wire[DATA_WIDTH-1:0] MP_Result_rd = R1_rd;
wire[DATA_WIDTH-1:0] ReLU_Result_rd = R1_rd;
wire[DATA_WIDTH-1:0] Conv_Result_rd = R1_rd;
wire[DATA_WIDTH-1:0] Flatten_temp_rd = R1_rd;

//1 Register
always @(posedge clk or negedge reset)
begin
    R1_rd <= reset ? 'd0 : R1_wr;
end

always @(*)
begin
    if(STATE_CONV)
    begin
        R1_wr = AdderOUT;
    end
    else if(STATE_ReLU)
    begin
        R1_wr = ReLU_Result_wr;
    end
    else if(STATE_MAXPOOLING)
    begin
        R1_wr = MP_PixelValue_i;
    end
    else if(STATE_FLATTEN)
    begin
        R1_wr = flatten_PixelValue_i;
    end
    else
    begin
        R1_wr = R1_rd;
    end
end

//Multiplier
assign SerialMultiplierIN2 = kernal_input;
assign SerialMultiplierIN1 = ZeroPad_flag ? 'd0 : idata;

always @(*)
begin
    case(sharedCnt1)
        'd0:
            kernal_input = K0_mode ? KERNAL0_00 : KERNAL1_00;
        'd1:
            kernal_input = K0_mode ? KERNAL0_01 : KERNAL1_01;
        'd2:
            kernal_input = K0_mode ? KERNAL0_02 : KERNAL1_02;
        'd3:
            kernal_input = K0_mode ? KERNAL0_10 : KERNAL1_10;
        'd4:
            kernal_input = K0_mode ? KERNAL0_11 : KERNAL1_11;
        'd5:
            kernal_input = K0_mode ? KERNAL0_12 : KERNAL1_12;
        'd6:
            kernal_input = K0_mode ? KERNAL0_20 : KERNAL1_20;
        'd7:
            kernal_input = K0_mode ? KERNAL0_21 : KERNAL1_21;
        'd8:
            kernal_input = K0_mode ? KERNAL0_22 : KERNAL1_22;
        default:
            kernal_input = 'd0 ;
    endcase
end

//Adder
always @(*)
begin
    if(STATE_CONV)
    begin
        AdderIN1 = Conv_Result_rd;
        AdderIN2 = SerialMultiplierOUT[35:16];
    end
    else if(STATE_ReLU)
    begin
        AdderIN1 = Conv_Result_rd;
        AdderIN2 = BIAS;
    end
    else
    begin
        AdderIN1 = 'd0;
        AdderIN2 = 'd0;
    end
end

//Comparator Input
always @(*)
begin
    if(STATE_ReLU)
    begin
        ComparatorIN1 = biased_Result;
        ComparatorIN2 = 'd0;
    end
    else if(STATE_MAXPOOLING)
    begin
        ComparatorIN1 = MP_PixelValue_i;
        ComparatorIN2 = MP_Result_rd;
    end
    else
    begin
        ComparatorIN1 = 'd0;
        ComparatorIN2 = 'd0;
    end
end

endmodule
