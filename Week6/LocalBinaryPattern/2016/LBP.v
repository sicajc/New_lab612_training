
`timescale 1ns/10ps
module LBP ( clk, reset, gray_addr, gray_req, gray_ready, gray_data, lbp_addr, lbp_valid, lbp_data, finish);
input   	    clk;
input   	    reset;
output  [13:0] 	gray_addr;
output         	gray_req;
input   	    gray_ready;
input   [7:0] 	gray_data;
output  [13:0] 	lbp_addr;
output  	    lbp_valid;
output  [7:0] 	lbp_data;
output  	    finish;
//====================================================================//
//CONSTANTS
parameter PTR_LENGTH = 7;
parameter DATA_WIDTH = 8;
parameter ADDR_WIDTH = 14;
parameter IMG_WIDTH = 128;
//state registers, next state logic
reg[1:0] currentState, nextState;

//states
parameter RD_PIXEL = 'd0;
parameter LBP =  'd1;
parameter DONE = 'd2;

//states indicators
wire STATE_RD_PIXEL  = currentState == RD_PIXEL;
wire STATE_LBP       = currentState == LBP;
wire STATE_DONE      = currentState == DONE;

//cnt
reg[3:0] cnt;

//PTRS
reg[PTR_LENGTH - 1 : 0] imgColPTR,imgRowPTR;
reg[PTR_LENGTH - 1 : 0] imgColOffsetPTR,imgRowOffsetPTR;

//flags
wire ImgRightBoundReach_flag  = (imgColPTR == IMG_WIDTH - 2) ;
wire ImgBottomBoundReach_flag = (imgRowPTR == IMG_WIDTH - 2) ;
wire Local_LBP_done_WB_flag = (cnt == 'd7);
wire LBP_done_flag = ImgRightBoundReach_flag && ImgBottomBoundReach_flag ;


//InterConnections
wire signed[DATA_WIDTH-1:0] pixelValue_i;
wire[DATA_WIDTH-1:0] shift_weighted_result;
wire local_threshold;
wire threshold_compared_gt;

//Registers
reg[DATA_WIDTH-1:0] lbp_tempReg_rd;
reg signed[DATA_WIDTH-1:0] centerPixelReg;
wire[DATA_WIDTH-1:0] lbp_temp_wr;


//State register
always @(posedge clk or posedge reset)
begin
    currentState <= reset ? RD_PIXEL : nextState;
end

always @(*)
begin
    case(currentState)
        RD_PIXEL:
            nextState = LBP;
        LBP:
            nextState = (Local_LBP_done_WB_flag ? (LBP_done_flag ? DONE : RD_PIXEL) : LBP);
        DONE:
            nextState = DONE;
        default:
            nextState = RD_PIXEL;
    endcase
end

//PTRs
always @(posedge clk or posedge reset)
begin
    if(reset)
    begin
        {imgRowPTR,imgColPTR} <= {7'd1,7'd1};
    end
    else if(STATE_LBP)
    begin
        imgRowPTR <=Local_LBP_done_WB_flag ? (ImgRightBoundReach_flag ? (imgRowPTR + 'd1):(imgRowPTR)) : (imgRowPTR) ;
        imgColPTR <=Local_LBP_done_WB_flag ? (ImgRightBoundReach_flag ? ('d1) : imgColPTR) : (imgColPTR + 'd1);
    end
    else
    begin
        {imgRowPTR,imgColPTR} <= {imgRowPTR,imgColPTR};
    end
end

always @(*)
begin
    case(cnt)
        3'd0:
            {imgRowOffsetPTR,imgColOffsetPTR} <= {imgRowPTR - 7'd1,imgColPTR - 7'd1};
        3'd1:
            {imgRowOffsetPTR,imgColOffsetPTR} <= {imgRowPTR - 7'd1,imgColPTR};
        3'd2:
            {imgRowOffsetPTR,imgColOffsetPTR} <= {imgRowPTR - 7'd1,imgColPTR + 7'd1};
        3'd3:
            {imgRowOffsetPTR,imgColOffsetPTR} <= {imgRowPTR,imgColPTR-7'd1};
        3'd4:
            {imgRowOffsetPTR,imgColOffsetPTR} <= {imgRowPTR,imgColPTR + 7'd1};
        3'd5:
            {imgRowOffsetPTR,imgColOffsetPTR} <= {imgRowPTR + 7'd1,imgColPTR - 7'd1};
        3'd6:
            {imgRowOffsetPTR,imgColOffsetPTR} <= {imgRowPTR + 7'd1,imgColPTR};
        3'd7:
            {imgRowOffsetPTR,imgColOffsetPTR} <= {imgRowPTR + 7'd1,imgColPTR + 7'd1};
        default:
            {imgRowOffsetPTR,imgColOffsetPTR} <= {imgRowPTR,imgColPTR};
    endcase
end

//cnt
always @(posedge clk or posedge reset)
begin
    if(reset)
    begin
        cnt <= 'd0;
    end
    else if(STATE_LBP && gray_ready)
    begin
        cnt <= Local_LBP_done_WB_flag ? 'd0 : (gray_ready ? (cnt + 'd1) : cnt);
    end
    else
    begin
        cnt <= cnt;
    end
end

//I/O
assign pixelValue_i = gray_data;
assign gray_addr = STATE_RD_PIXEL ?  {imgRowPTR,imgColPTR}: {imgRowOffsetPTR,imgColOffsetPTR};
assign gray_req  = STATE_LBP;
assign lbp_addr  = gray_addr;
assign lbp_data  = lbp_temp_wr;
assign lbp_valid = Local_LBP_done_WB_flag;
assign finish = STATE_DONE;


//----------------------------DP------------------------//
always @(posedge clk or posedge reset)
begin
    lbp_tempReg_rd <= reset ? 'd0 : (Local_LBP_done_WB_flag ?  'd0 : lbp_temp_wr);
end
assign lbp_temp_wr = lbp_tempReg_rd + shift_weighted_result;

always @(posedge clk or posedge reset)
begin
    centerPixelReg <= reset ? 'd0 : (STATE_RD_PIXEL ?  gray_data : centerPixelReg);
end

assign threshold_compared_gt = ( pixelValue_i >= centerPixelReg);
assign local_threshold     =   threshold_compared_gt ? 1 : 0;
assign shift_weighted_result = local_threshold << cnt;



endmodule
