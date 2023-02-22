
`timescale 1ns/10ps
`define C2Q 5
module LBP ( clk, reset, gray_addr, gray_req, gray_ready, gray_data, lbp_addr, lbp_valid, lbp_data, finish);
input   	        clk;
input   	        reset;
output  reg[13:0] 	gray_addr;
output  reg       	gray_req;
input   	        gray_ready;
input   [7:0] 	    gray_data;
output  reg[13:0] 	lbp_addr;
output  reg	        lbp_valid;
output  reg[7:0] 	lbp_data;
output  reg	        finish;
//====================================================================//
//==================
//  PARAMETERS
//==================
parameter BYTE = 8 ;
parameter WORD = 8 ;
parameter PTR_WIDTH = BYTE;
parameter CNT_WIDTH = BYTE;
parameter IMG_WIDTH = 128;

//==================
//  stateS
//==================
//L1 FSM
localparam IDLE                      = 6'b000001 ;
localparam RD_CENTER                 = 6'b000010 ;
localparam RD_SURROUND_PIXEL         = 6'b000100 ;
localparam ACC                       = 6'b001000 ;
localparam WB                        = 6'b010000 ;
localparam DONE                      = 6'b100000 ;

//================================
//  MAIN CTR
//================================
reg [7:0] l1_curState,l1_nxtState;

//================================
//  state_indicators
//================================
wire state_IDLE                          =  l1_curState [0];
wire state_RD_CENTER                     =  l1_curState [1];
wire state_RD_SURROUND_PIXEL             =  l1_curState [2];
wire state_ACC                           =  l1_curState [3];
wire state_WB                            =  l1_curState [4];
wire state_DONE                          =  l1_curState [5];

//================================
//  Image PTRS
//================================
reg[PTR_WIDTH-1:0] row_ptr;
reg[PTR_WIDTH-1:0] col_ptr;

//================================
//  OFFSET PTRS
//================================
reg[PTR_WIDTH-1:0] offset_row;
reg[PTR_WIDTH-1:0] offset_col;

//================================
//  OFFSET CNT
//================================
reg[PTR_WIDTH-1:0] offset_cnt;

//================================
//  LUT
//================================
reg[WORD-1:0] surrounding_pixel_value;

//=======================================
//  Registers
//=======================================
reg[WORD-1:0] acc_ff;
reg[WORD-1:0] center_pixel_ff;
reg[WORD-1:0] surrounding_pixel_ff;

//================================
//  CONTROL FLAGS
//================================
wire imgRightBoundReach_f = (col_ptr == (IMG_WIDTH-2));
wire imgBottomBoundReach_f = (row_ptr == (IMG_WIDTH-2));


wire acc_done_f = offset_cnt == 7;
wire lbp_done_f = imgRightBoundReach_f && imgBottomBoundReach_f;


//================================================================
//  MAIN DESIGN
//================================================================
//================================
//  FSM
//================================
always @(posedge clk or posedge reset)
begin:L1_FSM
    //synopsys_translate_off
    #`C2Q;
    //synopsys_translate_on
    if(reset)
    begin
        l1_curState <=  IDLE;
    end
    else
    begin
        l1_curState <=  l1_nxtState;
    end
end

always @(*)
begin:L1_FSM_NXT
    case(l1_curState)
        IDLE:
        begin
            l1_nxtState =  gray_ready ?  RD_CENTER : IDLE;
            finish = 0;
        end
        RD_CENTER:
        begin
            l1_nxtState = RD_SURROUND_PIXEL;
            finish = 0;
        end
        RD_SURROUND_PIXEL:
        begin
            l1_nxtState = ACC;
            finish = 0;
        end
        ACC:
        begin
            l1_nxtState = acc_done_f ? WB : RD_SURROUND_PIXEL ;
            finish = 0;
        end
        WB:
        begin
            l1_nxtState = lbp_done_f ? DONE : RD_CENTER;
            finish = 0;
        end
        DONE:
        begin
            l1_nxtState = IDLE;
            finish = 1;
        end
        default:
        begin
            l1_nxtState = IDLE;
            finish = 0;
        end
    endcase
end
//================================
//  I/O
//================================
always @(*)
begin: LBP_MEM_CTR
    lbp_addr  = state_WB ? row_ptr*(IMG_WIDTH) + col_ptr : 'dz;
    lbp_data  = acc_ff;
    lbp_valid = state_WB;
end

always @(*)
begin: GRAY_MEM_CTR
    if(state_RD_CENTER)
    begin
        gray_addr = row_ptr * (IMG_WIDTH) + col_ptr;
    end
    else if(state_RD_SURROUND_PIXEL)
    begin
        gray_addr = offset_row * (IMG_WIDTH) + offset_col;
    end
    else
    begin
        gray_addr = 'dz;
    end

    gray_req  = state_RD_CENTER || state_RD_SURROUND_PIXEL;
end

//================================
//  COUNTERS
//================================
always @(posedge clk or posedge reset)
begin: OFFFSET_CNT
    //synopsys_translate_off
    #`C2Q;
    //synopsys_translate_on
    if(reset)
    begin
        offset_cnt <= 0;
    end
    else if(state_RD_CENTER)
    begin
        offset_cnt <= 0;
    end
    else if(state_ACC)
    begin
        offset_cnt <= offset_cnt + 1;
    end
    else
    begin
        offset_cnt <= offset_cnt;
    end
end

//================================
//  POINTERS
//================================
always @(posedge clk or posedge reset)
begin: PTRS
    //synopsys_translate_off
    #`C2Q;
    //synopsys_translate_on
    if(reset)
    begin
        row_ptr <= 1;
        col_ptr <= 1;
    end
    else if(state_WB)
    begin
        row_ptr <= imgRightBoundReach_f ? row_ptr+1 : row_ptr;
        col_ptr <= imgRightBoundReach_f ? 1 : col_ptr + 1;
    end
    else
    begin
        row_ptr <= row_ptr;
        col_ptr <= col_ptr;
    end
end
//================================
//  DET_SURROUNDING_PIXEL_VALUE
//================================
always @(*)
begin: DET_PIXEL_VALUE
    case(offset_cnt)
        'd0:
        begin
            surrounding_pixel_value = 8'b0000_0001;
            offset_row = row_ptr - 1;
            offset_col = col_ptr - 1;
        end
        'd1:
        begin
            surrounding_pixel_value = 8'b0000_0010;
            offset_row = row_ptr - 1;
            offset_col = col_ptr;
        end
        'd2:
        begin
            surrounding_pixel_value = 8'b0000_0100;
            offset_row = row_ptr - 1;
            offset_col = col_ptr + 1;
        end
        'd3:
        begin
            surrounding_pixel_value = 8'b0000_1000;
            offset_row = row_ptr;
            offset_col = col_ptr - 1;
        end
        'd4:
        begin
            surrounding_pixel_value = 8'b0001_0000;
            offset_row = row_ptr;
            offset_col = col_ptr+1;
        end
        'd5:
        begin
            surrounding_pixel_value = 8'b0010_0000;
            offset_row = row_ptr+1;
            offset_col = col_ptr-1;
        end
        'd6:
        begin
            surrounding_pixel_value = 8'b0100_0000;
            offset_row = row_ptr+1;
            offset_col = col_ptr;
        end
        'd7:
        begin
            surrounding_pixel_value = 8'b1000_0000;
            offset_row = row_ptr+1;
            offset_col = col_ptr+1;
        end
        default:
        begin
            surrounding_pixel_value = 8'b0000_0001;
            offset_row = row_ptr;
            offset_col = col_ptr;
        end
    endcase
end

//================================
//  RD_GARY_DATA
//================================
always @(posedge clk or posedge reset)
begin: RD_GRAY_DATA
    //synopsys_translate_off
    #`C2Q;
    //synopsys_translate_on
    if(reset)
    begin
        center_pixel_ff <= 8'd0;
        surrounding_pixel_ff <= 8'd0;
    end
    else if(state_RD_CENTER)
    begin
        center_pixel_ff <= gray_data;
        surrounding_pixel_ff <= surrounding_pixel_ff;
    end
    else if(state_RD_SURROUND_PIXEL)
    begin
        center_pixel_ff <= center_pixel_ff;
        surrounding_pixel_ff <= gray_data;
    end
    else
    begin
        center_pixel_ff <= center_pixel_ff;
        surrounding_pixel_ff <= surrounding_pixel_ff;
    end
end

//================================
//  ACCUMULATION
//================================
always @(posedge clk or posedge reset)
begin: ACCUMULATION
    //synopsys_translate_off
    #`C2Q;
    //synopsys_translate_on
    if(reset)
    begin
        acc_ff <= 8'd0;
    end
    else if(state_ACC)
    begin
        acc_ff <= (surrounding_pixel_ff >= center_pixel_ff) ? (surrounding_pixel_value + acc_ff) : acc_ff;
    end
    else if(state_RD_CENTER)
    begin
        acc_ff <= 8'd0;
    end
    else
    begin
        acc_ff <= acc_ff;
    end
end

endmodule
