module DT(input 			clk,
          input			reset,
          output	done,
          output	sti_rd,
          output	[9:0]	sti_addr,
          input		[15:0]	sti_di,
          output	res_wr,
          output	res_rd,
          output	[13:0]	res_addr,
          output	[7:0]	res_do,
          input		[7:0]	res_di);
//CONSTANTS
parameter IMAGE_WIDTH   = 128 ;
parameter IMAGE_HEIGT   = 128;
parameter STI_ROM_DEPTH = 1024;
parameter RES_RAM_DEPTH = 16384;
parameter DATA_WIDTH = 8;
parameter PTR_LENGTH = 8;

//States
parameter IDLE = 'd0;
parameter DET_OBJECT = 'd1;
parameter FORWARD_WINDOWING = 'd2;
parameter FORWARD_WINDOW_WB = 'd3;
parameter BACKWARD_WINDOWING= 'd4;
parameter BACKWARD_WINDOW_WB= 'd5;
parameter DONE= 'd6;

//States Indicators
wire STATE_IDLE                 =                  currentState == IDLE ;
wire STATE_DET_OBJECT           =                  currentState == DET_OBJECT ;
wire STATE_FORWARD_WINDOWING    =                  currentState == FORWARD_WINDOWING ;
wire STATE_FORWARD_WINDOW_WB    =                  currentState == FORWARD_WINDOW_WB ;
wire STATE_BACKWARD_WINDOWING   =                  currentState == BACKWARD_WINDOWING ;
wire STATE_BACKWARD_WINDOW_WB   =                  currentState == BACKWARD_WINDOW_WB ;
wire STATE_DONE                 =                  currentState == DONE ;
wire MODE_fowardWindow          =                  imgProcessMode == 'd1;
wire MODE_backwardWindow        =                  imgProcessMode == 'd0;

//CTR
reg[2:0] currentState,nextState;
reg[9:0] stiROM_addrCnt;
reg[14:0] resRAM_addrCnt;
reg[4:0] pixelOffsetCnt;
wire[PTR_LENGTH-1:0] imgColPTR = resRAM_addrCnt[6:0];
wire[PTR_LENGTH-1:0] imgRowPTR = resRAM_addrCnt[13:7];
reg imgProcessMode;

//Flags
wire isObject_flag                          = STATE_DET_OBJECT ? (imgPixel == 'd1) : 0;
wire forwardWindowingDone_flag              = STATE_DET_OBJECT ? (resRAM_addrCnt == RES_RAM_DEPTH) : 0;
wire backwardWindowingDone_flag             = STATE_DET_OBJECT ? (resRAM_addrCnt == 'd0) : 0;
wire localForwardWindowDone_flag            = STATE_FORWARD_WINDOWING  ?  localImageReadDone_flag : 0    ;
wire localBackwardWindowDone_flag           = STATE_BACKWARD_WINDOWING ?  localImageReadDone_flag : 0    ;

//Variables
wire[2:0] ForwardWindowingDetObject = (isObject_flag ? FORWARD_WINDOWING : FORWARD_WINDOW_WB);
wire[2:0] BackwardWindowingDetObject = (isObject_flag ? BACKWARD_WINDOWING : BACKWARD_WINDOW_WB);

wire[2:0] ForwardWindowing = (forwardWindowingDone_flag ? DET_OBJECT :  ForwardWindowingDetObject);
wire[2:0] BackwardWindowing = (backwardWindowingDone_flag ? DONE : BackwardWindowingDetObject);

//MAINCTR
always @(posedge clk or negedge reset)
begin
    currentState <= !reset ? IDLE : nextState;
end

always @(*)
begin
    case(currentState)
        IDLE:
        begin
            nextState = DET_OBJECT;
        end
        DET_OBJECT:
        begin
            nextState = imgProcessMode ? ForwardWindowing : BackwardWindowing;
        end
        FORWARD_WINDOWING:
        begin
            nextState = localForwardWindowDone_flag ? FORWARD_WINDOW_WB : FORWARD_WINDOWING;
        end
        FORWARD_WINDOW_WB:
        begin
            nextState = DET_OBJECT;
        end
        BACKWARD_WINDOWING:
        begin
            nextState = localBackwardWindowDone_flag ? BACKWARD_WINDOW_WB : BACKWARD_WINDOWING;
        end
        BACKWARD_WINDOW_WB:
        begin
            nextState = DET_OBJECT;
        end
        DONE:
        begin
            nextState = DONE;
        end
        default:
        begin
            nextState = IDLE;
        end
    endcase
end
//PixelOffsetCnt
wire ReadOffsetPixel = STATE_BACKWARD_WINDOWING || FORWARD_WINDOWING;
wire localImageReadDone_flag = (pixelOffsetCnt == 'd3) ;

always @(posedge clk or negedge reset)
begin
    if(!reset)
    begin
        pixelOffsetCnt <= 'd0;
    end
    else if(ReadOffsetPixel)
    begin
        pixelOffsetCnt <= localImageReadDone_flag ? 'd0 : 'd1 + pixelOffsetCnt;
    end
    else
    begin
        pixelOffsetCnt <= pixelOffsetCnt;
    end
end

//LOCAL_IMG_OFFSET_PTRs
reg[6:0] localImgOffsetColPTR;
reg[6:0] localImgOffsetRowPTR;
always @(*)
begin
    case(currentState)
        FORWARD_WINDOWING:
        begin
            case(pixelOffsetCnt)
                'd0:
                begin
                    {localImgOffsetRowPTR,localImgOffsetColPTR} = {imgRowPTR,imgColPTR-7'd1};
                end
                'd1:
                begin
                    {localImgOffsetRowPTR,localImgOffsetColPTR} = {imgRowPTR-7'd1,imgColPTR+7'd1};
                end
                'd2:
                begin
                    {localImgOffsetRowPTR,localImgOffsetColPTR} = {imgRowPTR-7'd1,imgColPTR};
                end
                'd3:
                begin
                    {localImgOffsetRowPTR,localImgOffsetColPTR} = {imgRowPTR-7'd1,imgColPTR-7'd1};
                end
                default:
                begin
                    {localImgOffsetRowPTR,localImgOffsetColPTR} = {imgRowPTR,imgColPTR};
                end
            endcase
        end
        BACKWARD_WINDOWING:
        begin
            case(pixelOffsetCnt)
                'd0:
                begin
                    {localImgOffsetRowPTR,localImgOffsetColPTR} = {imgRowPTR,imgColPTR + 7'd1};
                end
                'd1:
                begin
                    {localImgOffsetRowPTR,localImgOffsetColPTR} = {imgRowPTR + 7'd1,imgColPTR - 7'd1};
                end
                'd2:
                begin
                    {localImgOffsetRowPTR,localImgOffsetColPTR} = {imgRowPTR - 7'd1,imgColPTR};
                end
                'd3:
                begin
                    {localImgOffsetRowPTR,localImgOffsetColPTR} = {imgRowPTR + 7'd1,imgColPTR + 7'd1};
                end
                default:
                begin
                    {localImgOffsetRowPTR,localImgOffsetColPTR} = {imgRowPTR,imgColPTR};
                end
            endcase
        end
        default:
        begin
            {localImgOffsetRowPTR,localImgOffsetColPTR} = {imgRowPTR,imgColPTR};
        end
    endcase
end

// I/O
wire[9:0]  imgPixelAddr_o;
wire[15:0] imgPixelArray_i;
wire[7:0]  pixelReadData_i;

wire[13:0] pixelWriteAddr_o;
wire[13:0] pixelReadAddr_o;
wire[7:0]  pixelWriteData_o;

wire ReadLocalPixel = STATE_FORWARD_WINDOWING||STATE_BACKWARD_WINDOWING;
wire WriteLocalPixel = STATE_BACKWARD_WINDOW_WB||STATE_FORWARD_WINDOW_WB;

//sti_rd,sti_addr,sti_di
assign {sti_rd,sti_addr,imgPixelArray_i} = STATE_DET_OBJECT ? {1'b1,imgPixelAddr_o,sti_di} : {1'b0,10'd0,16'd0};

//res_addr , res_wr , res_rd ,res_do , res_di
assign res_addr =         WriteLocalPixel ? pixelWriteAddr_o : (ReadLocalPixel ? pixelReadAddr_o : 'd0);
assign res_wr   =         WriteLocalPixel ? 1 : 0;
assign res_do   =         WriteLocalPixel ? pixelWriteData_o : 'd0;
assign res_rd          =  ReadLocalPixel  ? 1 : 0;
assign pixelReadData_i =  ReadLocalPixel ? res_di : 'd0;

//sti_ROM_AddrCnt & resRAM_addrCnt
wire forwardWindow_nextPixelArray_flag = (stiROM_addrCnt[3:0] == 4'b1111);
wire backwardWindow_nextPixelArray_flag = (stiROM_addrCnt[3:0] == 4'b0000);

always @(posedge clk or negedge reset)
begin
    if(!reset)
    begin
        stiROM_addrCnt <= 'd0;
        resRAM_addrCnt <= 'd0;
    end
    else if(STATE_FORWARD_WINDOW_WB)
    begin
        stiROM_addrCnt <= forwardWindowingDone_flag ? (STI_ROM_DEPTH-1) : (forwardWindow_nextPixelArray_flag ? stiROM_addrCnt + 'd1 : stiROM_addrCnt);
        resRAM_addrCnt <= forwardWindowingDone_flag ? (RES_RAM_DEPTH-1) : (WriteLocalPixel ? resRAM_addrCnt + 'd1 :resRAM_addrCnt) ;
    end
    else if(STATE_BACKWARD_WINDOW_WB)
    begin
        stiROM_addrCnt <=backwardWindowingDone_flag ? 'd0 :(backwardWindow_nextPixelArray_flag ? stiROM_addrCnt - 'd1 : stiROM_addrCnt);
        resRAM_addrCnt <=backwardWindowingDone_flag ? 'd0 : WriteLocalPixel ? resRAM_addrCnt - 'd1 : resRAM_addrCnt;
    end
    else
    begin
        stiROM_addrCnt <= stiROM_addrCnt;
        resRAM_addrCnt <= resRAM_addrCnt;
    end
end

//image process mode
always @(posedge clk or negedge reset)
begin
    imgProcessMode <= !reset ? 'd1 : forwardWindowingDone_flag ? 'd0 : imgProcessMode;
end

//-------------------------DATAPATH----------------------//
reg[DATA_WIDTH-1:0] minPixelTempReg;
wire[DATA_WIDTH-1:0] forwardwindowingResult,backwardwindowingResult;
assign forwardwindowingResult  = STATE_FORWARD_WINDOW_WB  ? minPixelTempReg + 'd1 : 'd0;
assign backwardwindowingResult = STATE_BACKWARD_WINDOW_WB ? minPixelTempReg : 'd0;

wire pixel_value_ltTemp_flag = pixelReadData_i < minPixelTempReg;

always @(posedge clk or negedge reset)
begin
    if(!reset)
    begin
        minPixelTempReg <= 'd255;
    end
    else if(ReadLocalPixel)
    begin
        minPixelTempReg <= pixel_value_ltTemp_flag ? pixelReadData_i : minPixelTempReg;
    end
    else if(WriteLocalPixel)
    begin
        minPixelTempReg <= 'd255;
    end
    else
    begin
        minPixelTempReg <= minPixelTempReg;
    end
end

assign pixelWriteData_o = MODE_fowardWindow ? forwardwindowingResult : backwardwindowingResult;


endmodule
