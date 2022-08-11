module CNT(rst,clk,cnt_rd1);
input wire rst;
input wire clk;
output [3:0] cnt_rd1;

reg[3:0] cnt_rd;
reg[3:0] cnt_wr;

//cnt coding style 1
always @(posedge clk)
begin
    cnt_rd <= cnt_wr;
end

always @(*)
begin
    if(rst)
    begin
        cnt_wr = 'd0;
    end
    else
    begin
        cnt_wr = cnt_rd + 'd1;
    end
end

assign cnt_rd1 = cnt_rd;

//cnt coding style 2
always @(posedge clk)
begin
    if(rst)
    begin
        cnt_rd <= 'd0;
    end
    else
    begin
        cnt_rd <= cnt_rd + 'd1;
    end
end

//cnt coding style 3
always @(posedge clk)
begin
    cnt_rd <= cnt_wr;
end

//c = a + b  ;
//d = c + d  ;
//f = e + c  ;



endmodule
