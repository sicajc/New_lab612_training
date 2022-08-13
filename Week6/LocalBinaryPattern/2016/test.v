

reg[3:0] cnt;
wire[1:0] clk_out;

always @(posedge clk)
begin
    if(rst)
    begin
        cnt <= 'd0;
    end
    else if(cnt == 'd10)
    begin
        cnt <= 'd0;
    end
    else
    begin
        cnt <= cnt + 'd1;
    end
end

assign clk_out = (cnt>='d6) ? 'd0 : clk;