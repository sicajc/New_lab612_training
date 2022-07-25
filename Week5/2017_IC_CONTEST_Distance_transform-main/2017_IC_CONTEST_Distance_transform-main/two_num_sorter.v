module two_num_sorter#(parameter DATA_WIDTH)
                      (a,
                       b,
                       min,
                       max);

    input [DATA_WIDTH-1:0] a;
    input [DATA_WIDTH-1:0]b;
    output [DATA_WIDTH-1:0] min;
    output [DATA_WIDTH-1:0] max;

    assign max = a > b ? a : b;
    assign min = a < b ? a : b;

endmodule
