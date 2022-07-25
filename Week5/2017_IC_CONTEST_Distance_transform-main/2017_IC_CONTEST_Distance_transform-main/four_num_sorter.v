`include "two_num_sorter.v"
module four_num_sorter#(parameter DATA_WIDTH)
                       (a,
                        b,
                        c,
                        d,
                        min);

    input [DATA_WIDTH-1:0] a;
    input [DATA_WIDTH-1:0] b;
    input [DATA_WIDTH-1:0] c;
    input [DATA_WIDTH-1:0] d;
    output [DATA_WIDTH-1:0] min;

    wire[DATA_WIDTH-1:0] max1_1,max2_1,min1_1,min2_1;
    wire[DATA_WIDTH-1:0] max1_2,min1_2;


    two_num_sorter #(DATA_WIDTH) SORTER1(.a(a),.b(b),.min(min1_1),.max(max1_1));
    two_num_sorter #(DATA_WIDTH) SORTER2(.a(c),.b(d),.min(min2_1),.max(max2_1));

    two_num_sorter #(DATA_WIDTH) SORTER3(.a(min1_1),.b(min2_1),.min(min1_2),.max(max1_2));

    assign min = min1_2;



endmodule
