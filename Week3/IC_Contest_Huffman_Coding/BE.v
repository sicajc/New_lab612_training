module BE
#(parameter N= 7)
(input[N-1:0] a,input[N-1:0] b,output[N-1:0] min,output[N-1:0] max);

assign {min,max} = (a>b) ? {b,a} : {a,b};

endmodule
