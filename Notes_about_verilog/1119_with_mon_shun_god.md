# Verilog
2022/11/19
1. No matter what, when you are making an IP, the output must be register, so that your design will not contaminate other people's critical path.
2. The input should also be kept in a certain temporal register so that the critical path from other people's terrible design will not contaminate your circuit at first.
3. generate block as a certain type of way of writing s.t. it can has more hint to the coder.
```verilog
    generate
    for(i=0;i<n;i++) (loop_out) //This macro is important for specifying your design loop
    begin


    end
    end generate
```


4. Veriog code can actually exceute python script using $ command, for more detail, search for documentation or simply ask mon shun for src code.

5. A well UVM i.e. a good testbench should contain info about
<br />
   1. Where the error is
   2. Good coverage
   3. Final result
   4. What is the correct golden model s.t. you can make comparison.
