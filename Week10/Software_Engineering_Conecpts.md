# Software Engineering
## Introduction
You must try to develop your software and hardware in a steady way s.t. your project amd program can be less error prone. We usually try to divide a certain project into several phases to give ourself a clear view about what to do and what should not be done during a certain phase. One can only move from a certain phase to another only if they reach the requirements of different phases. The cost of crossing through the layer casually will lead to catastrophic results and lead to a waste of time and more overWork. So! <br />
>NEVER TRY TO SIMPLY CODE AND TEST YOUR OWN CODE! AND WRITE TERRIBLE DOCUMENTATION! OR SIMPLY DOES NOT WRITE ANY DOCUMENTATION!!OMG. IT CAUSES FURTHER PROBLEM TO OTHER USERS AND TEAM MEMBERS! ALSO IT GET EXTREMELY HARD TO DEBUG YOUR SPAGHETTI CODE.

## MODELS
### WaterFall models
```mermaid

    graph TD
    A[Problem Specification] -->A1{Works?}-->|T|B[Design];
    A1 --> |F|A

    B[Design] -->B1{Works?}-->|T|C[Implementaion];
    B1 --> |F|B

    C[Implementation] -->C1{Works?}-->|T|D[System Integration];
    C1 --> |F|C

    D[Implementation] -->D1{Works?}-->|T|E[Deployment];
    D1 --> |F|D

    E[Implementation] -->E1{Works?}-->|T|F[Maintainence];
    E1 --> |F|E

```
