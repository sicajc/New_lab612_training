# Notes of LBP
## Algorithm
0. Given a $A=R^{128x128}$ image, with every pixel being 8 bits.
1. For every pixel belongs to $A_{ij}$ where, $127>i>0$ and $127>j>0$, perform LBP operation.
2. From the coordinate $(i,j)$ of every pixel,denote it as the center pixel, compare it with the surrounding 8 pixels.
3. If the value of the selected surrounding pixel is greater than the center pixel, accumulate the corresponding LBP value.
4. The LBP value is determined by the position of that surrounding pixel, which can be stored into a LUT.
5. After summing up the whole 8 surrounding LBP values, we get the final LBP value for the center pixel.
6. Repeat the process in row major order for all 128x128 pixels,yet ignores the outer surrounding pixels. Stops at $A_{(126,126)}$