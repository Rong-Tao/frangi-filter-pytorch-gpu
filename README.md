# Frangi-Filter-Pytorch-Gpu
This is a pytorch implementation of [Original code in numpy](https://github.com/isyiming/Frangi-filter-based-Hessian)

I need to process large amount of XA videos. For a greyscale XA video of [34, 1, 512, 512] (Time,Channel,Height,Width), the original code will take 12 minutes by doing a for loop. I rewrite the code in pytorch with GPU support, now the same XA video takes 0.6 seconds.

The original code would take me a year to finish all my data, now it takes 2 days, 600 hundred times faster!

## Get Started
    FrangiFilter2D(I)
    Input: [B,1,H,W] the image to be filtered
    Output: [B,1,H,W] 
