# Frangi-Filter-Pytorch-Gpu
This is a pytorch implementation of [Original code in numpy](https://github.com/isyiming/Frangi-filter-based-Hessian)

I need to process large amount of XA videos. For an XA video of [34, 1, 512, 512] (Time,Channel,Height,Width), the original code will take 12 minutes by doing a for loop. I rewrite the code in pytorch with GPU support, now the same XA video takes 0.6 seconds.
