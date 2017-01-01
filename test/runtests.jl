using Images
using CellSpeedEstimation
using Base.Test

# write your own tests here

# test grayscale image with 4 parallel lines
im = load("double_double_lines.png")

exact_slope = 51.714/201.000 # taken from corresponding svg file 
estimated_slope = slope_estimation(ROI_to_array(im),0,20,0,20)[1]

# for this test case we should have an error of < 2% (expected: 1.6%):
@test abs((exact_slope-estimated_slope)/exact_slope) < 0.02
