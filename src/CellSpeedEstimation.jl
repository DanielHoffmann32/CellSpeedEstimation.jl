module CellSpeedEstimation

# package code goes here
using DataFrames,
GLM,
Images,
StatPlots,
StatsBase

export column_correlations,
estimate_velocities,
invert_grayscale_image,
ROI_to_array,
slope_estimation

"""
Input: grayscale image

Output: inverted image
"""
function invert_grayscale_image(im::Image)
    return maximum(im)-im
end

"""
Input:

    - grayscale image

    - clip array in pixels (4 pos. integers, clockwise from top), default=[0,0,0,0]

    - flag wether to invert (default: true)

Output:

    - 2D float array corresponding to image
"""
function ROI_to_array(im::Image,
                      clip::Array{Int64,1}=[0, 0, 0, 0],
                      invert::Bool=true)
    if invert==true
        im = invert_grayscale_image(im)
    end
    A = convert(Array{Float64,2},im)
    n_rows = size(A)[1]
    n_cols = size(A)[2]
    if (clip[1]+clip[3]>=n_rows) || (clip[2]+clip[4]>=n_cols)
        error("clipped too much of ROI")
    end
    A[(1+clip[1]):(n_rows-clip[3]),(1+clip[4]):(n_cols-clip[2])]
end

"""
Input:

- inverted ROI array (movement of cells is horizontal)

- column difference in pixels

- minimum row difference

- maximum row difference

- whether the shadows are going from (top) left to (bottom) right (left_to_right=true) or from top right to bottom left (left_to_right=false)

- should plots been made? (default: false)

Output:

- array of cross correlations between ROI columns x and x + delta_x for lags between minimum and maximum row difference

"""
function column_correlations(
    iroia::Array{Float64,2}, #inverted ROI array
    delta_x::Int64; #column difference
    dy_min::Int64=0, #minimum row difference
    dy_max::Int64=10, #maximum row difference
    left_to_right::Bool=true, 
    plots::Bool=false #if true generates a heatmap and a plot showing the correlation between columns
    )
    
    x_start=0.
    x_end=0.
    n_dy=dy_max-dy_min+1
    
    if left_to_right == true
        x_start = 1 #leftmost position
        x_end = size(iroia)[2]-delta_x #rightmost position to include
        step = 1
    else
        x_start = size(iroia)[2] #rightmost
        x_end = delta_x+1 #leftmost
        delta_x *= -1
        step = -1
    end
    
    cors = zeros(size(iroia)[2]-abs(delta_x), n_dy)

    # This is the core: 
    # we compute the cross-correlations between columns of the ROI as basis for the determination of 
    # slopes of the cell shadows. If a cell has a linear shadow, it will turn up as a correlation of 
    # the gray values between columns:
    # *---
    # -*--
    # --*-
    # ---*
    # In this case we have a correlation along the shadow (*) between pixel i,j and i+1 and j+1, etc.
    # This means that the shadow has an angle of 45° (evaluation in function slope_estimation)
    
    i=1
    for x in x_start:step:x_end
        cors[i,:]=crosscor(iroia[:,x],iroia[:,x+delta_x],dy_min:dy_max)
        i+=1
    end
    
    if plots == true
        ci = 1.96*map(j->sem(cors[:,j]),1:n_dy) #95% CI under assumption of normality
        y = mean(cors,1)[1,:]
        p = plot(dy_min:dy_max, y, err=(ci,ci), lab="", xticks=dy_min:dy_max)
        xaxis!("\\Delta y")
        yaxis!("mean cross corr. between cols. (i,i+$delta_x)")

        hm = heatmap(cors)
        xaxis!("\\Delta y")
        yaxis!("image column")

        cors, hm, p
    else
        cors
    end
end

"""
Input:

- inverted ROI array (movement of cells is horizontal)

- min. delta_x used for computing cross-correlations between ROI columns

- max. delta_x 

- min. delta_y

- max. delta_y

- should peaks be interpolated by a parabolic approximation (default: true)

- should plots be drawn of delta_y vs. delta_x (default: false)

Output:

- slope beta0_Lr for the fit of a linear model to the data interpreted from left to right

- slope beta0_rL ... right to left

- standard error of beta0_Lr

- standard error of beta0_rL

- if plots==true: two plots, one for left-to-right, one for right-to-left

"""
function slope_estimation( #without peak interpolation
    iroia::Array{Float64,2}, #inverted ROI array
    delta_x_min::Int64,
    delta_x_max::Int64,
    delta_y_min::Int64=0,
    delta_y_max::Int64=10,
    plots::Bool=false
    )

    n_dx = delta_x_max-delta_x_min+1

    beta0 = zeros(2) #slopes of shadows for left-to-right and right-to-left interpretations
    beta0_err = zeros(2) #errors of the two slopes
    p = Array{Any}(2) #plots
    
    peak_pos = zeros(n_dx,2)
    
    i = 1
    for left_to_right in [true, false]
        for delta_x in delta_x_min:delta_x_max
            
            # determine correlations between ROI columns for a range of lags in both directions
            cors = column_correlations(
                iroia, 
                delta_x, 
                dy_min=delta_y_min, 
                dy_max=delta_y_max, 
                left_to_right=left_to_right
            )
            
            # We assume that the shadow produces the *strongest* correlation (=maximum of correlation).
            # To find the maximum Delta_y (see plotting in column_correlations) we compute for each row (Delta_y) the 
            # mean of intensity, and then pick the row with the highest mean intensity:
            peak_pos[delta_x-delta_x_min+1,:] = [delta_x, indmax(mean(cors,1))+delta_y_min-1]
        end
    
        #ordinary least square regression to determine the slope
        data = DataFrame(x=peak_pos[:,1],y=peak_pos[:,2])
        
        convergence = true        
        OLS = try glm(y~x, data, Normal(), IdentityLink()) 
            catch convergence
        end
        if convergence == true
            beta0[i] = coef(OLS)[2]
            beta0_err[i] = stderr(OLS)[2]
        end
        
        if plots == true
            p[i] = scatter(peak_pos[:,1],peak_pos[:,2],lab="")
            xaxis!("\\Delta x")
            yaxis!("\\Delta y")
        end
        i+=1
    end
    
    if plots == true
        beta0[1], beta0[2], beta0_err[1], beta0_err[2], p
    else
        beta0[1], beta0[2], beta0_err[1], beta0_err[2]
    end
end

"""
Input:

- imax: grid position with maximum value

- val: array of three floats, values of gridpoints one before imax, at imax, and one after imax

Output:
    
- increment ∈[-0.5,0.5] on imax to approximate the position of the peak
"""
function peak_interpolation(
    imax::Int64, #grid position with maximum value
    val::Array{Float64,1} #values of gridpoints one before imax, at imax, and one after imax
    )
    #parabolic approximation of the peak position interpolated from the three values:
    imax + 0.5*(val[1]-val[3])/(val[1]-2.*val[2]+val[3])
end

"""
Input:

   - input_filename: csv file with these columns:

      * Filename: list of file names (or paths), e.g. NA-3-10.tif

      * Type: factor of files, e.g. artery or vein

      * Series: another factor, e.g. artery3, etc.

      * Direction: d (for down = top-left to bottom-right) or u (bottom-left -> top-right)

      * LenPix: length of pixel in appropriate units

      * TimePerRow: time needed for scanning a row of pixels of the complete image (not only ROI)

   - clip: vector of four integers determining how many pixels to be clipped from top, right, bottom, left, e.g. the default is [10,10,10,10]

   - dxmin: minimum delta_x for the column correlation, default: 0

   - dxmax: maximum delta_x for the column correlation, default: 18

   - dymin: minimum delta_y for the column correlation, default: 0

   - dymax: maximum delta_y for the column correlation, default: 10

   - Plot: should violin plot be provided? default: true

Output:

   - data frame containing:

       * input data

       * estimated slope beta0_d for direction top-left to bottom-right

       * estimated slope beta0_u for direction bottom-left to top-right

       * standard error of beta0_d

       * standard error of beta0_u

       * velocity

  - if Plot==true: violin plot data structure with x = Series, y = Velocity
      
"""
function estimate_velocities(
                             input_filename::String;
                             clip::Array{Int64,1}=[0,0,0,0],
                             dxmin::Int64=0,
                             dxmax::Int64=18,
                             dymin::Int64=0,
                             dymax::Int64=10,
                             Plot::Bool=true
                             )

    files = readtable(input_filename)

    n_files = length(files[:,1])

    files = 
    hcat(files, 
         DataFrame(
                   beta0_d = zeros(n_files),
                   beta0_u = zeros(n_files),
                   beta0_d_err = zeros(n_files),
                   beta0_u_err = zeros(n_files),
                   Velocity = zeros(n_files)
                   ))

    for i in 1:n_files
        A = ROI_to_array(load(files[i,:Filename]), clip)
        files[i,:beta0_d],files[i,:beta0_u],files[i,:beta0_d_err],files[i,:beta0_u_err] = 
        slope_estimation(A, dxmin, dxmax, dymin, dymax)
        slope = files[i,:Direction]=="d" ? files[i,:beta0_d] : files[i,:beta0_u]
        files[i,:Velocity] = files[i,:LenPix]/files[i,:TimePerRow]*1./slope
    end

    if Plot==true
        p = violin(files[!isinf(files[:Velocity]),:], :Series, :Velocity, lab="")
        files, p
    else
        files
    end
end
end # module
