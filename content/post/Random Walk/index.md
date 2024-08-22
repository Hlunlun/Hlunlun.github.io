---
author: Lun
title: Random Walk Implement
date: 2024-03-27
description: paper progress
tags: [Algorithm, Probability]
categories: [Implement]
image: 
---

This is a rough draft for the first homework. I may add some more instructions later. In particular I may ask you to compared the mpirical results to theoretical expectation according to a Gaussian distribution.


## Programming
- Implement a program for simulating a particle making a random walk in D dimensions.
- Write the core part of the simulation yourself.
- In othe words **DO NOT** use a library for the core part of the simulation.


## Dimension
- The random walks are to be performed independently in each dimension.
- In 2D for example the particle will have two coordinates (x,y)
    > Let x⁽ⁱ⁾ denote the x coordinate of the particla at time step i.Then\
    > x⁽ⁱ⁾ = x⁽ⁱ⁾ +  {+1 with 50% prob, or -1 with 50% prob}\
    > y⁽ⁱ⁾ = y⁽ⁱ⁾ +  {+1 with 50% prob, or -1 with 50% prob}\
    > Note that movement in each dimension is independent of movement in the other dimensions.
- Experiments to do for dimensions D={1,2,3,4}

## Time
- At time zero, the particle always starts at the origin (e.g. the point (0,0) in 2D).
- After n steps the particle will be somewhere. For example for 2D, n=1; after 1 step the particle will be at each of the 4 coordinates (±1,±1) with probability ¼ each.

## Walks
- Perform 1000 walks of n steps, for n = {10²,10³,10⁴,10⁵,10⁶}.
- If your implementation is slow 
    - it might take a long time to do 1000 walks of length 10⁶, in which case I recommend you do {10²,10³,10⁴,10⁵} first 
    - then try to optimize your program to do the longer 10⁶ walks.
- For each walk
    - record several statistics 
    - make plots to summarize the results.
- With D dimensions, there are 2ᴰ sections (象限: half, quadrant, octant,...)
    > For example in 2D there are 4 quadrants. \
    > x,y > 0   x,y < 0   x < 0 < y   y < 0 < x
- For simplicity, we ignore time steps for which x=0 or y=0.
- For a reality check that your simulation seems okay
    - record which the number of time steps the particle is in each section 
    - confirm that average over all walks of a given length, the particle spends approximately the same time in each section (象限).
- For each walk
    - record any time steps for which the particle happens to come back to the origin.  
    - This could happen many times or not at all.
- For each combination D={1,2,3,4} × n={10²,10³,10⁴,10⁵,10⁶}, plot
    - the distribution of the L1-norm 
    - L2-norm of the final position of the particle for the origin

The horizontal axis of this plot should be distance (L1 or L2) and vertical axis should be an estimate of the probability density based on your data. One way to do this is to bin your data points and plots the data as a histogram with many bins.


## Notice !
- Again, do not just call a library function to take care of how to do the probability density estimation. Your code should do this.  
- It IS okay (of course) to use a library function to produce the image of the histogram.
- For each walk, also record the steps in which the particle is at the origin (means the particle returned to the origin).\
    This may happen many times, or may not happen at all.
- Use this data to estimate the distribution of the number of time steps required to return to the origin.  
    - This distribution depends strongly on the dimensionality, so estimate it separately for dimensionality D = {1,2,3,4}.  
    - Also estimate the expected value of the number or steps needed to return to the origin 
        - For the higher dimensionality you may find the particle often never returns to the origin during your simulation
        - in which case *use your data* to discuss a lower bound on the expected number of steps before returning to the origin
- For each dimensionality describe and vizualize these results.

- For 1D walk only.
    - Let x denote the position in the 1D walk.
        > Let n₋, n₀, n₊  denote the number of time steps for which\
        > x < 0, x = 0, x > 0 respectively

        Obviously n = n₋ + n₀ + n₊
    - Let m be defined as:  m ≝ ½n₀ + max(n₋,n₊)
    - Note that the time zero step does not count.
    - So for example with n=4, suppose your coin flips are (+1,-1,-1,-1)\
        The x values for time > 0 will be  +1,  0, -1, -2
    - Our statistics for this will be:
        > n₋ = 2, n₀ = 1, n₊ = 1, m = 2.5
        
  

For the 1000 samples from walk lengths n = {10²,10³,10⁴,10⁵,10⁶},
plot the distribution of m/n.  Discuss the results.

Note this experiment relates to the statistical notion of a martingale.
