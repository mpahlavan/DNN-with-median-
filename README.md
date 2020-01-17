# DNN-with-median-
using median function instead of summation in CNN 


PROGRAMMING ASSESSMENT
A 2D convolution operation in a CNN is illustrated in the figure below:

I*K

It involves sliding a 2D kernel of size 𝐾𝑥 x 𝐾𝑦 over the entire image where, at each step, the output is calculated by:
i) multiplying the kernel elements with the overlapping image elements
ii) summation of the resulting elements to produce a single number of the output 𝐼∗𝐾.

Your primary task is to write an alternate operation which replaces the above two steps as following:
i) takes the sinusoid of the product of kernel elements with the overlapping image elements,
ii) outputs the median of these numbers; instead of the summation.

Based on this operation, you will design a simple neural network with 1 hidden neuron that 
takes an image 𝐼 of size 5x5 as an input and applies the above mentioned modified convolution operation 
with a 3x3 kernel 𝐾 to produce an output image 𝑂 of size 3x3. Now calculate the error of the output using target 𝑇 
and perform a single back-propagation iteration with learning rate set to 1𝑒−3. You can use the following values for 𝐼, 𝐾 and 𝑇.
