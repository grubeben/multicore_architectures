// The following three defines are necessary to pick the correct OpenCL version on the machine:
#define VEXCL_HAVE_OPENCL_HPP
#define CL_HPP_TARGET_OPENCL_VERSION  120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
 
#include <iostream>
#include <stdexcept>
#include <vexcl/vexcl.hpp>

#include <numeric>
#include <vector>
#include <cstdlib>
#include <algorithm>

#include "timer.hpp"
 
int main() 
{
    vex::Context ctx(vex::Filter::GPU && vex::Filter::DoublePrecision);

    std::cout << ctx << std::endl; // print list of selected devices

    for (int N=16; N<1e8;  N*=4)
    {
        //host data
        std::vector<double> x_host(N, 1.0), y_host(N, 2.0);

        //gpu data
        vex::vector<double> x_device(ctx, x_host);
        vex::vector<double> y_device(ctx, y_host);

        Timer timer;
        float  time=0;

        for (int j = 0; j < 5; j++) {

            timer.reset();

            x_device = x_device + y_device;
            y_device = x_device - 2* y_device;
            x_device = x_device * y_device;

            // bring vector back for accumulation
            vex::copy(x_device, x_host);
            float result = std::accumulate(x_host.begin(), x_host.end(), 0);
            time+=timer.get();
        }
        std::cout << N<< " " << time/5 << std::endl;
    }
    return 0; 
}