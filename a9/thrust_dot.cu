#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <cstdlib>

#include <algorithm>
#include <iostream>
#include <vector>
#include <numeric>
#include "timer.hpp"


int main(void)
{
    Timer timer;

    for (int N=16; N<1e8;  N*=4)
    {
        // create host vectors
        thrust::host_vector<double> x_host(N);
        thrust::host_vector<double> y_host(N);
        thrust::host_vector<double> result_buffer_host(N);

        // fill them
        std::fill(x_host.begin(), x_host.end(), 1.0);
        std::fill(y_host.begin(), y_host.end(), 2.0);
        std::fill(result_buffer_host.begin(), result_buffer_host.end(), 0.0);


        // transfer data to the device
        thrust::device_vector<double> x_device = x_host;
        thrust::device_vector<double> y_device = y_host;
        thrust::device_vector<double> result_buffer_device = result_buffer_host;

        float  time=0;

        for (int j = 0; j < 5; j++) {

            timer.reset();
            thrust::plus<double> vec_add;
            thrust::transform(thrust::device, x_device.begin(), x_device.end(), 
                            y_device.begin(), x_device.begin(),vec_add);

            //subtract y twice
            thrust::minus<double> vec_sub1;
            thrust::transform(thrust::device, x_device.begin(), x_device.end(), 
                            y_device.begin(), y_device.begin(),vec_sub1);
            thrust::minus<double> vec_sub2;
            thrust::transform(thrust::device, x_device.begin(), x_device.end(), 
                            y_device.begin(), y_device.begin(),vec_sub2);
            
            // scalar product
            thrust::multiplies<double> scalar_product;
            thrust::transform(thrust::device, x_device.begin(), x_device.end(),
                            y_device.begin(),y_device.begin(),scalar_product);

            // transfer data back to host
            thrust::copy(y_device.begin(), y_device.end(), y_host.begin());

            float solution = std::accumulate(y_host.begin(),y_host.end(),0);
            time+= timer.get();

        }
        // transfer data back to host
        //thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

        std::cout << N << " " << time/5<< std::endl;

    }

    return 0;
}
