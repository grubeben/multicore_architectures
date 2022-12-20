
// specify use of OpenCL 1.2:
#define CL_TARGET_OPENCL_VERSION 120
#define CL_MINIMUM_OPENCL_VERSION 120

#include <vector>
#include <algorithm>
#include <iostream>

#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/functional/math.hpp>

#include "timer.hpp"

namespace compute = boost::compute;

// define boost multiply operation
BOOST_COMPUTE_FUNCTION(double, vec_add, (double x, double y),
                       {
                           return x + y;
                       });
BOOST_COMPUTE_FUNCTION(double, vec_sub, (double x, double y),
                       {
                           return x - y;
                       });

BOOST_COMPUTE_FUNCTION(double, scalarproduct, (double x, double y),
                       {
                           return x * y;
                       });

int main()
{
    // get default device and setup context
    compute::device device = compute::system::default_device();
    compute::context context(device);
    compute::command_queue queue(context, device);

    for (int N = 4; N < 1e8; N *= 4)
    {
        // generate and fill host vectors
        std::vector<double> host_x(N);
        std::vector<double> host_y(N);
        std::vector<double> host_result(N);
        std::fill(host_x.begin(), host_x.end(), 1.0);
        std::fill(host_y.begin(), host_y.end(), 2.0);
        std::fill(host_result.begin(), host_result.begin() + N, 0);

        // create vectors on the device
        compute::vector<double> x_device(host_x.size(), context);
        compute::vector<double> y_device(host_y.size(), context);
        compute::vector<double> result_device(host_result.size(), context);

        // transfer data from the host to the device
        compute::copy(host_x.begin(), host_x.end(), x_device.begin(), queue);
        compute::copy(host_y.begin(), host_y.end(), y_device.begin(), queue);
        compute::copy(host_result.begin(), host_result.end(), y_device.begin(), queue);

        Timer timer;
        float time = 0;

        for (int j = 0; j < 10; j++)
        {
            timer.reset();

            // generate scalar product (x-y,f+y)
            compute::transform(x_device.begin(), x_device.end(), y_device.begin(),
                               result_device.begin(), vec_add, queue); // use result as temporary buffer
            compute::transform(x_device.begin(), x_device.end(), y_device.begin(),
                               y_device.begin(), vec_sub, queue);
            compute::transform(result_device.begin(), result_device.end(), y_device.begin(),
                               result_device.begin(), scalarproduct, queue);

            // copy values back to the host
            compute::copy(result_device.begin(), result_device.end(), host_result.begin(), queue);

            // accumulate without stride
            float solution = std::accumulate(host_result.begin(),host_result.end(),0);

            time += timer.get();

            //std::cout << solution << std::endl;

        }
        std::cout << N << " " << time / 10 << std::endl;
    }
    return 0;
}