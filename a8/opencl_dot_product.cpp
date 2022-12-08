typedef double       ScalarType;


#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <stdexcept>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

// Helper include file for error checking
#include "ocl-error.hpp"
#include "timer.hpp"


const char *my_opencl_program = ""
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"    // required to enable 'double' inside OpenCL programs
""
"__kernel void dot_prod(__global double *x,\n"
"                      __global double *y,\n"
"                      __global double *partial_result,\n"
"                      unsigned int N\n)"
"{\n"
"   __local double shared_mem[512];"
"   double thread_sum=0;"
"   for (unsigned int i  = get_global_id(0);\n"
"                    i  < N;\n"
"                    i += get_global_size(0))\n"
"   {\n"
"       thread_sum+= x[i]* y[i];\n"
"   }\n"
""
"   shared_mem[get_local_id(0)]=thread_sum;\n"
"   for (unsigned int stride  = get_local_size(0)/2;\n"
"                    stride  > 0;\n"
"                    stride /= 2)\n"
"   {\n"
"       barrier(CLK_GLOBAL_MEM_FENCE);\n"
"       if (get_local_id(0)<stride) shared_mem[get_local_id(0)]+=shared_mem[get_local_id(0)+stride];\n"
"   }\n"

"   barrier(CLK_GLOBAL_MEM_FENCE);\n"
"   if (get_local_id(0)==0) partial_result[get_group_id(0)]=shared_mem[0];"
""
"}";  // you can have multiple kernels within a single OpenCL program. For simplicity, this OpenCL program contains only a single kernel.


int main()
{
  cl_int err;

  //
  /////////////////////////// Part 1: Set up an OpenCL context with one device ///////////////////////////////////
  //

  //
  // Query platform:
  //
  char platform_name[64];
  size_t platform_name_len = 0;
  cl_uint num_platforms;
  cl_platform_id platform_ids[42];   //no more than 42 platforms supported...
  err = clGetPlatformIDs(42, platform_ids, &num_platforms); OPENCL_ERR_CHECK(err);
  cl_platform_id my_platform = platform_ids[1];
  err= clGetPlatformInfo(my_platform, CL_PLATFORM_VENDOR, sizeof(char)*63, platform_name, &platform_name_len);



  //
  // Query devices:
  //
  cl_device_id device_ids[42];
  cl_uint num_devices;
  err = clGetDeviceIDs(my_platform, CL_DEVICE_TYPE_ALL, 42, device_ids, &num_devices); OPENCL_ERR_CHECK(err);
  cl_device_id my_device_id = device_ids[0]; //this is only applicable to a "one device" platfrom right?
                                            // in a different contxt I woudlnt know yet which device is the one I want

  char device_name[64];
  size_t device_name_len = 0;
  err = clGetDeviceInfo(my_device_id, CL_DEVICE_NAME, sizeof(char)*63, device_name, &device_name_len); OPENCL_ERR_CHECK(err);

  // platform and device info
  
  std::cout << "# Platforms found: " << num_platforms << std::endl;
  std::cout << "Using the following platform: " << platform_name << std::endl;
  std::cout << "# Devices found: " << num_devices << std::endl;
  std::cout << "Using the following device: \n\n" << device_name << std::endl;
  

  //
  // Create context: WHERE IS CONTEXT LINKED TO PLATFORM?
  //
  cl_context my_context = clCreateContext(0, 1, &my_device_id, NULL, NULL, &err); OPENCL_ERR_CHECK(err);


  //
  // create a command queue for the device:
  //
  cl_command_queue my_queue = clCreateCommandQueueWithProperties(my_context, my_device_id, 0, &err); OPENCL_ERR_CHECK(err);



  //
  /////////////////////////// Part 2: Create a program and extract kernels ///////////////////////////////////
  //

  Timer timer;
  timer.reset();

  //
  // Build the program:
  //
  size_t source_len = std::string(my_opencl_program).length();
  cl_program prog = clCreateProgramWithSource(my_context, 1, &my_opencl_program, &source_len, &err);OPENCL_ERR_CHECK(err);
  err = clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);

  //
  // Print compiler errors if there was a problem:
  //
  if (err != CL_SUCCESS) {

    char *build_log;
    size_t ret_val_size;
    err = clGetProgramBuildInfo(prog, my_device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
    build_log = (char *)malloc(sizeof(char) * (ret_val_size+1));
    err = clGetProgramBuildInfo(prog, my_device_id, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
    build_log[ret_val_size] = '\0'; // terminate string
    std::cout << "Log: " << build_log << std::endl;
    free(build_log);
    std::cout << "OpenCL program sources: " << std::endl << my_opencl_program << std::endl;
    return EXIT_FAILURE;
  }

  //
  // Extract the only kernel in the program:
  //
  cl_kernel my_kernel = clCreateKernel(prog, "dot_prod", &err); OPENCL_ERR_CHECK(err);

  //std::cout << "Time to compile and create kernel: " << timer.get() << std::endl;


  //
  /////////////////////////// Part 3: Create memory buffers ///////////////////////////////////
  //

  for (int N = 32; N <1e8; N*=2)
  {

        //
        // Set up buffers on host:
        //
        size_t local_size = 256;     //n_blocks
        size_t thread_size = 256;    //n_threads
        size_t global_size = local_size*thread_size; //n_blocks*n_threads

        cl_uint vector_size = N; //128*1024;
        cl_uint group_size= local_size;

        std::vector<ScalarType> x(vector_size, 1.0); 
        std::vector<ScalarType> y(vector_size, 2.0);
        std::vector<ScalarType> z(group_size, 0.0); //partial sum vector

        /*
        std::cout << std::endl;
        std::cout << "Vectors before kernel launch:" << std::endl;
        std::cout << "x: " << x[0] << " " << x[1] << " " << x[2] << " ..." << std::endl;
        std::cout << "y: " << y[0] << " " << y[1] << " " << y[2] << " ..." << std::endl;
        std::cout << "z: " << z[0] << " " << z[1] << " " << z[2] << " ..." << std::endl;
        */

        //
        // Now set up OpenCL buffers:
        //
        cl_mem ocl_x = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, vector_size * sizeof(ScalarType), &(x[0]), &err); OPENCL_ERR_CHECK(err);
        cl_mem ocl_y = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, vector_size * sizeof(ScalarType), &(y[0]), &err); OPENCL_ERR_CHECK(err);
        cl_mem ocl_z = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, group_size * sizeof(ScalarType), &(z[0]), &err); OPENCL_ERR_CHECK(err);


        //
        /////////////////////////// Part 4: Run kernel ///////////////////////////////////
        //

        //
        // Set kernel arguments:
        //
        err = clSetKernelArg(my_kernel, 0, sizeof(cl_mem),  (void*)&ocl_x); OPENCL_ERR_CHECK(err);
        err = clSetKernelArg(my_kernel, 1, sizeof(cl_mem),  (void*)&ocl_y); OPENCL_ERR_CHECK(err);
        err = clSetKernelArg(my_kernel, 2, sizeof(cl_mem), (void*)&ocl_z); OPENCL_ERR_CHECK(err);
        err = clSetKernelArg(my_kernel, 3, sizeof(cl_uint), (void*)&vector_size); OPENCL_ERR_CHECK(err);

        //
        // Enqueue kernel in command queue:
        //

        timer.reset();
        for (int i = 0; i < 3;i++) 
        {
            err = clEnqueueNDRangeKernel(my_queue, my_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL); OPENCL_ERR_CHECK(err);
        

            
            // wait for all operations in queue to finish:
            err = clFinish(my_queue); OPENCL_ERR_CHECK(err);


            //
            /////////////////////////// Part 5: Get data from OpenCL buffer ///////////////////////////////////
            //

            err = clEnqueueReadBuffer(my_queue, ocl_z, CL_TRUE, 0, sizeof(ScalarType) * z.size(), &(z[0]), 0, NULL, NULL); OPENCL_ERR_CHECK(err);

            // sum up partial sums on CPU
            double dot_product=0;
            for (int i = 0; i <z.size(); i++)
            {
                dot_product+=z[i];
            }
        }
        std::cout << N << " " << timer.get()/3 << std::endl;

        /*
        std::cout << std::endl;
        std::cout << "Vectors after kernel execution:" << std::endl;
        std::cout << "x: " << x[0] << " " << x[1] << " " << x[2] << " ..." << std::endl;
        std::cout << "y: " << y[0] << " " << y[1] << " " << y[2] << " ..." << std::endl;
        std::cout << "z: " << z[0] << " " << z[1] << " " << z[2] << " ..." << std::endl;

        std::cout <<"result: " << dot_product << std::endl;
        */

        //
        // cleanup
        //
        clReleaseMemObject(ocl_x);
        clReleaseMemObject(ocl_y);
        clReleaseMemObject(ocl_z);
    }

    clReleaseProgram(prog);
    clReleaseCommandQueue(my_queue);
    clReleaseContext(my_context);

    std::cout << std::endl;
    std::cout << "#" << std::endl;
    std::cout << "# My first OpenCL application finished successfully!" << std::endl;
    std::cout << "#" << std::endl;
    return EXIT_SUCCESS;
}

