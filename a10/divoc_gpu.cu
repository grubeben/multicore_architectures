#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "timer.hpp"
#include <algorithm>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// KERNEL RUN PARAMETERS
#define BLOCK_NUMBER 1
#define THREADS_PER_BLOCK 1


////////////////////////////////////////////////////////////////////////////////////////////////
// Data container //////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

// INPUT DATA
typedef struct
{
    ////////////////////
    // CPU ressources //
    ////////////////////

    size_t population_size; // Number of people to simulate
    double *rand_array;     // Random numbers

    //// Configuration
    int mask_threshold;      // Number of cases required for masks
    int lockdown_threshold;  // Number of cases required for lockdown
    int infection_delay;     // Number of days before an infected person can pass on the disease
    int infection_days;      // Number of days an infected person can pass on the disease
    int starting_infections; // Number of infected people at the start of the year
    int immunity_duration;   // Number of days a recovered person is immune

    // for each day:
    int *contacts_per_day;            // number of other persons met each day to whom the disease may be passed on
    double *transmission_probability; // how likely it is to pass on the infection to another person

    ////////////////////
    // GPU ressources //
    ////////////////////

    double *rand_array_dev; // Random numbers
    int *contacts_per_day_dev;
    double *transmission_probability_dev;
} SimInput_t;

// OUTPUT DATA
typedef struct
{
    ////////////////////
    // CPU ressources //
    ////////////////////

    // for each day:
    int *active_infections; // number of active infected on that day (including incubation period)
    int *lockdown;          // 0 if no lockdown on that day, 1 if lockdown

    // for each person:
    int *is_infected; // 0 if healthy, 1 if currently infected
    int *infected_on; // day of infection. negative if not yet infected. January 1 is Day 0.

    // BONUS
    double vaccination_rate; 

    ////////////////////
    // GPU ressources //
    ////////////////////

    // for each day:
    int *active_infections_dev;
    int *active_recovered_dev; // number of recovered people for each day
    int *lockdown_dev;

    // step 1& step 3 : for each person
    int *is_infected_dev;
    int *infected_on_dev;

    // BONUS
    double *vaccination_rate_dev;

} SimOutput_t;

// FREE DATA
// there must be a better way of doing this (as part of the struct?)
void destruction(SimInput_t input, SimOutput_t output)
{
    // input struct
    free(input.rand_array);
    free(input.contacts_per_day);
    free(input.transmission_probability);

    cudaFree(input.rand_array_dev);
    cudaFree(input.contacts_per_day_dev);
    cudaFree(input.transmission_probability_dev);

    // output struct
    free(output.active_infections);
    free(output.lockdown);
    free(output.is_infected);
    free(output.infected_on);

    cudaFree(output.active_infections_dev);
    cudaFree(output.lockdown_dev);
    cudaFree(output.is_infected_dev);
    cudaFree(output.infected_on_dev);
}

////////////////////////////////////////////////////////////////////////////////////////////////
// Data Initialization /////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

// INPUT DATA
void init_input(SimInput_t *input)
{
    ////////////////////
    // CPU ressources //
    ////////////////////

    input->population_size = 8916845; // Austria's population in 2020 according to Statistik Austria

    // RAND NUMBERS VERSION 1: generate rand number array on CPU and copy to GPU
    // FOR RAND NUMBERS VERSION 2 check kernel
    int num_rands = input->population_size * 3 * 365;
    input->rand_array = (double *)malloc(sizeof(double) * num_rands); // fill random number array
    srand(0);                                                         // initialize random seed
    for (int i = 0; i < num_rands; i++)
    {
        input->rand_array[i] = ((double)rand()) / (double)RAND_MAX; // random number between 0 and 1
    }

    input->mask_threshold = 5000;
    input->lockdown_threshold = 50000;
    input->infection_delay = 5; // 5 to 6 days incubation period (average) according to WHO
    input->infection_days = 3;  // assume three days of passing on the disease
    input->starting_infections = 10;
    input->immunity_duration = 180; // half a year of immunity

    input->contacts_per_day = (int *)malloc(sizeof(int) * 365);
    input->transmission_probability = (double *)malloc(sizeof(double) * 365);
    for (int day = 0; day < 365; ++day)
    {
        input->contacts_per_day[day] = 6;                                                 // arbitrary assumption of six possible transmission contacts per person per day, all year
        input->transmission_probability[day] = 0.2 + 0.1 * cos((day / 365.0) * 2 * M_PI); // higher transmission in winter, lower transmission during summer
    }

    ////////////////////
    // GPU ressources //
    ////////////////////

    // step2
    cudaMalloc(&input->contacts_per_day_dev, sizeof(int) * 365);
    cudaMemcpy(input->contacts_per_day_dev, input->contacts_per_day, sizeof(int) * 365, cudaMemcpyHostToDevice);

    cudaMalloc(&input->transmission_probability_dev, sizeof(double) * 365);
    cudaMemcpy(input->transmission_probability_dev, input->transmission_probability, sizeof(double) * 365, cudaMemcpyHostToDevice);

    // RAND NUMBERS VERSION 1
    cudaMalloc(&input->rand_array_dev, sizeof(double) * (num_rands));
    cudaMemcpy(input->rand_array_dev, input->rand_array, sizeof(double) * (num_rands), cudaMemcpyHostToDevice);

}

// OUTPUT DATA
void init_output(SimOutput_t *output, int population_size)
{
    ////////////////////
    // CPU ressources //
    ////////////////////

    output->active_infections = (int *)malloc(sizeof(int) * 365);
    output->lockdown = (int *)malloc(sizeof(int) * 365);
    for (int day = 0; day < 365; ++day)
    {
        output->active_infections[day] = 0;
        output->lockdown[day] = 0;
    }

    output->is_infected = (int *)malloc(sizeof(int) * population_size);
    output->infected_on = (int *)malloc(sizeof(int) * population_size);

    for (int i = 0; i < population_size; ++i)
    {
        output->is_infected[i] = 0;
        output->infected_on[i] = 0;
    }

    // BONUS
    output->vaccination_rate = log10(1); // vaccination rate at start of the year = 0

    ////////////////////
    // GPU ressources //
    ////////////////////

    // step2
    cudaMalloc(&output->active_infections_dev, sizeof(int) * 365);
    cudaMalloc(&output->active_recovered_dev, sizeof(int) * 365);
    cudaMalloc(&output->lockdown_dev, sizeof(int) * 365);
    cudaMemcpy(output->active_infections_dev, output->active_infections, sizeof(int) * 365, cudaMemcpyHostToDevice);
    cudaMemcpy(output->active_recovered_dev, output->is_infected, sizeof(int) * 365, cudaMemcpyHostToDevice); // use is_infected to initialize with zeros
    cudaMemcpy(output->lockdown_dev, output->lockdown, sizeof(int) * 365, cudaMemcpyHostToDevice);

    // step1 & step3
    cudaMalloc(&output->is_infected_dev, sizeof(int) * population_size);
    cudaMalloc(&output->infected_on_dev, sizeof(int) * population_size);
    cudaMemcpy(output->is_infected_dev, output->is_infected, sizeof(int) * population_size, cudaMemcpyHostToDevice);
    cudaMemcpy(output->infected_on_dev, output->infected_on, sizeof(int) * population_size, cudaMemcpyHostToDevice);

    // BONUS
    output->vaccination_rate = 0.74; // Gesundheitsministerium
    cudaMalloc(&output->vaccination_rate_dev, sizeof(double));
    cudaMemcpy(output->vaccination_rate_dev, &output->vaccination_rate, sizeof(double), cudaMemcpyHostToDevice);
}

////////////////////////////////////////////////////////////////////////////////////////////////
// CUDA KERNELS ////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

// RANDOM NUMBER GEN VERSION 2
// use rank as seed0, PNRG output0 as see1 and so on
__device__ double PNRG(int seed) //__device__ funtions can be called from kernels
{
    // inspo: https://en.wikipedia.org/wiki/Linear_congruential_generator
    //        https://stackoverflow.com/questions/19140589/linear-congruential-generator-in-python

    // generator input
    int m = pow(2, 32);
    int a = 1103515245;
    int c = 12345;

    // generating PRN
    int r = (a * seed + c) % m;

    // scale to [0,1]
    double r_scale = r / (pow(2,31) - 1);
    return r_scale;
}

// STEP 1 2 3 IN ONE KERNEL:
// 1: determine number of infections and recoveries;
// 2: determine today's transmission probability and contacts based on pandemic situation;
// 3: pass on infections within population;
__global__ void cuda_step123(int day, SimInput_t *input, SimOutput_t *output)
{
    if (blockIdx.x * blockDim.x + threadIdx.x == 0)
        printf("RANK %d starting kernel \n", threadIdx.x);
        printf("RANK %d: test 1 whether passing struct works: For scalar int located on CPU: Population size %d \n", threadIdx.x, &input->population_size);
        printf("RANK %d: test 2 whether passing struct works: For scalar double located on GPU: Vaccination rate %f \n", threadIdx.x, &output->vaccination_rate);
        printf("RANK %d: test 3 whether passing struct works: For array located on GPU: random number entry %f \n", threadIdx.x, &input->rand_array_dev);
        
    // STEP1
    // every thread counts the infected/recovered it handles (this is inspired by the dot product)
    __shared__ int num_infected_current_shared[256];
    __shared__ int num_recovered_current_shared[256];

    // let every thread deal with one person at a time
    int num_infected_current_local = 0;
    int num_recovered_current_local = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input->population_size; i += blockDim.x * gridDim.x)
    {
        if (blockIdx.x * blockDim.x + threadIdx.x == 0)
            printf("RANK %d entered infection counting loop \n", threadIdx.x);
            
        if (output->is_infected_dev[i] > 0) // if person i is infected
        {
            if (output->infected_on_dev[i] > day - input->infection_delay - input->infection_days && output->infected_on_dev[i] <= day - input->infection_delay) // currently infected and incubation period over
                num_infected_current_local += 1;
            else if (output->infected_on_dev[i] < day - input->infection_delay - input->infection_days) // both incubation and infectionous time are over
                num_recovered_current_local += 1;
        }
    }

    if (blockIdx.x * blockDim.x + threadIdx.x == 0)
        printf("RANK %d counted local number of infections: %d \n", threadIdx.x, num_infected_current_local);

    // we need to sync threads here and reduce
    num_infected_current_shared[threadIdx.x] = num_infected_current_local;
    num_recovered_current_shared[threadIdx.x] = num_recovered_current_local;
    for (int k = blockDim.x / 2; k > 0; k /= 2)
    {
        __syncthreads();
        if (threadIdx.x < k)
        {
            num_infected_current_shared[threadIdx.x] += num_infected_current_shared[threadIdx.x + k];
            num_recovered_current_shared[threadIdx.x] += num_recovered_current_shared[threadIdx.x + k];
        }
    }

    if (blockIdx.x * blockDim.x + threadIdx.x == 0)
        printf("RANK %d done with strides; block %d counted %d infections \n", threadIdx.x, blockIdx.x, num_infected_current_shared[threadIdx.x]);

    // after stride thread 0 holds block_sums, it will now AtomicAdd them to the ouput-> GPU arrays
    if (threadIdx.x == 0)
    {
        atomicAdd(&output->active_infections_dev[day], num_infected_current_shared[0]);
        printf("RANK %d perfomrmed AtomicAdd; all blocks counted %d  \n", threadIdx.x, output->active_infections_dev[day]);

    }
    // care for non-parallelizable stuff with only one thread
    if (blockIdx.x * blockDim.x + threadIdx.x == 0)
    {
        printf("RANK %d of block %d started serial, one-thread section \n", threadIdx.x, blockIdx.x);

        if (day > 0 && output->lockdown_dev[day - 1] == 1)
        { // end lockdown if number of infections has reduced significantly
            output->lockdown_dev[day] = (output->active_infections_dev[day] < input->lockdown_threshold / 3) ? 0 : 1;
        }
        // daily announcement
        char lockdown[] = " [LOCKDOWN]";
        char normal[] = "";
        printf("Day %d%s: %d active, %d recovered\n", day, output->lockdown_dev[day] ? lockdown : normal, output->active_infections_dev[day], num_recovered_current_shared[0]);

        // STEP2
        if (output->active_infections_dev[day] > input->mask_threshold)
        { // transmission is reduced with masks. Arbitrary factor: 2
            input->transmission_probability_dev[day] /= 2.0;
        }
        if (output->lockdown_dev[day])
        { // contacts are significantly reduced in lockdown. Arbitrary factor: 4
            input->contacts_per_day_dev[day] /= 4;
        }
    }

    // not sure if necessary, but I want to ensure every thread grabs the manipulated values (STEP2)
    __syncthreads();

    double contacts_today = input->contacts_per_day_dev[day];
    double transmission_probability_today = input->transmission_probability_dev[day];

    // STEP 3 - we back in parallel mode
    if (blockIdx.x * blockDim.x + threadIdx.x == 0)
        printf("RANK %d starts into STEP \n", threadIdx.x);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input->population_size; i += blockDim.x * gridDim.x)
    {
        if (output->is_infected_dev[i] > 0 && output->infected_on_dev[i] > day - input->infection_delay - input->infection_days // currently infected
            && output->infected_on_dev[i] <= day - input->infection_delay)                                                      // already infectious
        {
            // pass on infection to other persons with transmission probability
            for (int j = 0; j < contacts_today; ++j)
            {
                // RANDOM NUMBER GEN PART //////////////////////////////////////////////////
                // VERSION 1 - VIA CPU
                double r1 = input->rand_array_dev[i];     // random number between 0 and 1
                double r2 = input->rand_array_dev[2 * i]; // new random number to determine a random other person to transmit the virus to
                double r3 = input->rand_array_dev[3 * i]; // new random number to determine a random other person to transmit the virus to
                
                // VERSION 2 - ON GPU
                // double r1 = PNRG(i);
                // double r2 = PNRG(int (10*r1));
                // double r3 = PNRG(int(10*r2));
                ////////////////////////////////////////////////////////////////////////////

                 if (blockIdx.x * blockDim.x + threadIdx.x == 0)
                    printf("RANK %d received these 3 random numbers: %f %f %f \n", r1,r2,r3);

                // BONUS: IS OTHER_PERSON VACCINATED? ////////////////////////////////////////////
                if (r3 < output->vaccination_rate)
                {
                    transmission_probability_today /= 2; // Likeliness of sickness outbreak is halfed im person is vaccinated
                }
                ////////////////////////////////////////////////////////////////////////////

                if (r1 < transmission_probability_today)
                {
                    int other_person = r2 * input->population_size;

                    // SHOULD THIS BE A SEQUENTIAL SECTION?
                    ////////////////////////////////////////////////////////////////////////
                    if (output->is_infected_dev[other_person] == 0                                 // other person is not infected
                        || output->infected_on_dev[other_person] < day - input->immunity_duration) // other person has no more immunity
                    {
                        output->is_infected_dev[other_person] = 1;
                        output->infected_on_dev[other_person] = day;
                    }
                    ///////////////////////////////////////////////////////////////////////
                }
                if (blockIdx.x * blockDim.x + threadIdx.x == 0)
                    printf("RANK %d is done with kernel \n");

            }
        }
    }
}

// KERNEL WRAP; init_input and init_output must be called prior because they fill dev arrays
void run_simulation_gpu(SimInput_t *input, SimOutput_t *output)
{
    int subject_of_log=1;
    for (int day = 0; day < 365; ++day) // loop over all days of the year
    {
        cuda_step123<<<BLOCK_NUMBER, THREADS_PER_BLOCK>>>(day, input, output);
        cudaDeviceSynchronize();

        // hand back current_infections_dev back to CPU as task demands?
        // but we can simply print within kernel, why create overhead?

        // increase vaccination rate
        subject_of_log+=0.0137;
        output->vaccination_rate+= log10(subject_of_log);
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////
// MAIN ////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{

    SimInput_t input;
    SimOutput_t output;
    // SimInput_t* input_gpu;
    // SimOutput_t* output_gpu;

    // initiliaze on host (and vopy to device)
    init_input(&input);
    init_output(&output, input.population_size);

    // // allocate memory
    // int inputSize    = sizeof(SimInput_t);
    // int outputSize    = sizeof(SimOutput_t);

    // cudaMalloc((SimInput_t**)&input_gpu, inputSize); //stack uses void**
    // cudaMalloc((SimOutput_t**)&output_gpu, outputSize);

    // // bring over CPU data
    // cudaMemcpy(input_gpu, &input, inputSize, cudaMemcpyHostToDevice);
    // cudaMemcpy(output_gpu, &output, outputSize, cudaMemcpyHostToDevice);
    
    // Go though simulation
    Timer timer;

    timer.reset();
    run_simulation_gpu(&input, &output);
    printf("Simulation time: %g\n", timer.get());

    destruction(input, output);

    return EXIT_SUCCESS;
}