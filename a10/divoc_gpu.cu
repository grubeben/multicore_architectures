#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "timer.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//
// CUDA KERNELS
//
__global__ void cuda_step1()
{

}

__global__ void cuda_step2()
{
    
}

__global__ void cuda_step3()
{
    
}

void run_simulation_gpu(const SimInput_t *input, SimOutput_t *output)
{
    cuda_step1<<<,>>>();
    //memory management
    cuda_step2<<<,>>>();
    //memory management
    cuda_step3<<<,>>>();
    //memory management
}

//
// Data container for simulation input
//
typedef struct
{
    ////////////////////
    // CPU ressources //
    ////////////////////

    size_t population_size; // Number of people to simulate
    double *rand_array; // Random numbers

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

void init_input(SimInput_t *input)
{
    ////////////////////
    // CPU ressources //
    ////////////////////

    input->population_size = 8916845;               // Austria's population in 2020 according to Statistik Austria

    num_rands = input->population_size * 2 * 365;
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
    
    //step1

    //step2
    cudaMalloc(&input->contacts_per_day_dev, sizeof(int) * 365);
    cudaMemcpy(input->contacts_per_day_dev, input->contacts_per_day, sizeof(int) * 365, cudaMemcpyHostToDevice);

    cudaMalloc(&input->transmission_probability_dev, sizeof(double) * 365);
    cudaMemcpy(input->transmission_probability_dev, input->transmission_probability, sizeof(double) * 365, cudaMemcpyHostToDevice);

    //step3
    cudaMalloc(&input->rand_array_dev, sizeof(double) * (num_rands));
    cudaMemcpy(input->rand_array_dev, input->rand_array, sizeof(double) * (num_rands), cudaMemcpyHostToDevice);
}

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

    ////////////////////
    // GPU ressources //
    ////////////////////

    // for each day:
    int *active_infections_dev; 
    int *lockdown_dev;          

    // step 1& step 3 : for each person 
    int *is_infected_dev; 
    int *infected_on_dev;

} SimOutput_t;

//
// Initializes the output data structure (values to zero, allocate arrays)
//
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
    
    ////////////////////
    // GPU ressources //
    ////////////////////
    
    //step2
    cudaMalloc(&output->active_infections_dev, sizeof(int) * 365);
    cudaMalloc(&output->lockdown_dev, sizeof(int) * 365);
    cudaMemcpy(output->active_infections_dev, output->active_infections, sizeof(int) * 365, cudaMemcpyHostToDevice);
    cudaMemcpy(output->lockdown_dev, output->lockdown, sizeof(int) * 365, cudaMemcpyHostToDevice);

    //step1 & step3
    cudaMalloc(&output->is_infected_dev, sizeof(int) * population_size);
    cudaMalloc(&output->infected_on_dev, sizeof(int) * population_size);
    cudaMemcpy(output->is_infected_dev, output->is_infected, sizeof(int) * population_size, cudaMemcpyHostToDevice);
    cudaMemcpy(output->infected_on_dev, output->infected_on, sizeof(int) * population_size, cudaMemcpyHostToDevice);
}

void destruction(SimInput_t input, SimOutput_t output)
{
    // input stuff
    free(input->rand_array);
    free(input->contacts_per_day);
    free(input->transmission_probability);

    cudaFree(input->rand_array_dev);
    cudaFree(input->contacts_per_day_dev);
    cudaFree(input->transmission_probability_dev);

    //output stuff
    free(output->active_infections);
    free(output->lockdown);
    free(output->is_infected);
    free(output->infected_on);

    cudaFree(output->active_infections_dev);
    cudaFree(output->lockdown_dev);
    cudaFree(output->is_infected_dev);
    cudaFree(output->infected_on_dev);
}
void run_simulation(const SimInput_t *input, SimOutput_t *output)
{
    //
    // Init data. For simplicity we set the first few people to 'infected'
    //
    for (int i = 0; i < input->population_size; ++i)
    {
        output->is_infected[i] = (i < input->starting_infections) ? 1 : 0;
        output->infected_on[i] = (i < input->starting_infections) ? 0 : -1; // infected on January 1
    }

    //
    // Run simulation
    //
    for (int day = 0; day < 365; ++day) // loop over all days of the year
    {
        //
        // Step 1: determine number of infections and recoveries
        //
        int num_infected_current = 0;
        int num_recovered_current = 0;
        for (int i = 0; i < input->population_size; ++i)
        {

            if (output->is_infected[i] > 0) // if person i is infected
            {
                if (output->infected_on[i] > day - input->infection_delay - input->infection_days && output->infected_on[i] <= day - input->infection_delay) // currently infected and incubation period over
                    num_infected_current += 1;
                else if (output->infected_on[i] < day - input->infection_delay - input->infection_days) // both incubation and infectionous time are over
                    num_recovered_current += 1;
            }
        }

        output->active_infections[day] = num_infected_current;
        if (num_infected_current > input->lockdown_threshold)
        {
            output->lockdown[day] = 1;
        }
        if (day > 0 && output->lockdown[day - 1] == 1)
        { // end lockdown if number of infections has reduced significantly
            output->lockdown[day] = (num_infected_current < input->lockdown_threshold / 3) ? 0 : 1;
        }
        char lockdown[] = " [LOCKDOWN]";
        char normal[] = "";
        printf("Day %d%s: %d active, %d recovered\n", day, output->lockdown[day] ? lockdown : normal, num_infected_current, num_recovered_current);

        //
        // Step 2: determine today's transmission probability and contacts based on pandemic situation
        //
        double contacts_today = input->contacts_per_day[day];
        double transmission_probability_today = input->transmission_probability[day];
        if (num_infected_current > input->mask_threshold)
        { // transmission is reduced with masks. Arbitrary factor: 2
            transmission_probability_today /= 2.0;
        }
        if (output->lockdown[day])
        { // contacts are significantly reduced in lockdown. Arbitrary factor: 4
            contacts_today /= 4;
        }

        //
        // Step 3: pass on infections within population
        //
        for (int i = 0; i < input->population_size; ++i) // loop over population
        {
            if (output->is_infected[i] > 0 && output->infected_on[i] > day - input->infection_delay - input->infection_days // currently infected
                && output->infected_on[i] <= day - input->infection_delay)                                                  // already infectious
            {
                // pass on infection to other persons with transmission probability
                for (int j = 0; j < contacts_today; ++j)
                {
                    double r = ((double)rand()) / (double)RAND_MAX; // random number between 0 and 1
                    if (r < transmission_probability_today)
                    {
                        r = ((double)rand()) / (double)RAND_MAX; // new random number to determine a random other person to transmit the virus to
                        int other_person = r * input->population_size;
                        if (output->is_infected[other_person] == 0                                 // other person is not infected
                            || output->infected_on[other_person] < day - input->immunity_duration) // other person has no more immunity
                        {
                            output->is_infected[other_person] = 1;
                            output->infected_on[other_person] = day;
                        }
                    }

                } // for contacts_per_day
            }     // if currently infected
        }         // for i

    } // for day
}

int main(int argc, char **argv)
{

    SimInput_t input;
    SimOutput_t output;

    init_input(&input);
    init_output(&output, input.population_size);

    Timer timer;

    srand(0); // initialize random seed for deterministic output
    timer.reset();
    run_simulation(&input, &output);
    printf("Simulation time: %g\n", timer.get());

    destruction(&input, &output);

    return EXIT_SUCCESS;
}