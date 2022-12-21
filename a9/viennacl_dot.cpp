#include <iostream>
#include <numeric>
#include <vector>
#include <cstdlib>
#include <algorithm>
 
#define VIENNACL_WITH_CUDA
 
#include "viennacl/vector.hpp"
#include "viennacl/linalg/inner_prod.hpp"


#include "timer.hpp"
 
 
int main() {
 
    Timer timer;
    for(int N = 16; N < 1e8; N *= 4) 
    { 
        viennacl::vector<double> x = viennacl::scalar_vector<double>(N, 1.0);
        viennacl::vector<double> y = viennacl::scalar_vector<double>(N, 2.0);

        float time=0;
        double result;
        
        for(int j=0;j<5;j++) 
        {
            timer.reset();
            result = viennacl::linalg::inner_prod(x+y,x-y);
            time+=timer.get();            
        }
        std::cout << N << " " << time/5<< std::endl;
    }
    return EXIT_SUCCESS;
}