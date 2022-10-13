#include <iostream>
#include <cstdlib>

// malloc() allocates a block of uninitialized memory to a pointer
// void* malloc(size_t size) with size being the number of bytes we want to allocate & *void a pointer to the memory (more specifically heap)

// an nxm matrix means: m columns, n rows
// represented in A) a one-dim array or B) two-dim array (1 n-dim array of m 1-dim arrays)

// A) (type) arr[n][m] = {a11, a12, ... anm}
// B) (type) arr[n][m] = {{a11, a12, ... a1m},..,{an1,an2, ... anm}}

int m = 2;
int n = 3;

int main()
{
    // A) save all matrix values in a contiguous chunk and use their adresses as matrix elements
    // assign memory for n*m int elements to "matrix"
    int *matrix_a;
    matrix_a = (int *)malloc(n * m * sizeof(int));

    // the matrix holds n arrays, each of which holds m int-values
    for (int i = 0; i < n * m; i++)
    {
        matrix_a[i] = i;
    }
    // deallocating the memory
    free(matrix_a);

    // B) save matrix rows as single contignous arrays and construct matrix from them
    // matrix is an array of n pointers, each of which points to m ints
    int **matrix_b;
    // matrix_b shall hold enough memory to story a pointer for every sub-array
    matrix_b = (int **)malloc(n * sizeof(int *));

    for (int i = 0; i < n; i++)
    {
        matrix_b[i] = (int *)malloc(m * sizeof(int)); // sub arrays shall be able to hold m ints
    }

    for (int i = 0; i < n; i++)
    {
        free(matrix_b[i]);
    }
    return 0;
}