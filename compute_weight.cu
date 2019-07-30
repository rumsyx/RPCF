__global__ void compute_weight( double * pi, double* b, double c ) 
{
    *pi += c;
}