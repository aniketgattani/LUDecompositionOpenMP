#include <iostream>
#include <iomanip>
#include <cmath>
#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>  
#include <math.h>

using namespace std;

void print_matrix(float**, int);

//print the matrix out
void print_matrix(float** matrix, int size)
{
    //for each row...
    for (int i = 0; i < size; i++)
    {
        //for each column
        for (int j = 0; j < size; j++)
        {
            //print out the cell
            cout << left << setw(9) << setprecision(3) << matrix[i][j] << left <<  setw(9);
        }
        //new line when ever row is done
        cout << endl;
    }
}

//do LU decomposition
//a is the matrix that will be split up into l and u
//array size for all is size x size
void l_u_d(float** a, float** l, float** u, int size)
{
    //initialize a simple lock for parallel region
    omp_lock_t lock;

    omp_init_lock(&lock);
    //for each column...
    //make the for loops of lu decomposition parallel. Parallel region
    #pragma omp parallel shared(a,l,u)
    {
    	#pragma omp single
	{
	for (int i = 0; i < size; i++)
        {
            l[i][i]=1; 
	   //for each row....
            //rows are split into seperate threads for processing
            
	    #pragma omp taskloop
            for (int j = i; j < size; j++)
            {
                //if j is smaller than i, set l[j][i] to
                //otherwise, do some math to get the right value
                u[i][j] = a[i][j];
                for (int k = 0; k < i; k++)
                {
                    //deduct from the current l cell the value of these 2 values multiplied
                    u[i][j] -= l[i][k] * u[k][j];
                }
            }
	 
            //for each row...
            //rows are split into seperate threads for processing
  
            #pragma omp taskloop
            for (int j = i+1; j < size; j++){
                //if j is smaller than i, set u's current index to 0

                //otherwise, do some math to get the right value
                l[j][i] = a[j][i] / u[i][i];
                for (int k = 0; k < i; k++)
                {
                    l[j][i] -= ((l[j][k] * u[k][i]) / u[i][i]);
                }
            
            }
        }
	}
    }
}

float check_diff(float **a, float **l, float **u, int n){
    double diff = 0;
    for(int i=0; i<n; i++){
        double s1=0;
        for(int j=0; j<n; j++){
            double s2 = 0;
            for(int k=0; k<n; k++){
                s2 += l[i][k] * u[k][j]; 
            } 
            s2 = a[i][j]-s2;
            s1 += s2*s2;
        }
        diff += sqrt(s1);
    }
    return diff;
}

//initialize the matrices
void initialize_matrices(float** a, float** l, float** u, int size)
{
    //for each row in the 2d array, initialize the values
    //values are processed by seperate threads
    #pragma omp for schedule(static)
    for (int i = 0; i < size; ++i)
    {
        a[i] = new float[size];
        l[i] = new float[size];
        u[i] = new float[size];
    }
}

//fill the array with random values (done for a)
void random_fill(float** matrix, int size)
{
    //fill a with random values
    cout << "Producing random values " << endl;
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            matrix[i][j] = ((rand()%10)+1) ;
        }
    }

    //Ensure the matrix is diagonal dominant to guarantee invertible-ness
    //diagCount well help keep track of which column the diagonal is in
    int diagCount = 0;
    float sum = 0;
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            //Sum all column vaalues
            sum += abs(matrix[i][j]);
        }
        //Remove the diagonal  value from the sum
        sum -= abs(matrix[i][diagCount]);
        //Add a random value to the sum and place in diagonal position
        matrix[i][diagCount] = sum + ((rand()%5)+1);
        ++diagCount;
        sum = 0;
    }
}


int main(int argc, char** argv)
{
    double runtime;
    int numThreads;
    //set how many threads you want to use
    omp_set_num_threads(atoi(argv[2]));
    //seed rng
    srand(1);

    //size of matrix
    int size = atoi(argv[1]);

    //initalize matrices
    float** a = new float* [size];
    float** l = new float* [size];
    float** u = new float* [size];
    initialize_matrices(a, l, u, size);
    //fill a with random values
    random_fill(a, size);
    //print A
    cout << "A Matrix: " << endl;
    //print_matrix(a, size);
    //do LU decomposition
    runtime = omp_get_wtime();
    l_u_d(a, l, u, size);
    //print l and u
    cout << "L Matrix: " << endl;
    //print_matrix(l, size);
    cout << "U Matrix:" << endl;
    //print_matrix(u, size);
    //get the runtime of the job
    runtime = omp_get_wtime() - runtime;
    cout<<check_diff(a, l, u, size)<<endl;
    cout << "Runtime: " << runtime << endl;
    return 0;
}
