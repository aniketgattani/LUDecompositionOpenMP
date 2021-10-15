#include <iostream>
#include <iomanip>
#include <cmath>
#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>  
#include <math.h>

using namespace std;

void print_matrix(double**, int);

//print the matrix out
void print_matrix(double** matrix, int size)
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

//fill the array with random values (done for a)
void random_fill(double** matrix, int size)
{
    //fill a with random values
    cout << "Producing random values " << endl;

  //#pragma omp taskloop grainsize(10) 
   for (int i = 0; i < size; i++)
    {
       srand(i+1);
       for (int j = 0; j < size; j++)
        {
            matrix[i][j] = ((rand()%10)+1) ;
        }
    }

    for(int i = 0; i < size; i++){
        double max = 0;
        int maxi = 0;
        for(int j = i; j < size; j++){
            if(max < matrix[j][i]){
                max = matrix[j][i];
                maxi = j;    
            }
        }    
        if(maxi!=i){
            double *temp = matrix[i];
	    matrix[i] = matrix[maxi];
	    matrix[maxi] = temp;    
        }
        
    }
}


//initialize the matrices
void initialize_matrices(double** a, double** l, double** u, int size)
{
    //for each row in the 2d array, initialize the values
    //values are processed by seperate threads
    //#pragma omp taskloop grainsize(10)	
    for (int i = 0; i < size; ++i)
    {
        a[i] = new double[size];
        l[i] = new double[size];
        u[i] = new double[size];
    }

    random_fill(a, size);

}

//do LU decomposition
//a is the matrix that will be split up into l and u
//array size for all is size x size
void l_u_d(double** a, double** l, double** u, int size, int threads)
{
    //initialize a simple lock for parallel region
    omp_lock_t lock;

    omp_init_lock(&lock);
    //for each column...
    //make the for loops of lu decomposition parallel. Parallel region
    #pragma omp parallel default(none) shared(a,l,u,size,threads)
    {
        #pragma omp single
	    {

            initialize_matrices(a, l, u, size);
            
        	for (int i = 0; i < size; i++){
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

double check_diff(double **a, double **l, double **u, int n){
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





int main(int argc, char** argv)
{
    double runtime;
    int numThreads = atoi(argv[2]);
    int VERBOSE = 0;
    if(argc >= 4 ) VERBOSE = atoi(argv[3]);
    
    //set how many threads you want to use
    omp_set_num_threads(numThreads);
    //seed rng

    //size of matrix
    int size = atoi(argv[1]);

    //initalize matrices
    double** a = new double* [size];
    double** l = new double* [size];
    double** u = new double* [size];

    
    runtime = omp_get_wtime();
    l_u_d(a, l, u, size, numThreads);
    runtime = omp_get_wtime() - runtime;
    
    //print A
    if(VERBOSE){
        cout << "A Matrix: " << endl;
        print_matrix(a, size);    
        //print l and u
        cout << "L Matrix: " << endl;
        print_matrix(l, size);
        cout << "U Matrix:" << endl;
        print_matrix(u, size);
    }

    //get the runtime of the job
    //cout<<check_diff(a, l, u, size)<<endl;
    cout << "Runtime: " << runtime << endl;
    return 0;
}
