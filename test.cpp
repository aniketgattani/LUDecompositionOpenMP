#include <iostream>
#include <iomanip>
#include <cmath>
#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>  
#include <math.h>

using namespace std;

#define PRECISION 6
typedef struct {double** mat; int rows; int cols;} matrix;


void print_matrix(matrix &a){
    int n = a.rows;
    int m = a.cols;
    cout<<"[";
    for (int i = 0; i < n; i++){
        cout<<"[ ";
        for (int j = 0; j < m-1; j++){ 
            cout << setprecision(PRECISION) << a.mat[i][j] << ' ';
        }
        cout << setprecision(PRECISION) << a.mat[i][m-1] << " ]";
        if(i!=n-1) cout << ", ";
        cout << endl;
    }
}

void fill_matrix(matrix &A){

    int n = A.rows;
    int m = A.cols;
    
  //#pragma omp taskloop grainsize(10) 
   for (int i = 0; i < n; i++){
       srand(i+1);
       for (int j = 0; j < m; j++){
            A.mat[i][j] = ((rand()%100)+1) ;
        }
    }

    for(int i = 0; i < n; i++){
        double max = 0;
        int maxi = 0;
        for(int j = i; j < m; j++){
            if(max < A.mat[j][i]){
                max = A.mat[j][i];
                maxi = j;    
            }
        }    
        if(maxi!=i){
            double *temp = A.mat[i];
	        A.mat[i] = A.mat[maxi];
	        A.mat[maxi] = temp;    
        }
        
    }
}


void initialise_matrices(matrix &a, matrix &l, matrix &u, int n){
    a.rows = n;
    a.cols = n;
    a.mat = new double*[n];
    
    l.rows = n;
    l.cols = n;
    l.mat = new double*[n];

    u.rows = n;
    u.cols = n;
    u.mat = new double*[n];

    for (int i = 0; i < n; ++i)
    {
        a.mat[i] = new double[n];
        l.mat[i] = new double[n];
        u.mat[i] = new double[n];
    }
}

void lu_decomp(matrix &a, int n, int nworkers)
{   
    omp_set_num_threads(nworkers);
    a.mat = new double*[n];
    long long tot = 0;
    #pragma omp parallel default(none) shared(a, n) reduction(+: tot)
    {
        
        #pragma omp for schedule(static) 
        for(int i = 0; i < n; ++i){
            a.mat[i] = new double[n];
        }

    	for (int i = 0; i < n; i++){   
            #pragma omp for schedule(static)
            for(int j = 0; j < n; j++){
                for(int k = 0; k < n; k++){
                    a.mat[j][k] = 1;
                    tot++; 
                }
            }   
        }

        
    }
    cout<<tot<<endl;
}



int main(int argc, char** argv)
{
    double time;
    double execution_time;
    int n;
    int nworkers;
    
    // parse the input
    n = atoi(argv[1]);
    nworkers = atoi(argv[2]);
    
    matrix a;
    
    time = omp_get_wtime();
    lu_decomp(a, n, nworkers);
    execution_time = omp_get_wtime() - time;
    cout << "Execution time for LU decomp: " << execution_time << endl;

    cout << "Total time: " << execution_time << " threads: " << nworkers << endl;
    return 0;
}
