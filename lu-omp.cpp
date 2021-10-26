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

/* print the matrix */
void print_matrix(matrix &a){
    int n = a.rows;
    int m = a.cols;
    cout<<"[";
    for (int i = 0; i < n; i++){
        cout<<"[ ";
        for (int j = 0; j < m-1; j++){ 
            cout << setprecision(PRECISION) << a.mat[i][j] << " ,";
        }
        cout << setprecision(PRECISION) << a.mat[i][m-1] << " ]";
        if(i!=n-1) cout << ", ";
        cout << endl;
    }
}

/* print the transpose matrix. Useful for printing transpose(u) */
void print_transpose_matrix(matrix &a){
    int n = a.rows;
    int m = a.cols;
    cout<<"[";
    for (int i = 0; i < n; i++){
        cout<<"[ ";
        for (int j = 0; j < m-1; j++){ 
            cout << setprecision(PRECISION) << a.mat[j][i] << " ,";
        }
        cout << setprecision(PRECISION) << a.mat[m-1][i] << " ]";
        if(i!=n-1) cout << ", ";
        cout << endl;
    }
}


void initialise_matrix(matrix &a, int n){
    a.rows = n;
    a.cols = n;
    a.mat = (double**)malloc(sizeof(double*)*n);
}

/*
    a stores the reordered matrix and
    a_org stores the original matrix 
    find the max value from a set of columns and replace the two rows
    since we are using a double pointer for matrices we can just swap the pointers which saves time
*/
void perform_pivoting(matrix &a, matrix &a_org, int n){
    for(int i = 0; i < n; i++){
        double max = 0;
        int maxi = 0;
        for(int j = i; j < n; j++){
            if(max < a.mat[j][i]){
                max = a.mat[j][i];
                maxi = j;    
            }
        }

        if(maxi!=i){
            double *temp = a.mat[i];
            a.mat[i] = a.mat[maxi];
            a.mat[maxi] = temp; 
            temp = a_org.mat[i];
            a_org.mat[i] = a_org.mat[maxi];
            a_org.mat[maxi] = temp; 
        }
    }
}

/*  
    perform the lu decomposition
    Inputs: 
        matrices: a, l, u_t 
        n: matrix size 
        nworkers: no of threads
    Updates the matrices l and u_t
    l is lower triangular
    u_t is the transpose of an upper triangular matrix 
*/
void lu_decomp(matrix &a, matrix &a_org, matrix &l, matrix &u_t, int n, int nworkers)
{   
    omp_set_num_threads(nworkers);

    initialise_matrix(a, n);
    initialise_matrix(a_org, n);
    initialise_matrix(l, n);
    initialise_matrix(u_t, n);
    
    /*
        initializing the matrices. We are trying to perform row-wise computations.
        Since Linux uses first touch policy, the workers should touch the rows which they 
        first touched to make effective use of caches.
        Now malloc does not allot memory unless the memory locations are accessed.    
        So we try to iterate over all the row indices.

        Suppose there are 4 workers. 
        Thread 0 first touches the rows 0,4,8,12 ......
        Thread 1 first touches the rows 1,5,9,13 ......   
    */

    #pragma omp parallel for schedule(static) default(none) shared(a, a_org, l, u_t, n, nworkers)
    for (int w = 0; w < nworkers; w++){
        for(int i = w; i < n; i += nworkers){
            unsigned int random = i;
            l.mat[i] = (double*)malloc(sizeof(double)*n); 
            u_t.mat[i] = (double*)malloc(sizeof(double)*n); 
            a.mat[i] = (double*)malloc(sizeof(double)*n); 
            a_org.mat[i] = (double*)malloc(sizeof(double)*n);
            for (int j = 0; j < n; j++){
                a.mat[i][j] = (rand_r(&random)%100+1);
                a_org.mat[i][j] = a.mat[i][j];
    		    u_t.mat[i][j] = 0;
    		    l.mat[i][j] = 0;
            }
            l.mat[i][i]=1;
        } 
    }   
     
    /*
        Perform row pivoting. This can lead to a little performance overhead
        since some of the rows are reshuffled but row pivoting is not a significant
        portion of computation and not too many rows are swapped in every computation.
    */
    perform_pivoting(a, a_org, n);
        
    for (int i = 0; i < n; i++){
        /* 
            diving work on rows assigned during allocation.
            In case when number of workers are more than number of rows below ith row,
            we should redivide the remaining work. This isn't the most cache-efficient allocation 
            but when number of workers are <<< size of matrix. So this isn't a problem.
        */
        #pragma omp parallel for default(none) shared(a, l, u_t, n, i, nworkers) schedule(static) 
        for (int w = 0; w < min(n-i, nworkers); w++){
            int u_w = min(n-i, nworkers);
            int start = (i/u_w)*u_w + w;
            if(start < i) start += u_w;
            for(int j = start ; j < n; j += u_w){
                double t = a.mat[i][j];
                /*  
                    the access pattern here uses u in column pattern. 
                    Hence, we should use u_t matrix 
                */
                for (int k = 0; k < i; k++){
                    t -= l.mat[i][k] * u_t.mat[j][k];
                }
                u_t.mat[j][i] = t;
            }
        }        	 
            
        #pragma omp parallel for default(none) shared(a, l, u_t, n, i, nworkers) schedule(static)
        for (int w = 0; w < min(n-i, nworkers); w++){
            int u_w = min(n-i, nworkers);
            int start = ((i+1)/u_w)*u_w + w;
            if(start < i+1) start += u_w;
            for(int j = start ; j < n; j += u_w){
                double t = a.mat[j][i] / u_t.mat[i][i];
                /*  
                    the access pattern here uses l in row pattern. 
                    Hence, we should use l as it is. Again, u is used in
                    column pattern, so using u_t makes sense.
                */
                for (int k = 0; k < i; k++){
                    t -= ((l.mat[j][k] * u_t.mat[i][k]) / u_t.mat[i][i]);
                }
                l.mat[j][i] = t;
            }
        }
    }
}


/* 
    checking L2,1 norm between a, l, u
*/
double check_diff(matrix &a, matrix &l, matrix &u_t, int n, int nworkers){
    
    omp_set_num_threads(nworkers);
    
    double tot_diff = 0;
    
    /*
        row_diff is the sqrt of sum of squares of differences between A[i][j] and LU[i][j] for one row.
        tot_diff is sum of all row_diff        
    */
    #pragma omp parallel for shared(a, l, u_t, n, nworkers) schedule(static) reduction(+: tot_diff)
    for(int i = 0; i < n; i++){ 
        double row_diff = 0;
        for(int j=0; j<n; j++){
            double diff = 0;
            for(int k=0; k<n; k++){
                diff += l.mat[i][k] * u_t.mat[j][k]; 
            } 
            diff -= a.mat[i][j];
            row_diff += diff * diff;
        }
        tot_diff += sqrt(row_diff);
    }
    return tot_diff;        
}


int main(int argc, char** argv)
{
    double time;
    double execution_time;
    double diff;
    int matrix_size;
    int nworkers;
    int VERBOSE = 0;
    int DIFF = 0;
    
    // parse the input
    matrix_size = atoi(argv[1]);
    nworkers = atoi(argv[2]);
    if(argc >= 4 ) VERBOSE = atoi(argv[3]);
    if(argc >= 5) DIFF = atoi(argv[4]);
         
    matrix a_org;
    matrix a;
    matrix l;
    /* u_t is the u transpose matrix */
    matrix u_t;

    time = omp_get_wtime();
    lu_decomp(a, a_org, l, u_t, matrix_size, nworkers);
    execution_time = omp_get_wtime() - time;
    cout << "Execution time for LU decomp: " << execution_time << endl;

    if(DIFF){
    	diff = check_diff(a, l, u_t, matrix_size, nworkers);
    }
    execution_time = omp_get_wtime() - time;
    cout<<"Execution time with checking diff: " << execution_time << endl;
    
    if(VERBOSE){
        cout << "A Matrix:" << endl;
        print_matrix(a_org);    
        cout << "PA Matrix:" << endl;
        print_matrix(a);    
        cout << "L Matrix:" << endl;
        print_matrix(l);
        cout << "U Matrix:" << endl;
        print_transpose_matrix(u_t);
    }

    cout << "Total time: " << execution_time << " threads: " << nworkers << " diff: " << diff << endl;
    return 0;
}
