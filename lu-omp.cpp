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

/* print the upper_triangular part of the matrix */
void print_upper_triangular_matrix(matrix &a){
    int n = a.rows;
    int m = a.cols;
    cout<<"[";
    for (int i = 0; i < n; i++){
        cout<<"[ ";
        
        for (int j = 0; j < i; j++){ 
            cout << 0 << " ,";
        }

        for (int j = i; j < m-1; j++){ 
            cout << setprecision(PRECISION) << a.mat[j][i] << " ,";
        }
        cout << setprecision(PRECISION) << a.mat[m-1][i] << " ]";
        if(i!=n-1) cout << ", ";
        cout << endl;
    }
}

/* alloc matrix as a series of n double*. n is number of matrix rows */
void initialise_matrix_rows(matrix &a, int n, int m){
    a.rows = n;
    a.cols = m;
    a.mat = (double**)malloc(sizeof(double*)*n);
}


void swap(matrix &a, int x, int y){
    double *temp = a.mat[x];
    a.mat[x] = a.mat[y];
    a.mat[y] = temp;
}

/*
    * a stores the reordered matrix and
    * l is the lower triangular matrix
    * p is the permutation matrix
    * pivot_index is the row index on which we have to pivot upon
    * n is matrix size
    
    * We find the max value from a set of columns and swap the two rows.
    * Since we are using a double pointer for matrices we can just swap the pointers which saves time
*/
void perform_pivoting(matrix &a, matrix &l, matrix &p, int pivot_index, int n){
    int max_row = pivot_index;
    double max_val = abs(a.mat[pivot_index][pivot_index]);
    for(int j = pivot_index; j < n; j++){
        if(max_val < abs(a.mat[j][pivot_index])){
            max_val = abs(a.mat[j][pivot_index]);
            max_row = j;    
        }
    }

    if(max_row != pivot_index){
        swap(a, pivot_index, max_row);
        for(int j=0; j < pivot_index; j++) {
            double temp =  l.mat[pivot_index][j];
            l.mat[pivot_index][j] = l.mat[max_row][j];
            l.mat[max_row][j] = temp;
        }
        double temp =  *p.mat[pivot_index];
        *p.mat[pivot_index] = *p.mat[max_row];
        *p.mat[max_row] = temp;
    }
}


/*  
    * perform the lu decomposition
    * Inputs: 
        matrices: a, a_org, l, p
        n: matrix size 
        nworkers: no of threads
        l is lower triangular matrix
        a is the matrix which is initialized with random values.
        a_org is the original a matrix without pivoting
        p is a 1-D permutation matrix and stores the pivoting info. p[i] signifies the column number = 1 for row = i for a sparse permutation matrix 
    
    * Updates the matrices l and a
    * let the upper triangular portion of a be u, 
        then l X u = p X a_org
      
*/
void lu_decomp(matrix &a, matrix &a_org, matrix &p, matrix &l, int n, int nworkers)
{   
    omp_set_num_threads(nworkers);

    initialise_matrix_rows(a, n, n);
    initialise_matrix_rows(a_org, n, n);
    initialise_matrix_rows(p, n, 1);
    initialise_matrix_rows(l, n, n);
    
    /*
        initializing the matrices. We are trying to perform row-wise computations.
        Since Linux uses first touch policy, the workers should touch the rows which they 
        are about to compute upon to make effective use of caches.
        Now malloc does not allot memory unless the memory locations are accessed.    
        So we try to iterate over all the row indices.

        Suppose there are 4 workers. 
        Thread 0 first touches the rows 0,4,8,12 ......
        Thread 1 first touches the rows 1,5,9,13 ......  
        Thread 2 first touches the rows 2,6,10,14 ......
        Thread 3 first touches the rows 3,7,11,15 ......  
    */

    #pragma omp parallel for schedule(static) default(none) shared(a, a_org, p, l, n, nworkers)
    for (int w = 0; w < nworkers; w++){
        for(int i = w; i < n; i += nworkers){
            unsigned int random = i;
            l.mat[i] = (double*)malloc(sizeof(double)*n);  
            a.mat[i] = (double*)malloc(sizeof(double)*n); 
            a_org.mat[i] = (double*)malloc(sizeof(double)*n); 
            p.mat[i] = (double*)malloc(sizeof(double)*1);
            for (int j = 0; j < n; j++){
                a.mat[i][j] = (rand_r(&random)%100+1);
                a_org.mat[i][j] = a.mat[i][j];
    		    l.mat[i][j] = 0;
            }
            *p.mat[i] = i;
            l.mat[i][i]=1;
        } 
    }   
     
            
    for (int i = 0; i < n; i++){

        /*
            Perform row pivoting. This can lead to a little performance overhead
            since some of the rows are reshuffled but row pivoting is not a significant
            portion of computation and not too many rows are swapped in every computation.
        */
        perform_pivoting(a, l, p, i, n);
        

        /* 
            diving work on rows assigned during allocation.
            In case when number of workers are more than number of rows below ith row,
            we should redivide the remaining work. This isn't the most cache-efficient allocation 
            but this happens when number of workers are <<< size of matrix. So this isn't a problem.
        */
        #pragma omp parallel default(none) shared(a, l, n, i, nworkers) 
        {
            #pragma omp for schedule(static) 
            for (int w = 0; w < min(n-i, nworkers); w++){
                int u_w = min(n-i, nworkers);
                int start = ((i+1)/u_w)*u_w + w;
                if(start < (i+1)) start += u_w;
                for(int j = start ; j < n; j += u_w){
                    l.mat[j][i] = a.mat[j][i]/a.mat[i][i];
                }
            }

            #pragma omp for schedule(static)     
            for (int w = 0; w < min(n-i, nworkers); w++){
                int u_w = min(n-i, nworkers);
                int start = ((i+1)/u_w)*u_w + w;
                if(start < i+1) start += u_w;
                for(int j = start ; j < n; j += u_w){
                    for (int k = i+1; k < n; k++){
                        a.mat[j][k] -= ((l.mat[j][i] * a.mat[i][k]));
                    }
                }
            }
        }
    }
}


/* 
    checking L2,1 norm between p X a, l X u
*/
double check_diff(matrix &a, matrix &p, matrix &l, matrix &u, int n, int nworkers){
    
    omp_set_num_threads(nworkers);
    
    double tot_diff = 0;
    
    /*
        row_diff is the sqrt of sum of squares of differences between pa[i][j] and lu[i][j] for row i and j = 0 to N.
        tot_diff is sum of all row_diff        
    */
    #pragma omp parallel for shared(a, p, l, u, n, nworkers) schedule(static) reduction(+: tot_diff)
    for(int i = 0; i < n; i++){ 
        double row_diff = 0;
        for(int j=0; j<n; j++){
            double diff = 0;
            for(int k=0; k<n and k<=j; k++){
                diff += l.mat[i][k] * u.mat[k][j]; 
            } 
            int t = *p.mat[i];
            diff -= a.mat[t][j];
            row_diff += diff * diff;
        }
        tot_diff += sqrt(row_diff);
    }
    return tot_diff;        
}


int main(int argc, char** argv)
{
    double clock_time;
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
    matrix p;
    matrix a;
    matrix l;
    
    clock_time = omp_get_wtime();
    lu_decomp(a, a_org, p, l, matrix_size, nworkers);
    execution_time = omp_get_wtime() - clock_time;
    cout << "Execution time for LU decomp: " << execution_time << endl;

    if(DIFF){
    	diff = check_diff(a_org, p, l, a, matrix_size, nworkers);
    }
    execution_time = omp_get_wtime() - clock_time;
    cout<<"Execution time with checking diff: " << execution_time << endl;
    
    if(VERBOSE){
        cout << "A Matrix:" << endl;
        print_matrix(a_org);    
        cout << "P Matrix:" << endl;
        print_matrix(p);    
        cout << "L Matrix:" << endl;
        print_matrix(l);
        cout << "U Matrix:" << endl;
        print_upper_triangular_matrix(a);
    }

    cout << "Total time: " << execution_time << " matrix size: " << matrix_size << " threads: " << nworkers << " diff: " << diff << endl;
    return 0;
}
