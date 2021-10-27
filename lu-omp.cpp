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


void initialise_matrix(matrix &a, int n, int m){
    a.rows = n;
    a.cols = m;
    a.mat = (double**)malloc(sizeof(double*)*n);
}

/*
    a stores the reordered matrix and
    p stores the original matrix 
    find the max value from a set of columns and replace the two rows
    since we are using a double pointer for matrices we can just swap the pointers which saves time
*/
void perform_pivoting(matrix &a, matrix &p, int n){
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
        }
    }
}

void swap(matrix &a, int x, int y){
    double *temp = a.mat[x];
    a.mat[x] = a.mat[y];
    a.mat[y] = temp;
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
void lu_decomp(matrix &a, matrix &a_org, matrix &p, matrix &l, int n, int nworkers)
{   
    omp_set_num_threads(nworkers);

    initialise_matrix(a, n, n);
    initialise_matrix(a_org, n, n);
    initialise_matrix(p, n, 1);
    initialise_matrix(l, n, n);
    
    /*
        initializing the matrices. We are trying to perform row-wise computations.
        Since Linux uses first touch policy, the workers should touch the rows which they 
        are about to compute upon to make effective use of caches.
        Now malloc does not allot memory unless the memory locations are accessed.    
        So we try to iterate over all the row indices.

        Suppose there are 4 workers. 
        Thread 0 first touches the rows 0,4,8,12 ......
        Thread 1 first touches the rows 1,5,9,13 ......   
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
     
    /*
        Perform row pivoting. This can lead to a little performance overhead
        since some of the rows are reshuffled but row pivoting is not a significant
        portion of computation and not too many rows are swapped in every computation.
    */
    // perform_pivoting(a, p, n);
        
    for (int i = 0; i < n; i++){

        int max_row = i;
        double max_val = a.mat[i][i];
        for(int j = i; j < n; j++){
            if(max_val < a.mat[j][i]){
                max_val = a.mat[j][i];
                max_row = j;    
            }
        }

        if(max_row != i){
            swap(a, i, max_row);
            for(int j=0; j < i; j++) {
                double temp =  l.mat[i][j];
                l.mat[i][j] = l.mat[max_row][j];
                l.mat[max_row][j] = temp;
            }
            double temp =  *p.mat[i];
            *p.mat[i] = *p.mat[max_row];
            *p.mat[max_row] = temp;
        }

        /* 
            diving work on rows assigned during allocation.
            In case when number of workers are more than number of rows below ith row,
            we should redivide the remaining work. This isn't the most cache-efficient allocation 
            but when number of workers are <<< size of matrix. So this isn't a problem.
        */
        #pragma omp parallel for default(none) shared(a, l, n, i, nworkers) schedule(static) 
        for (int w = 0; w < min(n-i, nworkers); w++){
            int u_w = min(n-i, nworkers);
            int start = ((i+1)/u_w)*u_w + w;
            if(start < (i+1)) start += u_w;
            for(int j = start ; j < n; j += u_w){
                l.mat[j][i] = a.mat[j][i]/a.mat[i][i];
            }
        }        	 
            
        #pragma omp parallel for default(none) shared(a, l, n, i, nworkers) schedule(static)
        for (int w = 0; w < min(n-i, nworkers); w++){
            int u_w = min(n-i, nworkers);
            int start = ((i+1)/u_w)*u_w + w;
            if(start < i+1) start += u_w;
            for(int j = start ; j < n; j += u_w){
                /*  
                    the access pattern here uses l in row pattern. 
                    Hence, we should use l as it is. Again, u is used in
                    column pattern, so using u_t makes sense.
                */

                for (int k = i+1; k < n; k++){
                    a.mat[j][k] -= ((l.mat[j][i] * a.mat[i][k]));
                }
            }
        }
    }
}


/* 
    checking L2,1 norm between a, l, u
*/
double check_diff(matrix &a, matrix &p, matrix &l, matrix &u, int n, int nworkers){
    
    omp_set_num_threads(nworkers);
    
    double tot_diff = 0;
    
    /*
        row_diff is the sqrt of sum of squares of differences between A[i][j] and LU[i][j] for one row.
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
    matrix p;
    matrix a;
    matrix l;
    /* u_t is the u transpose matrix */
    //matrix u_t;

    time = omp_get_wtime();
    lu_decomp(a, a_org, p, l, matrix_size, nworkers);
    execution_time = omp_get_wtime() - time;
    cout << "Execution time for LU decomp: " << execution_time << endl;

    if(DIFF){
    	diff = check_diff(a_org, p, l, a, matrix_size, nworkers);
    }
    execution_time = omp_get_wtime() - time;
    cout<<"Execution time with checking diff: " << execution_time << endl;
    
    if(VERBOSE){
        cout << "A Matrix:" << endl;
        print_matrix(p);    
        cout << "PA Matrix:" << endl;
        print_matrix(p);    
        cout << "L Matrix:" << endl;
        print_matrix(l);
        cout << "U Matrix:" << endl;
        print_transpose_matrix(a);
    }

    cout << "Total time: " << execution_time << " threads: " << nworkers << " diff: " << diff << endl;
    return 0;
}
