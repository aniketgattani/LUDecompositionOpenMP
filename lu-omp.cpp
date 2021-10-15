#include <iostream>
#include <omp.h>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "timer.h" // to calculate time taken by program in UNIX
#include <utility>

using namespace std;

#define DEFAULT_VAL 0.0
#define POS(X,Y,R) (X*R + Y)

int VERBOSE = 0;
int BLOCK_SIZE = 100 ;


typedef struct { vector<double> mat; int rows; int cols; } matrix;

void create_matrix(matrix &A, int n, int m){
    A.rows = n;
    A.cols = m;
    A.mat.resize(n*m, DEFAULT_VAL);
}

void swap_matrix_rows(matrix &P, int x, int y){
    int n = P.rows;
    for(int i = 0; i < n; i++){
        swap(P.mat[POS(x, i, n)], P.mat[POS(y, i, n)]);
    }
}

void print_matrix(matrix &A){
    int n = A.rows;
    cout<<"[";
    for(int i=0; i<n; i++){
        cout<<"[";
        for(int j=0; j<n-1; j++){
            cout<<A.mat[POS(i, j, n)]<<", "; 
        }       
        cout<<A.mat[POS(i,n-1,n)];
        cout<<"],"<<endl;
    }   
    cout<<"]"<<endl;
}

void create_matrix_mult(matrix &A, matrix &B, matrix &C){
    int n = A.rows;
    int m = B.cols;
    int m1 = A.cols;
    
    create_matrix(C, n, m);

    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            for(int k=0; k<m1; k++){
                C.mat[POS(i, j, n)] += A.mat[POS(i, k, n)] * B.mat[POS(k, j, m1)];
            }
        }
    }    
    
    
}
// A = A - B
void matrix_subtract(matrix &A, matrix &B){
    int n = A.rows;
    int m = A.cols;
    int b = BLOCK_SIZE;
	// #pragma omp single
    {
        for(int i=0; i < n; i++){
            for(int j=0; j < m; j++){
               
                // #pragma omp task firstprivate(i, j)
                A.mat[POS(i, j, n)] = A.mat[POS(i,j,n)] - B.mat[POS(i,j,n)];
            }
        }
                    
    } 
    
    // #pragma omp taskwait
}

void create_identity_matrix(matrix &P, int n){
    create_matrix(P, n, n);
    for(int i=0; i < n; i++){
        P.mat[POS(i,i,n)] = 1.0;
    }
}

void create_permutation_matrix(matrix &P, matrix &A){
    int n = A.rows;
    create_identity_matrix(P, n);

    for(int k = 0; k < n; k++){
        double max = 0;
        int maxi = 0;
        for(int i = k; i < n; i++){
            if(max < A.mat[POS(i,k,n)]){
                max = A.mat[POS(i,k,n)];
                maxi = i;    
            }
        }    
        
        swap_matrix_rows(P, k, maxi);
    }
}
void copy_matrix(matrix &A, matrix &B, int ax, int ay, int bx, int by, int rows, int cols){
    int b = BLOCK_SIZE;
    int na = A.rows;
    int nb = B.rows;
    // #pragma omp single 
    {
        for(int i=0; i < rows; i++){
            for(int j=0; j < cols; j++){
            
               // #pragma omp task firstprivate(i, j)
                B.mat[POS(bx+i, by+j, nb)] = A.mat[POS(ax+i, ay+j, na)];
            }
        } 

        // #pragma omp taskwait

    }   
}

void divide_matrix(matrix &A, matrix &A00,  matrix &A01, matrix &A10, matrix &A11, int b){
    int n = A.rows;

    // #pragma omp single
    {    
        create_matrix(A00, b, b);
	    create_matrix(A01, b, n-b);
	    create_matrix(A10, n-b, b);
        
        copy_matrix(A, A01, 0, b, 0, 0, b, n-b);
        copy_matrix(A, A10, b, 0, 0, 0, n-b, b);        
        copy_matrix(A, A00, 0, 0, 0, 0, b, b);
    
        // #pragma omp task 
        { 
	    	create_matrix(A11, n-b, n-b);
            copy_matrix(A, A11, b, b, 0, 0, n-b, n-b);
        }        

        // #pragma omp taskwait    
    }
        
}


void combine(matrix &A, matrix &A00, matrix &A01, matrix &A10, matrix &A11, int b){
    int n = A.cols;
    copy_matrix(A00, A, 0, 0, 0, 0, b, b);
    copy_matrix(A01, A, 0, 0, 0, b, b, n-b);
    copy_matrix(A10, A, 0, 0, b, 0, n-b, b);
    copy_matrix(A11, A, 0, 0, b, b, n-b, n-b);
}

void lu_decomp(matrix &A, matrix &L, matrix &U, int n){
    for(int i=0; i < n; i++){
        for(int j=i+1 ; j < n; j++){
            double fac =  U.mat[POS(j, i, n)]/U.mat[POS(i, i, n)];
            for(int k = i; k < n; k++){
                U.mat[POS(j, k, n)] -= U.mat[POS(i, k, n)] * fac;
            }   
            L.mat[POS(j, i, n)] =  fac;
        }
    }          
}

double check_diff(matrix &PA, matrix &L, matrix &U, int n){
    matrix LU;
    create_matrix_mult(L, U, LU);
    double diff = 0;
    for(int i=0;i<n;i++){
        double s1 = 0;
        for(int j=0; j<n; j++){
            double x = PA.mat[POS(i, j, n)] - LU.mat[POS(i, j, n)];
            s1 += x*x;
        }    
        s1 = sqrt(s1);
        diff += s1;
    }
    return diff;
}

void perform_decomposition(int n, int nworkers){

    omp_set_num_threads(nworkers);

    matrix A, P, PA, L, U;

    // {{{7, 3, -1, 2}, {3, 8, 1, -4}, {-1, 1, 4, -1}, {2, -4, -1, 6}}};

    timer_start();


    #pragma omp parallel default(none) shared(n, A, L, U, P, PA, VERBOSE, nworkers)
    {

        #pragma omp single
        {
            create_matrix(A, n, n);
            create_matrix(U, n, n);
            create_identity_matrix(L, n);
            
            // #pragma omp for
            for(int i=0; i < n; i++){
                srand(i+1);
                for(int j=0; j < n; j++) {
                    A.mat[POS(i, j, n)] =  rand()%100 + 1;
                }
            }

            #pragma omp task shared(P, A, PA) depend(out: PA)
            {
                create_permutation_matrix(P, A);
                create_matrix_mult(P, A, PA);
            }

            #pragma omp task shared(U, PA, n) depend(in: PA) depend(out: U)
            {
                copy_matrix(PA, U, 0, 0, 0, 0, n, n);
            }
            
            
            #pragma omp task shared(PA, L, U) depend(in: PA, U)
            {
                lu_decomp(PA, L, U, n);     
            }

            #pragma omp taskwait
        }    
    }


    if(VERBOSE){    
        cout<<"A"<<endl;
        print_matrix(A);
        cout<<"PA"<<endl;
        print_matrix(PA);
        cout<<"L"<<endl;
        print_matrix(L);
        cout<<"U"<<endl;
        print_matrix(U);
    }

    double execution_time = timer_elapsed();

    cout<<"Diff :" << check_diff(PA, L, U, n)<<endl;
    cout<<"Time taken: "<< execution_time << " with workers: "<<nworkers<<endl;

}

void usage(const char *name){
	std::cout << "usage: " << name << " matrix-size nworkers" << endl;
 	exit(-1);
}



int main(int argc, char **argv)
{
    const char *name = argv[0];

    if (argc < 3) usage(name);

    int matrix_size = atoi(argv[1]);

    int nworkers = atoi(argv[2]);

    if(argc >= 4)
    BLOCK_SIZE = atoi(argv[3]);
    
    if(argc >= 5)
    VERBOSE = atoi(argv[4]);

    std::cout << name << ": " 
        << matrix_size << " " << nworkers
        << std::endl;

    
    // srand = time(0);
    perform_decomposition(matrix_size, nworkers);
    

    // #pragma omp parallel
    // {
    //     int tid = omp_get_thread_num();
    //     int myN = 20 - tid;
    //     if (myN < 16) myN = 16;
    //     // #pragma omp critical
    //     value = tid; // data race
    //     printf("thread %d fib(%d) = %ld\n", tid, myN, res);
    // }
    return 0;
}
