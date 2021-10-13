#include <iostream>
#include <omp.h>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include "timer.h" // to calculate time taken by program in UNIX
#include <utility>

using namespace std;

#define DEFAULT_VAL 0.0
int VERBOSE = 0;
int BLOCK_SIZE = 2 ;


typedef struct { vector<vector<double>> mat; } matrix;

void create_matrix(matrix &A, int n, int m){
    A.mat.resize(n);
    for(int i=0; i<n; i++){
        A.mat[i].resize(m, DEFAULT_VAL);
    } 
}

void swap_matrix_rows(matrix &P, int x, int y){
    int n = P.mat.size();
    for(int i = 0; i < n; i++){
        swap(P.mat[x][i], P.mat[y][i]);
    }
}

void print_matrix(matrix &A){
    int n = A.mat.size();
    cout<<"[";
    for(int i=0; i<n; i++){
        cout<<"[";
        for(int j=0; j<n-1; j++){
            cout<<A.mat[i][j]<<", "; 
        }       
        cout<<A.mat[i][n-1];
        cout<<"],"<<endl;
    }   
    cout<<"]"<<endl;
}

void create_matrix_mult(matrix &A, matrix &B, matrix &C){
    int n = A.mat.size();
    int m = B.mat[0].size();
    int m1 = A.mat[0].size();
    
    create_matrix(C, n, m);

    #pragma omp parallel default(none) shared(A, B, C, n, m, m1)
    {
        #pragma omp for
        for(int i=0; i<n; i++){
            for(int j=0; j<m; j++){
                for(int k=0; k<m1; k++){
                    C.mat[i][j] += A.mat[i][k] * B.mat[k][j];
                }
            }
        }    
    }
    
}
// A = A - B
void matrix_subtract(matrix &A, matrix &B){
    int n = A.mat.size();
    int m = A.mat[0].size();

    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            A.mat[i][j] = A.mat[i][j] - B.mat[i][j];
        }
    }   
}

void create_identity_matrix(matrix &P, int n){
    create_matrix(P, n, n);
    for(int i=0; i < n; i++){
        P.mat[i][i] = 1.0;
    }
}

void create_permutation_matrix(matrix &P, matrix &A){
    int n = A.mat.size();
    create_identity_matrix(P, n);

    for(int k = 0; k < n; k++){
        double max = 0;
        int maxi = 0;
        for(int i = k; i < n; i++){
            if(max < A.mat[i][k]){
                max = A.mat[i][k];
                maxi = i;    
            }
        }    
        
        swap_matrix_rows(P, k, maxi);
    }
}
void copy_matrix(matrix &A, matrix &B, int ax, int ay, int bx, int by, int rows, int cols){
    int b = BLOCK_SIZE;
    #pragma omp parallel default(none) shared(A, B, ax, ay, bx, by, rows, cols, b) 
    {
        #pragma omp single 
        {
            for(int i=0; i < rows/b; i++){
                for(int j=0; j < cols/b; j++){
                
                    #pragma omp task firstprivate(i, j)
                    {
                        for(int ii = i * b; ii < (i+1) * b; ii++){
                            for(int jj = j * b; jj < (j+1) * b; jj++){
                                #pragma omp atomic write
                                B.mat[bx+ii][by+jj] = A.mat[ax+ii][ay+jj];
                            }
                        }        
                    }
                }
            } 

            for(int ii = b*(rows/b) ; ii < rows; ii++){
                for(int jj = 0;  jj < cols; jj++){
                    #pragma omp atomic write
                    B.mat[bx+ii][by+jj] = A.mat[ax+ii][ay+jj];
                }
            }

            for(int jj = b*(cols/b) ; jj < cols; jj++){
                for(int ii = 0;  ii < rows; ii++){
                    #pragma omp atomic write
                    B.mat[bx+ii][by+jj] = A.mat[ax+ii][ay+jj];
                }
            }

            #pragma omp taskwait

        }   
    }
}

void divide_matrix(matrix &A, matrix &A00,  matrix &A01, matrix &A10, matrix &A11, int b){
    int n = A.mat.size();

    create_matrix(A00, b, b);
    create_matrix(A01, b, n-b);
    create_matrix(A10, n-b, b);
    create_matrix(A11, n-b, n-b);
    
    #pragma omp parallel default(none) shared(A, A00, A01, A10, A11, b, n)
    {   
        
        #pragma omp single
        {
            #pragma omp task
            {
                copy_matrix(A, A11, b, b, 0, 0, n-b, n-b);
            }        
            
            copy_matrix(A, A01, 0, b, 0, 0, b, n-b);
            copy_matrix(A, A10, b, 0, 0, 0, n-b, b);        
            copy_matrix(A, A00, 0, 0, 0, 0, b, b);
        
            #pragma omp taskwait    
        }
        
        
    }
    
}


bool is_singular_matrix(matrix &P){
    int n = P.mat.size();

    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            if(i==j and P.mat[i][j]==0){
                return false;
            }
            if(i!=j and P.mat[i][j]!=0){
                return false;
            }
        }
    }
    return true;
}

void findLU(matrix &A, matrix &L, matrix &U){
    int n = A.mat.size();
    for(int k = 0; k < n; k++){    
        U.mat[k][k] = A.mat[k][k];

        /*  A = LU;
            Hence, A[i,j] = L[i,k]*U[k,j];
        */
        for(int i=0; i < k+1; i++){
            double s1 = 0;
            for(int j=0; j < i; j++){
                s1 += L.mat[i][j] * U.mat[j][k];
            }
            U.mat[i][k] = A.mat[i][k] - s1;
        }
        
        for(int i = k ; i < n; i++){
            double s2 = 0;
            for(int j=0; j < i; j++){
                s2 += L.mat[i][j] * U.mat[j][k];
            }
            L.mat[i][k] = (A.mat[i][k] - s2) / U.mat[k][k];
        }
        
    }
}

void findU(matrix &A, matrix &L, matrix &U){
    int n = U.mat.size();
    int m = U.mat[0].size();
    
    for(int i=0; i < n; i++){
        for(int j=0; j < m; j++){
            double s1 = 0;
            for(int k=0; k < i; k++){
                s1 += L.mat[i][k] * U.mat[k][j];    
            }
            U.mat[i][j] = (A.mat[i][j] - s1)/L.mat[i][i];    
        }
    }
}

void findL(matrix &A, matrix &L, matrix &U){
    int n = L.mat.size();
    int m = L.mat[0].size();
    
    for(int i=0; i < n; i++){
        for(int j=0; j < m; j++){
            double s1 = 0;
            for(int k=0; k < j; k++){
                s1 += L.mat[i][k] * U.mat[k][j];    
            }
            L.mat[i][j] = (A.mat[i][j] - s1)/U.mat[j][j];    
        }
    }
}

void combine(matrix &A, matrix &A00, matrix &A01, matrix &A10, matrix &A11, int b){
    int n = A.mat.size();
    copy_matrix(A00, A, 0, 0, 0, 0, b, b);
    copy_matrix(A01, A, 0, 0, 0, b, b, n-b);
    copy_matrix(A10, A, 0, 0, b, 0, n-b, b);
    copy_matrix(A11, A, 0, 0, b, b, n-b, n-b);
}

void lu_decomp(matrix &A, matrix &L, matrix &U, int n){
    int b = BLOCK_SIZE;

    #pragma omp parallel default(none) shared(A, L, U, n, b)
    {            
        #pragma omp single
        {
            if (n > b){

                matrix A00, A01, A10, A11;
                matrix L00, L01, L10, L11;
                matrix U00, U01, U10, U11;
                
                // #pragma omp task shared(A00, A01, A10, A11, b)
                {
                    divide_matrix(A, A00, A01, A10, A11, b); 
                }

                // #pragma omp task shared(L00, L01, L10, L11, b)
                {
                    divide_matrix(L, L00, L01, L10, L11, b);
                }

                // #pragma omp task shared(U00, U01, U10, U11, b)
                {
                    divide_matrix(U, U00, U01, U10, U11, b);
                }

                // #pragma omp taskwait

                findLU(A00, L00, U00);
            
                #pragma omp task shared(A01, L00, U01)
                {
                    findU(A01, L00, U01);    
                }    

                #pragma omp task shared(A10, L10, U00)
                {
                    findL(A10, L10, U00);    
                }

                #pragma omp taskwait
                
                matrix L10U01;
                matrix P11;
                matrix P11A11;        
                
                create_matrix_mult(L10, U01, L10U01); 
                matrix_subtract(A11, L10U01);
                
                #pragma omp task shared(L11, U11, A11, b)
                {
                    lu_decomp(A11, L11, U11, n-b);
                }

                #pragma omp taskwait
            
                #pragma omp task shared(L00, L01, L10, L11)
                {
                    combine(L, L00, L01, L10, L11, b);
                }

                #pragma omp task shared(U00, U01, U10, U11)
                {
                    combine(U, U00, U01, U10, U11, b);
                }

                #pragma omp taskwait
            }

            else{
                findLU(A, L, U);
            }              
        }
    }
}

void perform_decomposition(int n, int nworkers){

    omp_set_num_threads(nworkers);

    matrix A, P, PA, L, U;

    timer_start();

    create_matrix(A, n, n);
    // {{{7, 3, -1, 2}, {3, 8, 1, -4}, {-1, 1, 4, -1}, {2, -4, -1, 6}}};


    #pragma omp parallel default(none) shared(n, A)
    {
        #pragma omp for
        for(int i=0; i < n; i++){
            srand(i);
            for(int j=0; j < n; j++) A.mat[i][j] =  rand()%100 + 1;
        }
        
    }

    
    create_identity_matrix(L, n);
    create_matrix(U, n, n);
    create_permutation_matrix(P, A);
    create_matrix_mult(P, A, PA);


    lu_decomp(PA, L, U, n);

    double execution_time = timer_elapsed();

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
