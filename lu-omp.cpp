#include <iostream>
#include <omp.h>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include "timer.h" // to calculate time taken by program in UNIX

using namespace std;

#define DEFAULT_VAL 0.0

int value = 0;

typedef struct { vector<vector<double>> mat; } matrix;

void create_matrix(matrix *A, int n){
    A->mat.resize(n);
    for(int i=0; i<n; i++){
        A->mat[i].resize(n, DEFAULT_VAL);
    } 
}

void swap_matrix_rows(matrix *P, int x, int y){
    int n = P->mat.size();
    for(int i = 0; i < n; i++){
        swap(P->mat[x][i], P->mat[y][i]);
    }
}

void print_matrix(matrix *A){
    int n = A->mat.size();
    cout<<"[";
    for(int i=0; i<n; i++){
        cout<<"[";
        for(int j=0; j<n; j++){
            cout<<A->mat[i][j]<<", "; 
        }       
        cout<<"],"<<endl;
    }   
    cout<<"]"<<endl;
}

void create_matrix_mult(matrix *A, matrix *B, matrix *C){
    int n = A->mat.size();
    create_matrix(C, n);
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            for(int k=0; k<n; k++){
                C->mat[i][j] += A->mat[i][k] * B->mat[k][j];
            }
        }
    }
}

void create_identity_matrix(matrix *P, int n){
    create_matrix(P, n);
    for(int i=0; i < n; i++){
        P->mat[i][i] = 1.0;
    }
}

void create_permutation_matrix(matrix *P, matrix *A){
    int n = A->mat.size();
    create_identity_matrix(P, n);

    for(int k = 0; k < n; k++){
        int max = 0;
        int maxi = 0;
        for(int i = k; i < n; i++){
            if(max < A->mat[i][k]){
                max = A->mat[i][k];
                maxi = i;    
            }
        }    
        
        swap_matrix_rows(P, k, maxi);
    }
}


bool is_singular_matrix(matrix *P){
    int n = P->mat.size();

    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            if(i==j and P->mat[i][j]==0){
                return false;
            }
            if(i!=j and P->mat[i][j]!=0){
                return false;
            }
        }
    }
    return true;
}

void lu_decomp(matrix *A, matrix *L, matrix *U, int n){
    
    //#pragma omp parallel default(none)
    
    for(int k = 0; k < n; k++){
        
        U->mat[k][k] = A->mat[k][k];
        
        /*  A = LU;
            Hence, A[i,j] = L[i,k]*U[k,j];
        */
        for(int i=0; i < k+1; i++){
            double s1 = 0;
            for(int j=0; j < i; j++){
                s1 += L->mat[i][j] * U->mat[j][k];
            }
            U->mat[i][k] = A->mat[i][k] - s1;
        }
        
        for(int i = k ; i < n; i++){
            double s2 = 0;
            for(int j=0; j < i; j++){
                s2 += L->mat[i][j] * U->mat[j][k];
            }
            L->mat[i][k] = (A->mat[i][k] - s2) / U->mat[k][k];
        }
        
    }
       
}
void perform_decomposition(int n, int nworkers){

    omp_set_num_threads(nworkers);

    matrix A, P, PA, L, U;

    create_matrix(&A, n);
    // {{{7, 3, -1, 2}, {3, 8, 1, -4}, {-1, 1, 4, -1}, {2, -4, -1, 6}}};

    for(int i=0; i < n; i++){
        for(int j=0; j < n; j++){
            A.mat[i][j] =  rand()%100 + 1;
        }
    }
    
    create_identity_matrix(&L, n);
    create_matrix(&U, n);
    create_permutation_matrix(&P, &A);
    create_matrix_mult(&P, &A, &PA);

    timer_start();

    lu_decomp(&PA, &L, &U, n);

    double execution_time = timer_elapsed();

    cout<<"Time taken: "<< execution_time << " with workers: "<<nworkers<<endl;

    cout<<"A"<<endl;
    print_matrix(&A);
    cout<<"PA"<<endl;
    print_matrix(&PA);
    cout<<"L"<<endl;
    print_matrix(&L);
    cout<<"U"<<endl;
    print_matrix(&U);
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
