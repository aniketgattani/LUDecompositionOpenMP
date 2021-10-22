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
            cout << setprecision(PRECISION) << a.mat[i][j] << " ,";
        }
        cout << setprecision(PRECISION) << a.mat[i][m-1] << " ]";
        if(i!=n-1) cout << ", ";
        cout << endl;
    }
}

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

void lu_decomp(matrix &a, matrix &aa, matrix &l, matrix &u, int n, int nworkers)
{   
    omp_set_num_threads(nworkers);

    a.rows = n;
    a.cols = n;
    a.mat = (double**)malloc(sizeof(double*)*n);

    aa.rows = n;
    aa.cols = n;
    aa.mat = (double**)malloc(sizeof(double*)*n);
    
    l.rows = n;
    l.cols = n;
    l.mat = (double**)malloc(sizeof(double*)*n);

    u.rows = n;
    u.cols = n;
    u.mat = (double**)malloc(sizeof(double*)*n);

    #pragma omp parallel default(none) shared(a, aa, l, u, n, nworkers)
    {

        #pragma omp for schedule(static)
        for (int ii = 0; ii < nworkers; ++ii){
            for(int i = ii; i < n; i += nworkers){
                a.mat[i] = new double[n];
                aa.mat[i] = new double[n];
                l.mat[i] = new double[n];
                u.mat[i] = new double[n];  
            }            
        }


        #pragma omp for schedule(static) 
        for (int ii = 0; ii < nworkers; ++ii){
            for(int i = ii; i < n; i += nworkers){
                srand(i+1);
                l.mat[i] = (double*)malloc(sizeof(double)*n); 
                u.mat[i] = (double*)malloc(sizeof(double)*n); 
                a.mat[i] = (double*)malloc(sizeof(double)*n); 
                aa.mat[i] = (double*)malloc(sizeof(double)*n); 
                for (int j = 0; j < n; j++){
                    a.mat[i][j] = ((rand()%100)+1);
        		    u.mat[i][j] = 0;
        		    l.mat[i][j] = 0;
                    aa.mat[i][j] = a.mat[i][j];
                }
                l.mat[i][i]=1;
            } 
        }   
    }  
        
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
            temp = aa.mat[i];
            aa.mat[i] = aa.mat[maxi];
            aa.mat[maxi] = temp; 
        }
    }

        
    for (int i = 0; i < n; i++){
        u.mat[i][i]=a.mat[i][i];
	    #pragma omp parallel for default(none) shared(a,l,u,n,i,nworkers) schedule(static) 
        for (int jj = 0; jj < min(n-i, nworkers); jj++){
            int nn = min(n-i, nworkers);
            int st = ((i+1)/nn)*nn + jj;
            if(st < i+1) st+=nn;
            for(int j = st ; j < n; j+=nn){
                l.mat[j][i] = a.mat[j][i]/u.mat[i][i];
                u.mat[i][j] = a.mat[i][j];
            }
        }        	 
            
        #pragma omp parallel for default(none) shared(a,l,u,n,i,nworkers) schedule(static)
        for (int jj = 0; jj < min(n-i, nworkers); jj++){
            int nn = min(n-i, nworkers);
            int st = ((i+1)/nn)*nn + jj;
            if(st < i+1) st+=nn;
            for(int j = st ; j < n; j+=nn){
                for (int k = i+1; k < n; k++){
                    a.mat[j][k] -= ((l.mat[j][i] * u.mat[i][k]));
                }
            }
        }
    }
}

double check_diff(matrix &a, matrix &l, matrix &u, int n, int nworkers){
    double diff = 0;
    omp_set_num_threads(nworkers);
    omp_set_nested(1);
 int b = 5;
    //{

        //{
        #pragma omp parallel for shared(a,l,u,n,nworkers) schedule(static) reduction(+: diff)
        for(int i = 0; i < n; i++){ 
            double s1=0;
                //#pragma omp parallel for reduction(+: s1)
	        for(int j=0; j<n; j++){
                double s2 = 0;
                for(int k=0; k<n; k++){
                    s2 += l.mat[i][k] * u.mat[k][j]; 
                } 
                s2 = a.mat[i][j]-s2;
                s1 += s2*s2;
            }
                diff += sqrt(s1);
        }
      // }
   //}
    return diff;        
}


int main(int argc, char** argv)
{
    double time;
    double execution_time;
    double diff;
    int n;
    int nworkers;
    int VERBOSE = 0;
    int DIFF = 0;
    
    // parse the input
    n = atoi(argv[1]);
    nworkers = atoi(argv[2]);
    if(argc >= 4 ) VERBOSE = atoi(argv[3]);
    if(argc >= 5) DIFF = atoi(argv[4]);
    
     
    matrix a;
    matrix aa;
    matrix l;
    matrix u;

    
    time = omp_get_wtime();
    lu_decomp(a, aa, l, u, n, nworkers);
    execution_time = omp_get_wtime() - time;
    cout << "Execution time for LU decomp: " << execution_time << endl;

    if(DIFF){
    	diff = check_diff(aa,l,u,n,nworkers);
    }
    execution_time = omp_get_wtime() - time;
    cout<<"Execution time with checking diff: " << execution_time << endl;
    
    if(VERBOSE){
        cout << "A Matrix: " << endl;
        print_matrix(aa);    
        cout << "L Matrix: " << endl;
        print_matrix(l);
        cout << "U Matrix:" << endl;
        print_matrix(u);
    }

    cout << "Total time: " << execution_time << " threads: " << nworkers << " diff: " << diff << endl;
    return 0;
}
