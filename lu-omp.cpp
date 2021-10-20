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

void lu_decomp(matrix &a, matrix &l, matrix &u, int n, int nworkers)
{   
    omp_set_num_threads(nworkers);

    a.rows = n;
    a.cols = n;
    a.mat = new double*[n];
    
    l.rows = n;
    l.cols = n;
    l.mat = new double*[n];

    u.rows = n;
    u.cols = n;
    u.mat = new double*[n];

    #pragma omp parallel default(none) shared(a, l, u, n, nworkers)
    {

        #pragma omp for schedule(static)
        for (int ii = 0; ii < nworkers; ++ii){
            for(int i = ii; i < n; i += nworkers){
                a.mat[i] = new double[n];
                l.mat[i] = new double[n];
                u.mat[i] = new double[n];  
            }            
        }


        #pragma omp for schedule(static) 
        for (int ii = 0; ii < nworkers; ++ii){
            for(int i = ii; i < n; i += nworkers){
                srand(i+1);
                for (int j = 0; j < n; j++){
                    a.mat[i][j] = ((rand()%100)+1) ;
                    u.mat[i][j] = a.mat[i][j];
                }
            }  
        }    
        
        for (int i = 0; i < n; i++){
            l.mat[i][i]=1; 
    	   
    	    #pragma omp for schedule(static) 
            for (int jj = 0; jj < min(n-i, nworkers); jj++){
                int st = (i+1/nworkers)*nworkers + jj;
                for(int j = st ; j < n; j+=nworkers){
                    double t = a.mat[i][j];
        	        for (int k = 0; k < i; k++){
                        t -= l.mat[i][k] * u.mat[k][j];
                    }
    		        u.mat[i][j] = t;
    		    
                }
            }
            	 
                
            #pragma omp for schedule(static)
            for (int jj = 0; jj < min(n-i, nworkers); jj++){
                int st = (i+1/nworkers)*nworkers + jj;
                for(int j = st ; j < n; j+=nworkers){
                    double t = a.mat[j][i] / u.mat[i][i];
                    for (int k = 0; k < i; k++){
                        t -= ((l.mat[j][k] * u.mat[k][i]) / u.mat[i][i]);
                    }
                    l.mat[j][i] = t;
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
    #pragma omp parallel for reduction(+: diff)
    //{

        //{
            //#pragma omp taskloop reduction(+: diff)
            for(int ii=0; ii<n/b; ii++){
            for(int i = ii*b; i < (ii+1)*b; i++){ 
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
            }}
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
    
    // parse the input
    n = atoi(argv[1]);
    nworkers = atoi(argv[2]);
    if(argc >= 4 ) VERBOSE = atoi(argv[3]);
    
     
    matrix a;
    matrix l;
    matrix u;

    
    time = omp_get_wtime();
    lu_decomp(a, l, u, n, nworkers);
    execution_time = omp_get_wtime() - time;
    cout << "Execution time for LU decomp: " << execution_time << endl;

    //diff = check_diff(a,l,u,n,nworkers);
    execution_time = omp_get_wtime() - time;
    cout<<"Execution time with checking diff: " << execution_time << endl;
    
    if(VERBOSE){
        cout << "A Matrix: " << endl;
        print_matrix(a);    
        cout << "L Matrix: " << endl;
        print_matrix(l);
        cout << "U Matrix:" << endl;
        print_matrix(u);
    }

    cout << "Total time: " << execution_time << " threads: " << nworkers << " diff: " << diff << endl;
    return 0;
}
