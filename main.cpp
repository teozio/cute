#include <iostream>
#include <cstdlib>

#include <mpi.h>

#include "Matrix.h"

#define ERROR_MPI 100


using namespace std;

int main(int argc, char *argv[]){

  int info;

  info = MPI_Init ( &argc, &argv );
  if(info != 0){
    cerr << "MPI_Init error in main()" << endl;
    exit(ERROR_MPI);
  }

  // dimensions of the matrix
  int dims[2] = {25,25};

  Matrix *A = new Matrix(dims);

  A->info();
  //A->set_to(1.0);
  A->print();

  double val = A->min();
  cout << val << endl;
  
  delete A;

  info = MPI_Finalize();

  return EXIT_SUCCESS;
}
