#include <iostream>
#include <cstring>
#include <cmath>
#include <cstdlib>

#include <mpi.h>

#include "Matrix.h"

using namespace std;

#define BLOCK_SIZE 10

#define ERROR_MPI 100
#define ERROR_ARRAY_MISSMATCH 101
#define ERROR_BLAS 102
#define ERROR_LAPACK 103
#define ERROR_NOT_CODED 104

// Constructor
Matrix::Matrix(int *dims){
  
  // assign dimensions
  dims_[0] = dims[0];
  dims_[1] = dims[1];

  // --------------------------------------------
  // find the number of blocks in each dimensions
  // --------------------------------------------
  int x_blocks = dims_[0]/BLOCK_SIZE;
  if( dims_[0]%BLOCK_SIZE != 0) x_blocks++;
  x_blocks_ = x_blocks;

  int y_blocks = dims_[1]/BLOCK_SIZE;
  if( dims_[1]%BLOCK_SIZE != 0) y_blocks++;
  y_blocks_ = y_blocks;

  num_total_blocks_ = x_blocks_ * y_blocks_;
  // --------------------------------------------

  // --------------------------------------------
  // assign sizes of blocks
  // --------------------------------------------
  x_block_size_ = new int [x_blocks];
  for(int i=0; i<x_blocks_; i++){
    if( dims_[0]%BLOCK_SIZE != 0 && i == x_blocks_-1 ) {
      x_block_size_[i] = dims_[0]%BLOCK_SIZE;
    } else {
      x_block_size_[i] = BLOCK_SIZE;
    }
  } 

  y_block_size_ = new int [y_blocks];
  for(int i=0; i<y_blocks_; i++){
    if( dims_[1]%BLOCK_SIZE != 0 && i == y_blocks_-1 ) {
      y_block_size_[i] = dims_[1]%BLOCK_SIZE;
    } else {
      y_block_size_[i] = BLOCK_SIZE;
    }
  } 
  // --------------------------------------------
  
  // --------------------------------------------
  // assign processors to blocks
  // --------------------------------------------
  int num_procs, info;

  info = MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  if(info != 0){
    cerr << "MPI_Comm_size error in Matrix::Matrix(int *)" << endl;
    exit(ERROR_MPI);
  }

  block_owner_ = new int [num_total_blocks_];
  int block = 0;
  while( block < num_total_blocks_ ){
    for(int j=0; j<num_procs; j++){
            
      if( block < num_total_blocks_ ){
        block_owner_[block] = j;
        block++;
      } else {
        break;
      }

    }
  }
  // --------------------------------------------

  // --------------------------------------------
  // count number of local blocks on this processor
  // --------------------------------------------
  int my_id;
  info = MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
  if(info != 0){
    cerr << "MPI_Comm_rank error in Matrix::Matrix(int *)" << endl;
    exit(ERROR_MPI);
  }

  num_local_blocks_ = 0;
  for(int i=0; i<num_total_blocks_; i++){
    if( block_owner_[i] == my_id ) num_local_blocks_++;
  }

  // allocate space for coordinates of local blocks
  // ---------------------------------------------
  block_x_coo_ = new int [num_local_blocks_];
  block_y_coo_ = new int [num_local_blocks_];

  int total_idx = 0;
  int local_idx = 0;

  for(int i=0; i<y_blocks_; i++){
    for(int j=0; j<x_blocks_; j++){

      if( block_owner_[total_idx] == my_id ) {   

        block_x_coo_[local_idx] = j;
        block_y_coo_[local_idx] = i;
        local_idx++;

      }      

      total_idx++;
    }

  }
  // --------------------------------------------
  
  // --------------------------------------------
  // allocate the local blocks
  // --------------------------------------------
  data_ = new double *[num_local_blocks_];

  for(int i=0; i<num_local_blocks_; i++){

    int x = block_x_coo_[i];
    int y = block_y_coo_[i];

    int nelements = x_block_size_[x] * y_block_size_[y];

    data_[i] = new double [nelements];

    memset(data_[i],0,nelements*sizeof(double));

  }
  // --------------------------------------------

}

// Destructor
Matrix::~Matrix(){

  delete [] x_block_size_;
  delete [] y_block_size_;

  delete [] block_owner_;

  delete [] block_x_coo_;
  delete [] block_y_coo_;

  // deallocate blocks
  for(int i=0; i<num_local_blocks_; i++){
    delete [] data_[i];
  }
  delete [] data_;

}

// Matrix information
void Matrix::info(){

  int info, id, num_procs;

  info = MPI_Comm_rank(MPI_COMM_WORLD, &id);
  if(info != 0){
    cerr << "MPI_Comm_rank error in Matrix::info()" << endl;
    exit(ERROR_MPI);
  }
  info = MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  if(info != 0){
    cerr << "MPI_Comm_size error in Matrix::info()" << endl;
    exit(ERROR_MPI);
  }

  // ----------------------------------------------
  // Print GLOBAL information
  // ----------------------------------------------
  if( id == 0){

    // --------------------------------------------
    // print the number of blocks
    cout << " x_blocks = " << x_blocks_ << endl;
    cout << " y_blocks = " << y_blocks_ << endl;
    // --------------------------------------------
 
    // --------------------------------------------
    // print block sizes
    cout << endl;
    cout << "x block sizes : " << endl;
    cout << "----------------" << endl;
    cout << endl;
    for(int i=0; i<x_blocks_; i++){
      cout << i << " : " << x_block_size_[i] << endl;
    }
 
    cout << endl;
    cout << "y block sizes : " << endl;
    cout << "----------------" << endl;
    cout << endl;
    for(int i=0; i<y_blocks_; i++){
      cout << i << " : " << y_block_size_[i] << endl;
    }
    // --------------------------------------------

    // --------------------------------------------
    // print block owners
    cout << endl;
    cout << "Block owners : " << endl;
    cout << "---------------" << endl;
    cout << endl;
    cout << "    ";
    for(int i=0; i<x_blocks_; i++) cout << i << " ";
    cout << endl;
    cout << "---------------------------------" << endl;

    int idx=0;
    for(int i=0; i<y_blocks_; i++){
      cout << i << " | "; 
      for(int j=0; j<x_blocks_; j++){
        cout << block_owner_[idx] << " ";
        idx++;
      }
      cout << endl;
    }
    // --------------------------------------------

  }

  // ----------------------------------------------
  // Print LOCAL information
  // ----------------------------------------------

  // --------------------------------
  // print number of local blocks
  // --------------------------------
  info = MPI_Barrier(MPI_COMM_WORLD);
  if(info != 0){
    cerr << "MPI_Barrier error in Matrix::info()" << endl;
    exit(ERROR_MPI);
  }

  int prev = ( id - 1 + num_procs ) % num_procs;
  int next = ( id + 1 ) % num_procs;
  MPI_Status status[ sizeof( MPI_Status ) ];

  if( id == 0 ){

    cout << endl;
    cout << "Number of blocks stored on processes : " << endl;
    cout << "---------------------------------------" << endl;
    cout << endl;

    cout << "# local blocks on proc " << id;
    cout << " : " << num_local_blocks_ << endl;

    int msg = 1;
    info = MPI_Send(&msg,1,MPI_INT,next,0,MPI_COMM_WORLD);
    if(info != 0){
      cerr << "MPI_Send error in Matrix::info()" << endl;
      exit(ERROR_MPI);
    }
    info = MPI_Recv(&msg,1,MPI_INT,prev,0,MPI_COMM_WORLD,status);
    if(info != 0){
      cerr << "MPI_Recv error in Matrix::info()" << endl;
      exit(ERROR_MPI);
    }
    cout << endl;

  } else {
  
    int msg = 1;
    info = MPI_Recv(&msg,1,MPI_INT,prev,0,MPI_COMM_WORLD,status);
    if(info != 0){
      cerr << "MPI_Recv error in Matrix::info()" << endl;
      exit(ERROR_MPI);
    }
    cout << "# local blocks on proc " << id;
    cout << " : " << num_local_blocks_ << endl;
    info = MPI_Send(&msg,1,MPI_INT,next,0,MPI_COMM_WORLD);
    if(info != 0){
      cerr << "MPI_Send error in Matrix::info()" << endl;
      exit(ERROR_MPI);
    }

  }
  // --------------------------------

  // --------------------------------  
  // print coordinates of local blocks 
  // --------------------------------
  info = MPI_Barrier(MPI_COMM_WORLD);
  if(info != 0){
    cerr << "MPI_Barrier error in Matrix::info()" << endl;
    exit(ERROR_MPI);
  }

  if( id == 0){

    cout << "Local blocks on proc : " << id << endl;

    for(int j=0; j<num_local_blocks_; j++){
      cout << j << " : " << block_x_coo_[j] << ", " << block_y_coo_[j] << endl; 
    }
    cout << endl;

    int msg = 1;
    info = MPI_Send(&msg,1,MPI_INT,next,0,MPI_COMM_WORLD);
    if(info != 0){
      cerr << "MPI_Send error in Matrix::print()" << endl;
      exit(ERROR_MPI);
    }
    info = MPI_Recv(&msg,1,MPI_INT,prev,0,MPI_COMM_WORLD,status);
    if(info != 0){
      cerr << "MPI_Recv error in Matrix::info()" << endl;
      exit(ERROR_MPI);
    }

  } else {
  
    int msg = 1;

    MPI_Recv(&msg,1,MPI_INT,prev,0,MPI_COMM_WORLD,status);
    if(info != 0){
      cerr << "MPI_Recv error in Matrix::print()" << endl;
      exit(ERROR_MPI);
    }
    cout << "Local blocks on proc : " << id << endl;

    for(int j=0; j<num_local_blocks_; j++){
      cout << j << " : " << block_x_coo_[j] << ", " << block_y_coo_[j] << endl; 
    }
    cout << endl;

    MPI_Send(&msg,1,MPI_INT,next,0,MPI_COMM_WORLD);
    if(info != 0){
      cerr << "MPI_Send error in Matrix::print()" << endl;
      exit(ERROR_MPI);
    }

  }
  // --------------------------------

}

// Print raw data  
void Matrix::print(){

  int info, id, num_procs;

  info = MPI_Comm_rank(MPI_COMM_WORLD, &id);
  if(info != 0){
    cerr << "MPI_Comm_rank error in Matrix::print()" << endl;
    exit(ERROR_MPI);
  }
  info = MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  if(info != 0){
    cerr << "MPI_Comm_size error in Matrix::print()" << endl;
    exit(ERROR_MPI);
  }

  int prev = ( id - 1 + num_procs ) % num_procs;
  int next = ( id + 1 ) % num_procs;
  MPI_Status status[ sizeof( MPI_Status ) ];

  info = MPI_Barrier(MPI_COMM_WORLD);
  if(info != 0){
    cerr << "MPI_Barrier error in Matrix::print()" << endl;
    exit(ERROR_MPI);
  }

  if( id == 0 ){

    cout << endl;
    cout << "Process : " << id << endl;
    cout << "# local blocks : " << num_local_blocks_ << endl;

    for(int j=0; j<num_local_blocks_; j++){

      cout << endl;
      cout << "block #"<< j << endl;
      
      int x = block_x_coo_[j];
      int y = block_y_coo_[j];

      int x_size = x_block_size_[x];
      int y_size = y_block_size_[y];

      // ------------------------------
      // print data
      // ------------------------------
      for(int p=0; p<y_size; p++){
        int offset = p*x_size;
        for(int q=0; q<x_size; q++){
          cout << data_[j][offset + q] << " ";
        }
        cout << endl;
      }

    }

    int msg = 1;
    info = MPI_Send(&msg,1,MPI_INT,next,0,MPI_COMM_WORLD);
    if(info != 0){
      cerr << "MPI_Send error in Matrix::print()" << endl;
      exit(ERROR_MPI);
    }
    info = MPI_Recv(&msg,1,MPI_INT,prev,0,MPI_COMM_WORLD,status);
    if(info != 0){
      cerr << "MPI_Recv error in Matrix::print()" << endl;
      exit(ERROR_MPI);
    }

  } else {
  
    int msg = 1;
    info = MPI_Recv(&msg,1,MPI_INT,prev,0,MPI_COMM_WORLD,status);
    if(info != 0){
      cerr << "MPI_Recv error in Matrix::print()" << endl;
      exit(ERROR_MPI);
    }

    cout << endl;
    cout << "Process : " << id << endl;
    cout << "# local blocks : " << num_local_blocks_ << endl;

    for(int j=0; j<num_local_blocks_; j++){

      cout << endl;
      cout << "block #"<< j << endl;
      
      int x = block_x_coo_[j];
      int y = block_y_coo_[j];

      int x_size = x_block_size_[x];
      int y_size = y_block_size_[y];

      // --------------------------------
      // print data
      // --------------------------------
      for(int p=0; p<y_size; p++){
        int offset = p*x_size;
        for(int q=0; q<x_size; q++){
          cout << data_[j][offset + q] << " ";
        }
        cout << endl;
      }

    }

    info = MPI_Send(&msg,1,MPI_INT,next,0,MPI_COMM_WORLD);
    if(info != 0){
      cerr << "MPI_Send error in Matrix::print()" << endl;
      exit(ERROR_MPI);
    }
  }
  
}

void Matrix::set_to(double value){
  
  // loop over local blocks 
  for(int i=0; i<num_local_blocks_; i++){

    // coordinates of blocks
    int x = block_x_coo_[i];
    int y = block_y_coo_[i];

    // sizes of blocks
    int x_size = x_block_size_[x];
    int y_size = y_block_size_[y];

    // number of elements in the block
    int nelements = x_size * y_size;

    for(int j=0; j<nelements; j++){
      data_[i][j] = value;
    }
    
  }

}

void Matrix::zero(){
  
  // loop over local blocks 
  for(int i=0; i<num_local_blocks_; i++){

    // coordinates of blocks
    int x = block_x_coo_[i];
    int y = block_y_coo_[i];

    // sizes of blocks
    int x_size = x_block_size_[x];
    int y_size = y_block_size_[y];

    // number of elements in the block
    int nelements = x_size * y_size;

    memset(data_[i],0,nelements*sizeof(double));
  }

}

void Matrix::scale(double value){
  
  // loop over local blocks 
  for(int i=0; i<num_local_blocks_; i++){

    // coordinates of blocks
    int x = block_x_coo_[i];
    int y = block_y_coo_[i];

    // sizes of blocks
    int x_size = x_block_size_[x];
    int y_size = y_block_size_[y];

    // number of elements in the block
    int nelements = x_size * y_size;

    for(int j=0; j<nelements; j++){
      data_[i][j] *= value;
    }

  }

}

double Matrix::norm2(){

  int info;
  double local_norm = 0.0;
  double global_norm = 0.0;
  double norm2 = 0.0;

  // loop over local blocks
  for(int i=0; i<num_local_blocks_; i++){
    
    // coordinates of blocks
    int x = block_x_coo_[i];
    int y = block_y_coo_[i];

    // sizes of blocks
    int x_size = x_block_size_[x];
    int y_size = y_block_size_[y];

    // number of elements in the block
    int nelements = x_size * y_size;

    // FIXME: Do it with BLAS NORM2

    for(int j=0; j<nelements; j++){
      local_norm += data_[i][j];
    }

  }

  info = MPI_Allreduce(&local_norm,&global_norm,1, \
                  MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  if(info != 0){
    cerr << "MPI_Allreduce error in Matrix::norm2()" << endl;
    exit(ERROR_MPI);
  }

  norm2 = sqrt(global_norm);

  return norm2;

}

void Matrix::add_to_this(double alpha, Matrix *other){

  if( this->check_block_structure(other) != 0){
    cerr << "Terror Error :: Block structure of matrices does not match" << endl;
    cerr << "Error in Matrix::add_to_this" << endl;
    exit(ERROR_ARRAY_MISSMATCH);
  }

  double **A = this->data_;
  double **B = other->data_;

  // loop over local blocks 
  for(int i=0; i<num_local_blocks_; i++){
    
    // coordinates of blocks
    int x = block_x_coo_[i];
    int y = block_y_coo_[i];

    // sizes of blocks
    int x_size = x_block_size_[x];
    int y_size = y_block_size_[y];

    // number of elements in the block
    int nelements = x_size * y_size;

    // FIXME: Do it with BLAS DAXPY

    for(int j=0; j<nelements; j++){
      A[i][j] += alpha * B[i][j];
    }

  }

  A = NULL;
  B = NULL;

}

Matrix::Matrix(int *dims, int *nblocks, \
                int *xblocks_dims, int *yblocks_dims){

  // assign global dimensions 
  dims_[0] = dims[0];
  dims_[1] = dims[1];

  // assign number of blocks
  x_blocks_ = nblocks[0];
  y_blocks_ = nblocks[1];

  // total number of blocks
  num_total_blocks_ = x_blocks_ * y_blocks_; 

  // assign sizes of blocks
  x_block_size_ = new int [x_blocks_];
  memcpy(x_block_size_,xblocks_dims,x_blocks_*sizeof(int));

  y_block_size_ = new int [y_blocks_];
  memcpy(y_block_size_,yblocks_dims,y_blocks_*sizeof(int));

  // --------------------------------------------
  // assign processors to blocks
  // --------------------------------------------
  int num_procs, info;

  info = MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  if(info != 0){
    cerr << "MPI_Comm_size error in Matrix::Matrix(int *)" << endl;
    exit(ERROR_MPI);
  }

  block_owner_ = new int [num_total_blocks_];
  int block = 0;
  while( block < num_total_blocks_ ){
    for(int j=0; j<num_procs; j++){
            
      if( block < num_total_blocks_ ){
        block_owner_[block] = j;
        block++;
      } else {
        break;
      }

    }
  }
  // --------------------------------------------

  // --------------------------------------------
  // count number of local blocks on this processor
  // --------------------------------------------
  int my_id;
  info = MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
  if(info != 0){
    cerr << "MPI_Comm_rank error in Matrix::Matrix(int *)" << endl;
    exit(ERROR_MPI);
  }

  num_local_blocks_ = 0;
  for(int i=0; i<num_total_blocks_; i++){
    if( block_owner_[i] == my_id ) num_local_blocks_++;
  }

  // allocate space for coordinates of local blocks
  // ---------------------------------------------
  block_x_coo_ = new int [num_local_blocks_];
  block_y_coo_ = new int [num_local_blocks_];

  int total_idx = 0;
  int local_idx = 0;

  for(int i=0; i<y_blocks_; i++){
    for(int j=0; j<x_blocks_; j++){

      if( block_owner_[total_idx] == my_id ) {   

        block_x_coo_[local_idx] = j;
        block_y_coo_[local_idx] = i;
        local_idx++;

      }      

      total_idx++;
    }

  }
  // --------------------------------------------

  // --------------------------------------------
  // allocate the local blocks
  // --------------------------------------------
  data_ = new double *[num_local_blocks_];

  for(int i=0; i<num_local_blocks_; i++){

    int x = block_x_coo_[i];
    int y = block_y_coo_[i];

    int nelements = x_block_size_[x] * y_block_size_[y];

    data_[i] = new double [nelements];

    memset(data_[i],0,nelements*sizeof(double));

  }
  // --------------------------------------------

}

bool Matrix::check_block_structure(Matrix *other){

  // check the number of blocks 
  if(this->x_blocks_ != other->x_blocks_){
    return 1;
  }
  
  if(this->y_blocks_ != other->y_blocks_){
    return 1;
  }

  int block_missmatch = 0;

  // check sizes of blocks in x direction
  for(int i=0; i<x_blocks_; i++){

    int x1 = this->x_block_size_[i];
    int x2 = other->x_block_size_[i];

    block_missmatch = (x1 - x2);

    if(block_missmatch != 0) return 1;

  }

  // check sizes of blocks in y direction
  for(int i=0; i<y_blocks_; i++){

    int y1 = this->y_block_size_[i];
    int y2 = other->y_block_size_[i];

    block_missmatch = (y1 - y2);

    if(block_missmatch != 0) return 1;

  }

  return 0;
}

double Matrix::dot_product(Matrix *other){

  if( this->check_block_structure(other) != 0){
    cerr << "Terror Error :: Block structure of matrices does not match" << endl;
    cerr << "Error in Matrix::dot_product" << endl;
    exit(ERROR_ARRAY_MISSMATCH);
  }

  double local_dot_product = 0.0;
  double global_dot_product = 0.0;

  double **A = this->data_;
  double **B = other->data_;

  // loop over local blocks
  for(int i=0; i<num_local_blocks_; i++){
    
    // coordinates of blocks
    int x = block_x_coo_[i];
    int y = block_y_coo_[i];

    // sizes of blocks
    int x_size = x_block_size_[x];
    int y_size = y_block_size_[y];

    // number of elements
    int nelements = x_size * y_size;

    // FIXME: switch to BLAS DOT

    // product
    for(int j=0; j<nelements; j++){
      local_dot_product += A[i][j] * B[i][j];
    }

  }

  int info = MPI_Allreduce(&local_dot_product,&global_dot_product,1, \
                  MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  if(info != 0){
    cerr << "MPI_Allreduce error in Matrix::norm2()" << endl;
    exit(ERROR_MPI);
  }

  A = NULL;
  B = NULL;

  return global_dot_product;
}

double Matrix::max(){

  double local_max = -1e-15;
  double global_max = 0.0;

  // loop over local blocks
  for(int i=0; i<num_local_blocks_; i++){
    
    // coordinates of blocks
    int x = block_x_coo_[i];
    int y = block_y_coo_[i];

    // sizes of blocks
    int x_size = x_block_size_[x];
    int y_size = y_block_size_[y];

    // number of elements 
    int nelements = x_size * y_size;

    for(int j=0; j<nelements; j++){
      if( data_[i][j] > local_max) local_max = data_[i][j];
    }

  }

  int info = MPI_Allreduce(&local_max,&global_max,1, \
                  MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
  if(info != 0){
    cerr << "MPI_Allreduce error in Matrix::max()" << endl;
    exit(ERROR_MPI);
  }

  return global_max;

}

double Matrix::min(){

  double local_min = 1e15;
  double global_min = 0.0;

  // loop over local blocks
  for(int i=0; i<num_local_blocks_; i++){
    
    // coordinates of blocks
    int x = block_x_coo_[i];
    int y = block_y_coo_[i];

    // sizes of blocks
    int x_size = x_block_size_[x];
    int y_size = y_block_size_[y];

    // number of elements 
    int nelements = x_size * y_size;

    for(int j=0; j<nelements; j++){
      if( data_[i][j] < local_min) local_min = data_[i][j];
    }

  }

  int info = MPI_Allreduce(&local_min,&global_min,1, \
                  MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);
  if(info != 0){
    cerr << "MPI_Allreduce error in Matrix::min()" << endl;
    exit(ERROR_MPI);
  }

  return global_min;

}

void Matrix::unit(){

  // check if the matrix is symmetric with respect to blocks in x and y 
  if( x_blocks_ == y_blocks_ ) {

    // check block sizes
    for(int i=0; i<x_blocks_; i++){
      if( x_block_size_[i] != y_block_size_[i]){

        cerr << "Matrix blocking not symmetric, size of blocks " \
                << i << " do not match in x and y" << endl;
        cerr << "Matrix::unit() is not implemented for this type yet." << endl;
        exit(ERROR_NOT_CODED);
        
      }
    }

    // loop over local blocks
    for(int i=0; i<num_local_blocks_; i++){
      
      // block coordinates
      int x = block_x_coo_[i];
      int y = block_y_coo_[i];

      // block sizes
      int x_size = x_block_size_[x];
      int y_size = y_block_size_[y];

      // number of elements
      int nelements = x_size * y_size;

      if( x == y ){
        // diagonal block
        memset(data_[i],0,nelements*sizeof(double));
        
        int idx;
        for(int p=0; p<x_size; p++){
          idx = p*x_size+p;
          data_[i][idx] = 1.0;
        }

      } else {
        // off-diagonal block
        memset(data_[i],0,nelements*sizeof(double));
      }

    }

  } else {
    cerr << "Matrix is not symmetric with respect to blocking in x and y" << endl;
    cerr << "Matrix::unit() is not implemented for this type yet." << endl;
    exit(ERROR_NOT_CODED);
  }

}

