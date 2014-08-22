#ifndef MATRIX_H
#define MATRIX_H

class Matrix {

  public: 

    Matrix(int *); // constructor
    Matrix(int *, int *, int *, int *); // constructor with predefined blocks

    ~Matrix(); // destructor

    void info(); // info about the matrix
    void print(); // print matrix

    void set_to(double); // set all elements to a given value
    void zero(); // set all elements to zero
    void scale(double); // scale all elements by a given factor
    void unit(); // unit matrix 

    double max(); // value of the largest element
    double min(); // value of the smallest element

    double norm2(); // calculate 2-norm of a matrix
    double dot_product(Matrix *); // dot product of two matrices 

    void add_to_this(double, Matrix *); // add scaled matrix to this one

    // check if block structure and their distribution is 
    // the same for both matrices
    bool check_block_structure(Matrix *); 

  private: 

    // Global matrix ---------------------------------------- 
    int dims_[2]; // dimensions (num of columns and rows)

    int x_blocks_; // number of column blocks
    int y_blocks_; // number of row blocks

    int *x_block_size_; // sizes of blocks in columns
    int *y_block_size_; // sizes of blocks in rows

    int *block_owner_; // processor id for the owner of this block

    int num_total_blocks_; // total number of blocks

    // Local matrix -----------------------------------------
    int num_local_blocks_; // number of local blocks
    
    int *block_x_coo_; // x coordinate of local blocks
    int *block_y_coo_; // y coordinate of local blocks

    double **data_; // data 
};

#endif
