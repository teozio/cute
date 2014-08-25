#ifndef TENSOR_H
#define TENSOR_H

class Tensor {

  public:

  private:

    int rank_; // number of dimensions (indices)

    int *dims_; // dimensions of each rank

    int *nBlocks_; // number of blocks for each rank

    int **BlockSize_; // sizes of blocks

    int *BlockOwner_; // id process assigned to each block

    int nTotalBlocks_; // number of total blocks
    int nLocalBlocks_; // number of local blocks

    int **BlockCoo_; // coordinates of each local block

    double **data_; // data    

};

#endif
