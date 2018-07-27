#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define M 20

// RED = 0, BLACK = 1
enum nodeColor {
    RED,
    BLACK
};

enum result {
    Failure,
    Success,
    FirstInsert
};

enum caseFlag {
    NOOP,
    DID_CASE1,
    DID_CASE3
};

struct par_rbNode {
    int key, color;
    int flag;
    struct par_rbNode *left, *right, *parent;
};

// /*Function prototypes */
__device__ void createNIL();
__device__ struct par_rbNode * createNode(int);
__device__ void createTree();

__device__ struct par_rbNode *nodes;
__device__ struct par_rbNode *root;
__device__ struct par_rbNode *NIL;
__device__ struct par_rbNode *rtParent;
__device__ struct par_rbNode *rtSibling; // U might feel this is unncessary, but it will be used
__device__ int nodeIndex = 0;
__device__ int tmpIndex = 0;
__device__ struct par_rbNode *tmp[M];// need M tmps

__device__ int createFlag = false;

__device__ void createNIL(){
    NIL = &nodes[0];
    NIL->color = BLACK;
    NIL->key = -1;
    NIL->left = NIL->right = NIL->parent = NIL;
    printf("NIL created\n");
}

__device__ struct par_rbNode * createNode(int key){

    bool ok = false;
    do{
        ok = atomicCAS(&createFlag,false,true); //Capture the lock
    }while(ok);
    atomicAdd(&nodeIndex,1);
    atomicAdd(&tmpIndex,1);
    nodes[nodeIndex].key = key;
    nodes[nodeIndex].flag = true;
    nodes[nodeIndex].left = nodes[nodeIndex].right = nodes[nodeIndex].parent = NIL;
    tmp[tmpIndex] = &nodes[nodeIndex];
    // createFlag = false;
    atomicCAS(&createFlag,true,false); //Release the lock
    return tmp[tmpIndex]; // Even if this thread pauses it will eventually return the correct pointer
}

__device__ void createTree(){
    rtParent = createNode(-1);
    rtSibling = createNode(-1);
    // NIL = createNode(-1);
    root = NIL;
    rtParent->parent = NIL;
    rtSibling->parent = rtParent;
    rtSibling->right = NIL;
    rtSibling->left    = NIL;
    rtParent->left = root;
    //rtParent->left = root; Why only left, y not right?
    //ANS: Since we check for left parent condition first 
    //(if u don't understand, try to insert a node to a tree with only one node)
    rtParent->right = rtSibling;
    rtParent->flag = false;
    rtSibling->flag = false;
    rtParent->color = BLACK;
    rtSibling->color = BLACK;
    // NIL->left = NIL;
    // NIL->right = NIL;
    NIL->parent = rtParent;
    NIL->flag = false;
    // NIL->color = BLACK;
    printf("Tree Created \n");
    printf("\n");
}

__device__ void Insert(int key){
    struct par_rbNode *newNode = createNode(key); //Internally the flag of the newNode is held
}


__device__ int threadsFinished = 0;
__device__ int passCreate = 0;

__global__ void RBT(struct par_rbNode *d_nodes) {

    int id = blockIdx.x*blockDim.x+threadIdx.x;
    int threadCount = gridDim.x*blockDim.x;
    if(id == 0){
        nodes = d_nodes; // Make it a global variable
        createNIL();
        createTree();
        atomicAdd(&passCreate,1);
        printf("%d\n",passCreate);
    }

    // printf("hello\n");
    // Insert(5);
    // Insert(6);
    // Insert(4);

    while(1){
        if(passCreate){
            printf("Thread %d\n",id);
            Insert(id);
            break;
        }
    }
    // //Print the time
    // //This will keep track of number of threads that are done
    atomicAdd(&threadsFinished,1);
    // // //Print the tree after all the threads are done
    // if(threadsFinished == threadCount){
    //     if(id == 0){
    //         // printf("PreOrder: ");
    //         // printPreorder(root);
    //         // printf("\n");
    //         // printf("\n");
    //         // printf("InOrder: ");
    //         // printInorder(root);
    //         // printf("\n");
    //         // printf("\n");
    //         // printf("PostOrder: ");
    //         // printPostorder(root);
    //         // printf("\n");
    //         // printf("\n");
    //     }
    // }
//return to main
}

int main() {
    struct par_rbNode h_nodes[M];
    struct par_rbNode *d_nodes;
    float time;
    // 1. Allocate device array.
    cudaMalloc(&d_nodes, M * sizeof(struct par_rbNode));
    for(int i=0;i<M;i++){
            h_nodes[i].flag = false;
            h_nodes[i].color = RED;
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 2. Copy array contents from host to device.
    cudaMemcpy(d_nodes, h_nodes, M * sizeof(struct par_rbNode), cudaMemcpyHostToDevice);
    printf("Kernel Launched\n");
    cudaEventRecord(start, 0);
    RBT<<<1,2>>>(d_nodes);
    cudaMemcpy(h_nodes, d_nodes, M * sizeof(struct par_rbNode), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    printf("Came back\n");
    cudaEventElapsedTime(&time, start, stop);
    printf ("Time for the kernel: %f ms\n", time);
    return 0;
}