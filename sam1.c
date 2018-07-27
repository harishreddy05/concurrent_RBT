#include <cuda.h>
#include <stdio.h>

#define M 20;

// RED = 0, BLACK = 1
enum nodeColor {
    RED,
    BLACK
};

enum result {
	Success,
	Failure,
	FirstInsert
};

enum caseFlag { 
	NOOP,
	DID_CASE1,
	DID_CASE3
};


// __device__ void CreateTree();

// struct par_rbNode newNode(){
// 	atomicAdd(&nodeIndex,1);
// 	return nodes[nodeIndex];
// }

// __device__ struct par_rbNode *nodes; // device (global)
// __device__ int nodeIndex = 0; // device (global)
struct par_rbNode {
    int key, color;
    bool flag;
    struct par_rbNode *left, *right, *parent;
};

// __device__ struct par_rbNode *root;
// __device__ struct par_rbNode *NIL;
// __device__ struct par_rbNode *rtParent;
// __device__ struct par_rbNode *rtSibling; // U might feel this is unncessary, but it will be used


// __device__ void CreateTree(){
//     // rtParent = ; //PENDING
//     rtSibling = (struct par_rbNode *)malloc(sizeof(struct par_rbNode));
//     NIL = (struct par_rbNode *)malloc(sizeof(struct par_rbNode));
//     root = NIL;
//     rtParent->parent = NIL;
//     rtSibling->parent = rtParent;
//     rtSibling->right = NIL;
//     rtSibling->left	= NIL;
//     rtParent->left = root; 
//     //rtParent->left = root; Why only left, y not right?
//     //ANS: Since we check for left parent condition first 
//     //(if u don't understand, try to insert a node to a tree with only one node)
//     rtParent->right = rtSibling;
//     rtParent->flag = false;
//     rtSibling->flag = false;
//     rtParent->color = BLACK;
//     rtSibling->color = BLACK;
//     NIL->left = NIL;
//     NIL->right = NIL;
//     NIL->parent = rtParent;
//     NIL->flag = false;
//     NIL->color = BLACK;
// }


__global__ void RBT(struct par_rbNode *nodes) {
	printf("%d\n",nodes);
}



int main() {

	struct par_rbNode *a[M];
	struct par_rbNode *a_device;
    const size_t a_size = sizeof(struct par_rbNode) * size_t(M);
    cudaMalloc((void **)&a_device, a_size); 
    cudaMemcpy(a_device, a, a_size, cudaMemcpyHostToDevice); 
    RBT<<<1,1>>>(a_device); 
	cudaDeviceSynchronize();
	return 0;
}