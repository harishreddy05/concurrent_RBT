#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define M 10

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
    struct par_rbNode *left, *right, *parent;
};

// /*Function prototypes */
__device__ void createNIL();
__device__ struct par_rbNode * createNode(int);
__device__ void createTree();
__device__ struct par_rbNode * Traverse(int);
__device__ enum result PlaceNode(struct par_rbNode *, struct par_rbNode *);
__device__ void Insert_Rebalance(struct par_rbNode *);
__device__ bool Update_Rotation(struct par_rbNode *, enum caseFlag *);
__device__ void Left_Rotate(struct par_rbNode *);
__device__ void Right_Rotate(struct par_rbNode *);

__device__ struct par_rbNode *nodes;
__device__ struct par_rbNode *root;
__device__ struct par_rbNode *NIL;
__device__ struct par_rbNode *rtParent;
__device__ struct par_rbNode *rtSibling; // U might feel this is unncessary, but it will be used
__device__ int nodeIndex = 0;

__device__ void createNIL(){
    NIL = &nodes[0];
    NIL->color = BLACK;
    NIL->key = -1;
    NIL->left = NIL->right = NIL->parent = NIL;
    printf("NIL created\n");
}

__device__ struct par_rbNode * createNode(int key){

    atomicAdd(&nodeIndex,1);
    nodes[nodeIndex].key = key;
    nodes[nodeIndex].left = nodes[nodeIndex].right = nodes[nodeIndex].parent = NIL;
    return &nodes[nodeIndex]; // Even if this thread pauses it will eventually return the correct pointer

}

__device__ void createTree(){
    rtParent = createNode(-1);
    rtSibling = createNode(-1);
    root = NIL;
    rtParent->parent = NIL;
    rtSibling->parent = rtParent;
    rtSibling->right = NIL;
    rtSibling->left = NIL;
    rtParent->left = root;
    rtParent->right = rtSibling;
    rtParent->color = BLACK;
    rtSibling->color = BLACK;
    NIL->parent = rtParent;
    printf("Tree Created \n");
    printf("\n");
}

__device__ struct par_rbNode * Traverse(int d){
    struct par_rbNode *x;
    x = root;
    if(x == NIL){
        printf("Empty Tree\n");
        return NIL; 
    }
    while(x != NIL){
        if(x->key == d){
            printf("Found it!\n");
            return x;
        }else if(x->key > d){
            x = x->left;
        }else{
            x = x->right;
        }
    }
    printf("Couldn't find %d in this tree\n",d);
    return NIL; 
}

__device__ void Left_Rotate(struct par_rbNode *lptr){
    struct par_rbNode *y;
    y = lptr->right;
    lptr->right = y->left;
    if(y->left != NIL)
        y->left->parent = lptr;
    if(y!=NIL)
        y->parent = lptr->parent;
    if(lptr->parent == NIL){
        root = y;
    }else if(lptr == lptr->parent->left)
        lptr->parent->left = y;
    else
        lptr->parent->right = y;
    y->left = lptr;
    if(lptr != NIL)
        lptr->parent = y;
}

__device__ void Right_Rotate(struct par_rbNode *rptr){
    struct par_rbNode *y;
    y = rptr->left;
    rptr->left = y->right;
    if(y->right != NIL)
        y->right->parent = rptr;
    if(y!=NIL)
        y->parent = rptr->parent;
    if(rptr->parent == NIL){
        root = y;
    }else if(rptr == rptr->parent->right)
        rptr->parent->right = y;
    else
        rptr->parent->left = y;
    y->right = rptr;
    if(rptr != NIL)
        rptr->parent = y;
}

__device__ void Insert_fixup(struct par_rbNode *x){
    struct par_rbNode *u;
    while(x->parent->color == RED){
        if(x->parent == x->parent->parent->left){
            u = x->parent->parent->right;
            if(u->color == RED){//CASE 1
                x->parent->color = BLACK;
                u->color = BLACK;
                x->parent->parent->color = RED;
                x = x->parent->parent;
            }else if(x == x->parent->right){//CASE 2
                x = x->parent;
                Left_Rotate(x);
                x->parent->color = BLACK;
                x->parent->parent->color = RED;
                Right_Rotate(x->parent->parent);
            }else if(x == x->parent->left){
                x->parent->color = BLACK;
                x->parent->parent->color = RED;
                Right_Rotate(x->parent->parent);
            }   //CASE 3
        }else{
            u = x->parent->parent->left;
            if(u->color == RED){//CASE 1
                x->parent->color = BLACK;
                u->color = BLACK;
                x->parent->parent->color = RED;
                x = x->parent->parent;
            }else if(x == x->parent->left){//CASE 2
                x = x->parent;
                Right_Rotate(x);
                x->parent->color = BLACK;
                x->parent->parent->color = RED;
                Left_Rotate(x->parent->parent);
            }else if(x == x->parent->right){
                x->parent->color = BLACK;
                x->parent->parent->color = RED;
                Left_Rotate(x->parent->parent);
            }   //CASE 3
        }
    }
    root->color = BLACK;
}

__device__ void Insert(int d){
    if(root == NIL){
        root = createNode(d);
        root->color = BLACK;
        return;
    }
    struct par_rbNode *x,*z;
    x = root;
    while(x != NIL){
        z = x;
        if(d == x->key){ // Find if the node with this value is already there or not
            printf("Duplicate Nodes are not allowed\n");
            return;
        }
        if(d < x->key)
            x = x->left;
        else
            x = x->right;
    }//end while
    x=createNode(d);
    x->parent = z;
    if(x->key < z->key) //Check if y is the left child of z or not
        z->left = x;
    else
        z->right = x;
    //NEW NODE IS INSERTED, NOW FIX THE RB TREE
    Insert_fixup(x);
    // printInorder(root);
}

//Functions for printing the tree
__device__ void printPreorder(struct par_rbNode* node)
{
    if (node == NIL)
        return;
    /* first print the data of node */
    printf("%d-", node->key);
    printf("%d", node->color);
    printf("  ");  
    /* then recur on left child */
    printPreorder(node->left);
    /* now recur on right child */
    printPreorder(node->right);
}

__device__ void printInorder(struct par_rbNode* node)
{
    if (node == NIL)
        return;
    /* first recur on left child */
    printInorder(node->left);
    /* then print the data of node */
    printf("%d-", node->key);
    printf("%d", node->color);
    printf("  ");  
    /* now recur on right child */
    printInorder(node->right);
}

__device__ void printPostorder(struct par_rbNode* node)
{
    if (node == NIL)
        return;
    /* first recur on left child */
    printPostorder(node->left);
    /* then recur on right child */
    printPostorder(node->right);
    /* now print the data of node */
    printf("%d-", node->key);
    printf("%d", node->color);
    printf("  ");  
}

__global__ void RBT(struct par_rbNode *d_nodes) {

    printf("Starting the Tree\n");
    nodes = d_nodes; // Make it a global variable
    createNIL();
    createTree();
    for(int i=0;i<7;i++){
        Insert(i);
    }
    printf("PreOrder: ");
    printPreorder(root);
    printf("\n");
    printf("\n");
    printf("InOrder: ");
    printInorder(root);
    printf("\n");
    printf("\n");
    printf("PostOrder: ");
    printPostorder(root);
    printf("\n");
    printf("\n");
    printf("Done\n");
//return to main
}

int main() {
    struct par_rbNode h_nodes[M];
    struct par_rbNode *d_nodes;
    float time;
    // 1. Allocate device array.
    cudaMalloc(&d_nodes, M * sizeof(struct par_rbNode));
    for(int i=0;i<M;i++){
            h_nodes[i].color = RED;
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 2. Copy array contents from host to device.
    cudaMemcpy(d_nodes, h_nodes, M * sizeof(struct par_rbNode), cudaMemcpyHostToDevice);
    printf("Kernel Launched\n");
    cudaEventRecord(start, 0);
    RBT<<<1,1>>>(d_nodes);
    cudaMemcpy(h_nodes, d_nodes, M * sizeof(struct par_rbNode), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    printf("Came back\n");
    cudaEventElapsedTime(&time, start, stop);
    printf ("Time for the kernel: %f ms\n", time);
    return 0;
}