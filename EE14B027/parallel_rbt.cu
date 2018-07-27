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
    Success,
    Failure,
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
__device__ struct par_rbNode * Traverse(int);
__device__ enum result PlaceNode(struct par_rbNode *, struct par_rbNode *);
__device__ void Insert_Rebalance(struct par_rbNode *);
__device__ bool Update_Rotation(struct par_rbNode *, enum caseFlag *);
__device__ bool Left_Rotate(struct par_rbNode *);
__device__ bool Right_Rotate(struct par_rbNode *);

__device__ struct par_rbNode *nodes;
__device__ struct par_rbNode *root;
__device__ struct par_rbNode *NIL;
__device__ struct par_rbNode *rtParent;
__device__ struct par_rbNode *rtSibling; // U might feel this is unncessary, but it will be used
__device__ int nodeIndex = 0;
__device__ int tmpIndex = 2;
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

    bool ok;
    do{
        ok = atomicCAS(&createFlag,false,true); //Capture the lock
    }while(!ok);
    atomicAdd(&nodeIndex,1);
    atomicAdd(&tmpIndex,1);
    nodes[nodeIndex].key = key;
    nodes[nodeIndex].flag = true;
    nodes[nodeIndex].left = nodes[nodeIndex].right = nodes[nodeIndex].parent = NIL;
    tmp[tmpIndex] = &nodes[nodeIndex];
    createFlag = false;
    // atomicCAS(&createFlag,true,false); //Release the lock
    printf("Created %d\n",key);
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
}

__device__ struct par_rbNode * Traverse(int key){
    struct par_rbNode *x;
    struct par_rbNode *insertPoint = NIL;
    // struct par_rbNode *savert;
    bool success;

    // do{
    //  savert = root;
    //  success = DCAS(&root->flag,false,true,&root,savert,savert); //Catching the flag of the root
    // }while(!success);

    //An alternate for DCAS - should check if it works or not
    // do{
    //     savert = root;
    //     success = atomicCAS(&root->flag,false,true); //Catching the flag of the root
    // }while(savert!=root || !success);


    do{
        // savert = root;
        success = atomicCAS(&root->flag,false,true); //Catching the flag of the root
    }while(!success);
    //success => captured the root flag
    //savert != root => root has changed
    //!success => root is under lock
    //thread will come out of the loop only after "success" and "savert==root" 
    x = root;
    while(x != NIL){
        if(key == x->key) {
            x->flag = false; // Release the flag that was just caught
            return NULL; // Traversing is done. Node is already there so Insert() fails.
        }
        insertPoint = x;
        if(key < x->key){
            if(x->left != NIL){
                success = atomicCAS(&x->left->flag,false,true);
                if(!success){
                    x->flag = false; // Release the flag of x
                    return NULL;
                }//end if
                x = x->left;
                insertPoint->flag = false;  
            }else{
                x = x->left;
            }//end if
        }else{
            if(x->right != NIL){
                success = atomicCAS(&x->right->flag,false,true);
                if(!success){
                    x->flag = false;
                    return NULL;
                }//end if
                x = x->right;
                insertPoint->flag = false;
            }else{
                x = x->right;
            }//end if
        }//end if
    }//end while
    return insertPoint;
}

__device__ enum result PlaceNode(struct par_rbNode *newNode, struct par_rbNode *insertPoint){
    //flags on newNode and insertPoint are held
    bool ok;
    // struct par_rbNode *uncle,*savep;

    if(insertPoint == NIL){ //tree is empty
        newNode->color = BLACK;
        newNode->parent = rtParent;
        rtParent->left = newNode;
        root=newNode;
        NIL->flag = false; // release NIL node, that u caught during Traverse
        newNode->flag = false;
        return FirstInsert;
    }else{ // the tree is not empty so...
        newNode->parent = insertPoint;
        //set the flags of the grandparent and uncle
        if(insertPoint == insertPoint->parent->left){ //uncle is right child
            // savep = insertPoint->parent; // save parent ptr
            // uncle = savep->right;   // rtSibling is used here, when insertPoint is root
            ok = atomicCAS(&insertPoint->parent->flag,false,true);
            if(ok){
                ok = atomicCAS(&insertPoint->parent->right->flag,false,true);
                // if(ok){
                //     ok = atomicCAS(&insertPoint->parent,savep,savep) && atomicCAS(&savep->right,uncle,uncle);
                // }
                if(!ok){ //back off
                    insertPoint->parent->flag = false;
                    insertPoint->parent->right->flag = false;
                }else{
                    insertPoint->parent->flag = false;
                }//end if
            }
        }else{// uncle is left child
            // savep = insertPoint->parent; // save parent ptr
            // uncle = savep->left;
            ok = atomicCAS(&insertPoint->parent->flag,false,true);
            if(ok){
                ok = atomicCAS(&insertPoint->parent->left->flag,false,true);
                // if(ok){
                //     ok = atomicCAS(&insertPoint->parent,savep,savep) && atomicCAS(&savep->left,uncle,uncle);
                // }
                if(!ok){ //back off
                    insertPoint->parent->flag = false;
                    insertPoint->parent->left->flag = false;
                }else{
                    insertPoint->parent->flag = false;
                }//end if
            }
        }//end if
        if(!ok){
         // This "!ok" is when u fail to capture the grandparent flag,
         // u haven't caught any extra flags so just get rid of the flag of insertPoint
            newNode->parent = NIL;
            insertPoint->flag = false; // release flag
            return Failure;         //avoid deadlock 
        }
        // When u have successfully captured all the required flags.
        // i.e. parent, grandparent, uncle
        if(newNode->key < insertPoint->key){
            //insert as left child
            insertPoint->left = newNode;
            return Success;
        }else{//insertas right child
            insertPoint->right = newNode;
            return Success;
        }
    }
}

__device__ void Insert_Rebalance(struct par_rbNode *x){ //THIS FUNCTION DOESN'T BACKOFF. IT KEEPS TRYING
    //we hold flags on x, p(x), p(p(x)), and uncle(x)
    struct par_rbNode *oldx;
    struct par_rbNode *uncle, *olduncle;
    // struct par_rbNode *savep, *savegp;
    struct par_rbNode *brother;
    struct par_rbNode *nephew;
    bool ok;
    bool updateSucceeds; //Update-Rotation successded?

    //caseF is short for caseFlag (avoiding confusion between global enum and local variable)
    enum caseFlag caseF = NOOP; // initially not doing any case
    //define uncle for first iteration
    if(x->parent == x->parent->parent->left){
        uncle = x->parent->parent->right;
    }else{ // uncle is the left child not right
        uncle = x->parent->parent->left;
    }
    while((x != root) && (x->parent->color == RED)){
        //do color-update and/or rotaion as required
        do{
            updateSucceeds = Update_Rotation(x,&caseF);
        }while(!updateSucceeds);

        //CASE 1: move to grandparent after color update
        if(caseF == DID_CASE1){
            oldx = x;   //save pointer to the old x
            olduncle = uncle; // save pointer to old uncle;
            x = x->parent->parent; // up to grandparent
            do{ //find new uncle of x and get flags
                if(x->parent == x->parent->parent->left){
                    // savep = x->parent;
                    // savegp = savep->parent;
                    // uncle = savegp->right;
                    ok = atomicCAS(&x->parent->flag,false,true);
                    if(ok){
                        ok = atomicCAS(&x->parent->parent->flag,false,true);
                        if(ok){
                            ok = atomicCAS(&x->parent->parent->right->flag,false,true);
                            if(!ok){
                                x->parent->flag = false;
                                x->parent->parent->flag = false;
                                x->parent->parent->right->flag = false;
                            }else{
                                x->parent->flag = false;
                                x->parent->parent->flag = false;
                            }
                        }else{
                            x->parent->flag = false;
                        }
                    }
                }else{
                    // savep = x->parent;
                    // savegp = savep->parent;
                    // uncle = savegp->left;
                    ok = atomicCAS(&x->parent->flag,false,true);
                    if(ok){
                        ok = atomicCAS(&x->parent->parent->flag,false,true);
                        if(ok){
                            ok = atomicCAS(&x->parent->parent->left->flag,false,true);
                            if(!ok){
                                x->parent->flag = false;
                                x->parent->parent->flag = false;
                                x->parent->parent->left->flag = false;
                            }else{
                                x->parent->flag = false;
                                x->parent->parent->flag = false;
                            }
                        }else{
                            x->parent->flag = false;
                        }
                    }
                }
            }while(!ok); //THIS FUNCTION DOESN'T BACKOFF. IT KEEPS TRYING
            //Release old flags for CASE 1
            oldx->parent->flag = false;
            olduncle->flag = false;
            oldx->flag = false;
        }
    //in CASE 3 loop will exit: parent will be BLACK
    }
    switch(caseF){
        case NOOP: //In the beginning of this function we had 
                    //x,p(x),p(p(x)),uncle(x) - release them
                    x->parent->parent->flag = false;
                    x->parent->flag = false;
                    uncle->flag = false;
                    x->flag = false;
                    break;
        case DID_CASE1: //Release the last set of flags acquired
                    x->parent->parent->flag = false;
                    x->parent->flag = false;
                    uncle->flag = false;
                    x->flag = false;
                    break;
        case DID_CASE3: //release flags on ROTATED x, etc
                    if(x == x->parent->left){
                        brother = x->parent->right;
                        nephew = x->parent->right->right;
                    }else{
                        brother = x->parent->left;
                        nephew = x->parent->left->left;
                    }
                    x->parent->flag = false;
                    brother->flag = false;
                    nephew->flag = false;
                    x->flag = false;
                    break;
    }
    root->color = BLACK; 
}

__device__ bool Update_Rotation(struct par_rbNode *x, enum caseFlag *caseF){
    //we hold flags on x, p(x), p(p(x)) and uncle(x)
    struct  par_rbNode *xUncle;
    struct  par_rbNode *oldx; //*ggp; // ggp -> greatgrandparent
    bool ok;

    if(x->parent == x->parent->parent->left){
        //the parent is a left child
        xUncle = x->parent->parent->right;
        if(xUncle->color == RED){
            //CASE 1 - recoloring
            // U have all the flags u need. So this is simple, similar to serial code
            x->parent->color = BLACK;
            xUncle->color = BLACK;
            x->parent->parent->color = RED;
            *caseF = DID_CASE1;
            return true; // This true is for "updateSucceeds"
        }else{ // rotation(s) will be needed
            if(x == x->parent->right){//CASE2
                oldx = x; // save old x in case rotate fails
                x = x->parent;
                ok = Left_Rotate(x);
                if(!ok){
                    x = oldx; //undo change to x
                    return false; //This false is for "updateSucceeds"
                }
            }
            //In CASE 3, if the right-rotation fails,
            //CASE 3 fails but the algorithm still works
            //beacuse the process will return false to 
            //Insert_Rebalance, and Insert_Rebalance will
            //call Update_Rotation again to complete CASE3
            do{ // get great grandparent's flag
                // ggp = x->parent->parent->parent;
                ok = atomicCAS(&x->parent->parent->parent->flag,false,true);
            }while(!ok);    //KEEPS TRYING, DOESN'T BACK OFF
            ok = Right_Rotate(x->parent->parent);
            if(!ok){
                x->parent->parent->parent->flag = false;
                return false; //This false is for "updateSucceeds"
            }else{
                x->parent->color = BLACK;
                x->parent->right->color = RED;
                *caseF = DID_CASE3;
                x->parent->parent->parent->flag = false; //remove the ggp flag as rotation was successful
                return true;
            }
        } 
        //symmetric to above code
    }else{
        //the parent is a right child
        xUncle = x->parent->parent->left;
        if(xUncle->color == RED){
            //CASE 1 - recoloring
            // U have all the flags u need. So this is simple, similar to serial code
            x->parent->color = BLACK;
            xUncle->color = BLACK;
            x->parent->parent->color = RED;
            *caseF = DID_CASE1;
            return true;
        }else{ // rotation(s) will be needed
            if(x == x->parent->left){//CASE2
                oldx = x; // save old x in case rotate fails
                x = x->parent;
                ok = Right_Rotate(x);
                if(!ok){
                    x = oldx; //undo change to x
                    return false;
                }
            }
            //In CASE 3, if the left-rotation fails,
            //CASE 3 fails but the algorithm still works
            //beacuse the process will return false to 
            //Insert_Rebalance, and Insert_Rebalance will
            //call Update_Rotation again to complete CASE3
            do{ // get great grandparent's flag
                // ggp = x->parent->parent->parent;
                ok = atomicCAS(&x->parent->parent->parent->flag,false,true);
            }while(!ok);
            ok = Left_Rotate(x->parent->parent);
            if(!ok){
                x->parent->parent->parent->flag = false;
                return false;
            }else{
                x->parent->color = BLACK;
                x->parent->left->color = RED;
                *caseF = DID_CASE3;
                x->parent->parent->parent->flag = false;
                return true;
            }
        }
    }
}

//A rotation will always be successful(true), as u can reach the rotate command
//only after u have cptured all the requried flags

__device__ bool Left_Rotate(struct par_rbNode *z){
    //z is the root of the rotation subtree. The locks
    // held at this point are : z,z->parent and z->right (and sibling of z but its not useful here)
    // bool ok;
    struct par_rbNode *zrl,*zr;

    if(z->parent == rtParent){
        //rotating at the root
        zrl = z->right->left;
        zr = z->right;
        // if a process has set the flag of a node q,
        //no other process can move one of the children of q away from q
        zrl->parent = z;
        z->right = zrl;
        // ok = CAS3(z->right,zrl,z->right,
        //      z->right,z,zrl->parent,
        //      zrl,zrl,z->right->left);
        //update other links
        root = zr;
        rtParent->left = root;
        root->parent = rtParent;
        z->parent = root;
        root->left = z;
    }else{
        //rotating under the root (parent, etc . exist)
        if(z == z->parent->left){
            //z is left child
            zrl = z->right->left;
            zr = z->right;
            // if a process has set the flag of a node q,
            //no other process can move one of the children of q away from q
            zrl->parent = z;
            z->right = zrl;
            //update other links
            z->parent->left = zr;
            z->right->parent = z->parent;
            z->parent = zr;
            z->right->left = z;
        }else{
            // z is right child
            zrl = z->right->left;
            zr = z->right;
            // if a process has set the flag of a node q,
            //no other process can move one of the children of q away from q
            zrl->parent = z;
            z->right = zrl;
            //update other links
            z->parent->right = zr;
            z->right->parent = z->parent;
            z->parent = zr;
            z->right->left = z;
        }
    }
    return true;
}

//symmetric to Left_rotate
__device__ bool Right_Rotate(struct par_rbNode *z){
    //z is the root of the rotation subtree. The locks
    // held at this point are : z,z->parent and z->left (and sibling of z but its not useful here)
    // bool ok;
    struct par_rbNode *zrl,*zr;

    if(z->parent == rtParent){
        //rotating at the root
        zrl = z->left->right;
        zr = z->left;
        // if a process has set the flag of a node q,
        //no other process can move one of the children of q away from q
        zrl->parent = z;
        z->left = zrl;
        // ok = CAS3(z->left,zrl,z->left,
        //      z->left,z,zrl->parent,
        //      zrl,zrl,z->left->right);
        //update other links
        root = zr;
        rtParent->right = root;
        root->parent = rtParent;
        z->parent = root;
        root->right = z;
    }else{
        //rotating under the root (parent, etc . exist)
        if(z == z->parent->right){
            //z is right child
            zrl = z->left->right;
            zr = z->left;
            // if a process has set the flag of a node q,
            //no other process can move one of the children of q away from q
            zrl->parent = z;
            z->left = zrl;
            //update other links
            z->parent->right = zr;
            z->left->parent = z->parent;
            z->parent = zr;
            z->left->right = z;
        }else{
            // z is left child
            zrl = z->left->right;
            zr = z->left;
            // if a process has set the flag of a node q,
            //no other process can move one of the children of q away from q
            zrl->parent = z;
            z->left = zrl;
            //update other links
            z->parent->left = zr;
            z->left->parent = z->parent;
            z->parent = zr;
            z->left->right = z;
        }
    }
    return true;
}

__device__ void Insert(int key){
    struct par_rbNode *newNode = createNode(key); //Internally the flag of the newNode is held
    struct par_rbNode *insertPoint;
    // enum result {Success,Failure,FirstInsert};
    // Create and initialize the new node
    enum result res;
    //insert the new node
    do{
        //Traverse tree to find insertion point
        insertPoint = Traverse(key);
        if(insertPoint != NULL){
            //add new node to tree
            res = PlaceNode(newNode,insertPoint);
            printf("%d\n",res);
            // res is short for result (avoiding confusion b/w global enum and local variable)
            if(res == Success){
                //node was added succcessfully so make 
                //tree red-black again by doing the 
                //necessary color updates and rotations
                Insert_Rebalance(newNode);
            }
        }else{
            break;
        }
    }while(!(res==Success)||(res==FirstInsert));
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

__device__ int threadsFinished = 0;
__device__ int passCreate = 0;

__global__ void RBT(struct par_rbNode *d_nodes) {

    int id = blockIdx.x*blockDim.x+threadIdx.x;
    int threadCount = gridDim.x*blockDim.x;
    
    if(id == 0){
        printf("Starting the Tree\n");
        nodes = d_nodes; // Make it a global variable
        createNIL();
        createTree();
        atomicAdd(&passCreate,1);
    }

    while(1){
        if(passCreate){
            Insert(id);
            break;
        }
    }

    // //Print the time
    // //This will keep track of number of threads that are done
    atomicAdd(&threadsFinished,1);
    // // //Print the tree after all the threads are done
    if(threadsFinished == threadCount){
        if(id == 0){
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
        }
    }
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