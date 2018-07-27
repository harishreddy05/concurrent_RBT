#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

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

/*Function prototypes */
void CreateTree();
struct par_rbNode * createNode(int);
struct par_rbNode * Traverse(int);
enum result PlaceNode(struct par_rbNode *, struct par_rbNode *);
void Insert_Rebalance(struct par_rbNode *);
bool Update_Rotation(struct par_rbNode *, enum caseFlag *);
bool Left_Rotate(struct par_rbNode *);
bool Right_Rotate(struct par_rbNode *);
/**/

struct par_rbNode {
    int key, color;
    bool flag;
    struct par_rbNode *left, *right, *parent;
};

struct par_rbNode *root;
struct par_rbNode *NIL;
struct par_rbNode *rtParent;
struct par_rbNode *rtSibling; // U might feel this is unncessary, but it will be used

void CreateTree(){
    rtParent = (struct par_rbNode *)malloc(sizeof(struct par_rbNode));
    rtSibling = (struct par_rbNode *)malloc(sizeof(struct par_rbNode));
    NIL = (struct par_rbNode *)malloc(sizeof(struct par_rbNode));
    root = NIL;
    rtParent->parent = NIL;
    rtSibling->parent = rtParent;
    rtSibling->right = NIL;
    rtSibling->left	= NIL;
    rtParent->left = root; 
    //rtParent->left = root; Why only left, y not right?
    //ANS: Since we check for left parent condition first 
    //(if u don't understand, try to insert a node to a tree with only one node)
    rtParent->right = rtSibling;
    rtParent->flag = false;
    rtSibling->flag = false;
    rtParent->color = BLACK;
    rtSibling->color = BLACK;
    NIL->left = NIL;
    NIL->right = NIL;
    NIL->parent = rtParent;
    NIL->flag = false;
    NIL->color = BLACK;
}

struct par_rbNode *createNode(int key) {
    struct par_rbNode *newnode;
    newnode = (struct par_rbNode *)malloc(sizeof(struct par_rbNode));
    newnode->key = key;
    newnode->color = RED;
    newnode->flag = true;
    newnode->left = newnode->right = newnode->parent = NIL;
    return newnode;
}

struct par_rbNode * Traverse(int key){
	struct par_rbNode *x;
	struct par_rbNode *insertPoint = NIL;
	struct par_rbNode *savert, *savec, *savelc;
	bool success;

	do{
		savert = root;
		success = DCAS(&root->flag,false,true,&root,savert,savert); //Catching the flag of the root
	}while(!success);

	x = root;
	while(x != NIL){
		if(key == x->key) {
			x->flag = false; // Release the flag that was just caught
			printf("Error_KeyExists");
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

enum result PlaceNode(struct par_rbNode *newNode, struct par_rbNode *insertPoint){
	//flags on newNode and insertPoint are held
	bool ok;
	struct par_rbNode *uncle,*savep;

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
			savep = insertPoint->parent; // save parent ptr
			uncle = savep->right;	// rtSibling is used here, when insertPoint is root
			ok = atomicCAS(&savep->flag,false,true);
			if(ok){
				ok = atomicCAS(&uncle->flag,false,true);
				if(ok){
					ok = DCAS(&insertPoint->parent,savep,savep,
						&savep->right,uncle,uncle);
					if(!ok){ //back off
						savep->flag = false;
						uncle->flag = false;
					}
				}else{
					savep->flag = false;
				}//end if
			}
		}else{// uncle is left child
			savep = insertPoint->parent; // save parent ptr
			uncle = savep->left;
			ok = atomicCAS(&savep->flag,false,true);
			if(ok){
				ok = atomicCAS(&uncle->flag,false,true);
				if(ok){
					ok = DCAS(&insertPoint->parent,savep,savep,
						&savep->left,uncle,uncle);
					if(!ok){ //back off
						savep->flag = false;
						uncle->flag = false;
					}
				}else{
					savep->flag = false;
				}//end if
			}
		}//end if
		if(!ok){
		 // This "!ok" is when u fail to capture the grandparent flag,
		 // u haven't caught any extra flags so just get rid of the flag of insertPoint
			newNode->parent = NIL;
			insertPoint->flag = false; // release flag
			return Failure; 		//avoid deadlock 
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

void Insert_Rebalance(struct par_rbNode *x){	//THIS FUNCTION DOESN'T BACKOFF. IT KEEPS TRYING
	//we hold flags on x, p(x), p(p(x)), and uncle(x)
	struct par_rbNode *oldx;
	struct par_rbNode *uncle, *olduncle;
	struct par_rbNode *savep, *savegp;
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
			oldx = x; 	//save pointer to the old x
			olduncle = uncle; // save pointer to old uncle;
			x = x->parent->parent; // up to grandparent
			do{	//find new uncle of x and get flags
				if(x->parent == x->parent->parent->left){
					savep = x->parent;
					savegp = savep->parent;
					uncle = savegp->right;
					ok = atomicCAS(&savep->flag,false,true);
					if(ok){
						ok = atomicCAS(&savegp->flag,false,true);
						if(ok){
							ok = atomicCAS(&uncle->flag,false,true);
							if(ok){
								ok = DCAS(&x->parent,savep,savep,
									&savep->parent,savegp,savegp);
								if(ok){
									ok = atomicCAS(uncle,uncle,savegp->right);
								}
								if(!ok){
									savep->flag = false;
									savegp->flag = false;
									uncle->flag = false;
								}
							}else{
								savep->flag = false;
								savegp->flag = false;
							}
						}else{
							savep->flag = false;
						}
					}
				}else{
					savep = x->parent;
					savegp = savep->parent;
					uncle = savegp->left;
					ok = atomicCAS(&savep->flag,false,true);
					if(ok){
						ok = atomicCAS(&savegp->flag,false,true);
						if(ok){
							ok = atomicCAS(&uncle->flag,false,true);
							if(ok){
								ok = DCAS(&x->parent,savep,savep,
									&savep->parent,savegp,savegp);

								if(ok){
									ok = atomicCAS(&savegp->left,uncle,uncle);
								}
								if(!ok){
									savep->flag = false;
									savegp->flag = false;
									uncle->flag = false;
								}
							}else{
								savep->flag = false;
								savegp->flag = false;
							}
						}else{
							savep->flag = false;
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

bool Update_Rotation(struct par_rbNode *x, enum caseFlag *caseF){
	//we hold flags on x, p(x), p(p(x)) and uncle(x)
	struct  par_rbNode *xUncle;
	struct  par_rbNode *oldx, *ggp; // ggp -> greatgrandparent
	bool OK;

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
				OK = Left_Rotate(x);
				if(!OK){
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
				ggp = x->parent->parent->parent;
				OK = DCAS(&ggp->flag,false,true,
						&x->parent->parent->parent,ggp,ggp);
			}while(!OK);	//KEEPS TRYING, DOESN'T BACK OFF
			OK = Right_Rotate(x->parent->parent);
			if(!OK){
				ggp->flag = false;
				return false; //This false is for "updateSucceeds"
			}else{
				x->parent->color = BLACK;
				x->parent->right->color = RED;
				*caseF = DID_CASE3;
				ggp->flag = false; //remove the ggp flag as rotation was successful
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
				OK = Right_Rotate(x);
				if(!OK){
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
				ggp = x->parent->parent->parent;
				OK = DCAS(&ggp->flag,false,true,
						&x->parent->parent->parent,ggp,ggp);
			}while(!OK);
			OK = Left_Rotate(x->parent->parent);
			if(!OK){
				ggp->flag = false;
				return false;
			}else{
				x->parent->color = BLACK;
				x->parent->left->color = RED;
				*caseF = DID_CASE3;
				ggp->flag = false;
				return true;
			}
		}
	}
}

//A rotation will always be successful(true), as u can reach the rotate command
//only after u have cptured all the requried flags

bool Left_Rotate(struct par_rbNode *z){
	//z is the root of the rotation subtree. The locks
	// held at this point are : z,z->parent and z->right (and sibling of z but its not useful here)
	bool OK;
	struct par_rbNode *zrl,*zr;

	if(z->parent == rtParent){
		//rotating at the root
		zrl = z->right->left;
		zr = z->right;
		// if a process has set the flag of a node q,
		//no other process can move one of the children of q away from q
		zrl->parent = z;
		z->right = zrl;
		// OK = CAS3(z->right,zrl,z->right,
		// 		z->right,z,zrl->parent,
		// 		zrl,zrl,z->right->left);
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
bool Right_Rotate(struct par_rbNode *z){
	//z is the root of the rotation subtree. The locks
	// held at this point are : z,z->parent and z->left (and sibling of z but its not useful here)
	bool OK;
	struct par_rbNode *zrl,*zr;

	if(z->parent == rtParent){
		//rotating at the root
		zrl = z->left->right;
		zr = z->left;
		// if a process has set the flag of a node q,
		//no other process can move one of the children of q away from q
		zrl->parent = z;
		z->left = zrl;
		// OK = CAS3(z->left,zrl,z->left,
		// 		z->left,z,zrl->parent,
		// 		zrl,zrl,z->left->right);
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


void Insert(int key){
	struct par_rbNode *newNode, *insertPoint;
	// enum result {Success,Failure,FirstInsert};
	// Create and initialize the new node
	newNode = createNode(key); //Internally the flag of the newNode is held
	enum result res;
	//insert the new node
	do{
		//Traverse tree to find insertion point
		insertPoint = Traverse(key);
		if(insertPoint != NULL){
			//add new node to tree
			enum result res = PlaceNode(newNode,insertPoint); 
			//res is short for result (avoiding confusion b/w global enum and local variable)
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


int main(){


	return 0;
}
