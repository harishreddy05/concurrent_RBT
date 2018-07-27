#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define M 100000 // Total number of nodes in the tree
// RED = 0, BLACK = 1
enum nodeColor {
    RED,
    BLACK
};

struct rbNode {
    int data, color;
    struct rbNode *left, *right, *parent;
};

struct rbNode *root;

struct rbNode *NIL;

struct rbNode * create_NIL(){
    NIL = (struct rbNode *)malloc(sizeof(struct rbNode));
    NIL->color = BLACK;
    NIL->data  = -1;
    NIL->left = NIL->right = NIL->parent = NIL;
    return NIL;
}

struct rbNode * createNode(int d) {
    struct rbNode *newnode;
    newnode = (struct rbNode *)malloc(sizeof(struct rbNode));
    newnode->data = d;
    newnode->color = RED;
    newnode->left = newnode->right = newnode->parent = NIL;
    return newnode;
}

//FUNCTION TO BUILD A DUMMY TREE
// void createDummyTree(){
// 	struct rbNode *p;
// 	root = createNode(10);
// 	root->color = BLACK;
// 	root->left = createNode(6);
// 	root->right = createNode(23);
// 	//Constructing left sub tree
// 	p = root->left;
// 	p->parent = root;
	
// 	p->left = createNode(3);
// 	p->left->color = BLACK;
// 	p->left->parent = p;
	
// 	p->right = createNode(8);
// 	p->right->color = BLACK;
// 	p->right->parent = p;
	
// 	p = p->right;
// 	p->left = createNode(7);
// 	p->left->parent = p;

// 	//Constructing left sub tree
// 	p = root->right;
// 	p->parent = root;
	
// 	p->left = createNode(13);
// 	p->left->color = BLACK;
// 	p->left->parent = p;

// 	p->right = createNode(33);
// 	p->right->color = BLACK;
// 	p->right->parent = p;


// 	p = p->left;
// 	p->right = createNode(14);
// 	p->right->parent = p;

// 	p = p->parent->right;
// 	p->right = createNode(43);
// 	p->right->parent = p;
// }
//DUMMY TREE CONSTRUCTED
void printPreorder(struct rbNode* node)
{
	if (node == NIL)
	    return;
	/* first print the data of node */
	printf("%d-", node->data);
	printf("%d", node->color);
	printf("  ");  
	/* then recur on left child */
	printPreorder(node->left);
	/* now recur on right child */
	printPreorder(node->right);
}
void printInorder(struct rbNode* node)
{
	if (node == NIL)
	    return;
	/* first recur on left child */
	printInorder(node->left);
	/* then print the data of node */
	printf("%d-", node->data);
	printf("%d", node->color);
	printf("  ");  
	/* now recur on right child */
	printInorder(node->right);
}
void printPostorder(struct rbNode* node)
{
	if (node == NIL)
	    return;
	/* first recur on left child */
	printPostorder(node->left);
	/* then recur on right child */
	printPostorder(node->right);
	/* now print the data of node */
	printf("%d-", node->data);
	printf("%d", node->color);
	printf("  ");  
}
void clocked_printInorder(struct rbNode* node)
{
	clock_t start_t = clock();
	printInorder(node);
	printf("\n");
	clock_t end_t = clock();
	clock_t total_t = end_t - start_t;
	printf("Time taken for printing the tree = %fs\n",(double)total_t/CLOCKS_PER_SEC);
}

struct rbNode * Traverse(int d){
	struct rbNode *x;
    x = root;
    if(x == NIL){
    	printf("Empty Tree\n");
    	return NIL; 
	}
	while(x != NIL){
		if(x->data == d){
			printf("Found it!\n");
			return x;
		}else if(x->data > d){
			x = x->left;
		}else{
			x = x->right;
		}
	}
	printf("Couldn't find %d in this tree\n",d);
	return NIL; 
}

struct rbNode * clocked_Traverse(int d){
	struct rbNode *x;
	clock_t start_t = clock();
	x = Traverse(d);
	clock_t end_t = clock();
	clock_t total_t = end_t - start_t;
	printf("Time taken for Traversing in CPU = %fs\n",(double)total_t/CLOCKS_PER_SEC);
	return x;
}

void Left_Rotate(struct rbNode *lptr){
	struct rbNode *y;
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

void Right_Rotate(struct rbNode *rptr){
	struct rbNode *y;
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

void Insert_fixup(struct rbNode *x){
	struct rbNode *u;
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
			}	//CASE 3
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
			}	//CASE 3
		}
	}
	root->color = BLACK;
}

void Insert(int d){
	if(root == NIL){
		root = createNode(d);
		root->color = BLACK;
		return;
	}
	struct rbNode *x,*z;
	x = root;
	while(x != NIL){
		z = x;
		if(d == x->data){ // Find if the node with this value is already there or not
			printf("Duplicate Nodes are not allowed\n");
			return;
		}
		if(d < x->data)
			x = x->left;
		else
			x = x->right;
	}//end while
	x=createNode(d);
	x->parent = z;
	if(x->data < z->data) //Check if y is the left child of z or not
		z->left = x;
	else
		z->right = x;
	//NEW NODE IS INSERTED, NOW FIX THE RB TREE
	Insert_fixup(x);
	// printInorder(root);
}

void clocked_Insert(int d){
	clock_t start_t = clock();
	Insert(d);
	printf("\n");
	clock_t end_t = clock();
	clock_t total_t = end_t - start_t;
	printf("Time taken to insert by the CPU = %fs\n",(double)total_t/CLOCKS_PER_SEC);
}

int main()
{
	struct rbNode *ptr;
	root = create_NIL();//initially root points to NIL
	// createDummyTree();	
	int option,key;
	
	clock_t start_t = clock();
	for(int i=0;i<M;i++){
		Insert(i);
		// printf("Inserted %d\n",i);
	}
	clock_t end_t = clock();
	clock_t total_t = end_t - start_t;
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
	printf("Time taken by CPU = %fs\n",(double)total_t/CLOCKS_PER_SEC);
	// while(1){
	// 	printf("1.Traverse, 2.Insert, 3.Delete, 4.Print, 5.Exit\n");
	// 	scanf("%d",&option);
	// 	printf("\n");
	// 	switch(option){
	// 		case 1:
	// 				printf("The value you wish to find = ");
	// 				scanf("%d",&key);
	// 				ptr = clocked_Traverse(key);
	// 				if(ptr != NIL){
	// 					printf("Color of node = %d\n", ptr->color);
	// 					if(ptr->left != NIL)
	// 						printf("Left Child = %d\n",ptr->left->data);
	// 					else
	// 						printf("No Left Child\n");
	// 					if(ptr->right != NIL)
	// 						printf("Right Child = %d\n",ptr->right->data);
	// 					else
	// 						printf("No Right Child\n");
	// 					if(ptr->parent != NIL)
	// 						printf("Parent = %d\n",ptr->parent->data);
	// 					else
	// 						printf("I am the boss\n");
	// 				}	
	// 				printf("\n");
	// 				break;
	// 		case 2:
	// 				printf("The value you wish to insert = ");
	// 				scanf("%d",&key);
	// 				clocked_Insert(key);
	// 				printf("\n");
	// 				break;
	// 		case 3:
	// 				printf("Under Construction\n");
	// 				printf("\n");
	// 				break;
	// 		case 4:
	// 				printf("PreOrder: ");
	// 				printPreorder(root);
	// 				printf("\n");
	// 				printf("InOrder: ");
	// 				printInorder(root);
	// 				printf("\n");
	// 				printf("PostOrder: ");
	// 				printPostorder(root);
	// 				printf("\n");
	// 				printf("\n");
	// 				break;
	// 		case 5:
	// 				exit(0);
	// 				break;
	// 		default:
	// 				printf("Enter a valid option\n");
	// 	}
	// }
	return 0;
}