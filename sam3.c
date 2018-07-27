#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

enum color {
	RED,
	BLUE,
	GREEN
};

enum color some_function(){
	return RED;
}

void foo1();
void foo(enum color *);


void foo1(){
	enum color c = RED;
	foo(&c);
	printf("%d\n",c);

}

void foo(enum color *x){
	*x = BLUE; 
}

// void now(int *x, int ptr1, int ptr2){
// 	if(*x == ptr1){
// 		ptr2 = ptr1;
// 	}
// }

int main(){
	int *ptr1, *ptr2, *x;
	*ptr1 = 10;
	x = ptr1;

	//passing x,ptr1,ptr2
	// now(x,(int)*ptr1,(int)*ptr2);
	printf("%d\n",*ptr1);
	printf("%ld\n",ptr1);
	// printf("%ld\n",&ptr1);




	return 0;
}