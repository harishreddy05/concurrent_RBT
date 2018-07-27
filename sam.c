#include <stdio.h>
#include <stdlib.h>

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

int main(){

	foo1();
	return 0;
}