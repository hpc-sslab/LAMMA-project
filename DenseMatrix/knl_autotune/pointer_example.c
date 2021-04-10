#include <stdio.h>

void hello()
{
	printf("Hello, world!\n");
}

void bonjour()
{
	printf("bonjour le monde!\n");
}

int main()
{
	void (*fp[2][2])();

	fp[0][0] = hello;
	fp[0][0]();

	fp[1][1] = bonjour;
	fp[1][1]();

	return 0;
}
