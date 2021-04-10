#include<stdio.h>
#include<string.h>
#include "../inc/readfile.h"

long getFileSize(const char *filename){
	long bufsize;
	FILE *fp = fopen(filename, "r");
	//printf("%s\n", filename);
	if (fp != NULL) {
	    /* Go to the end of the file. */
		if (fseek(fp, 0L, SEEK_END) == 0) {
        /* Get the size of the file. */
			bufsize = ftell(fp);
			if (bufsize == -1) { /* Error */ }
		}
	}
	fclose(fp);
	return bufsize;
}

void readText(const char *filename, char* source, long bufsize){
	FILE *fp = fopen(filename, "r");
	*source = 0;
	
	if (fp != NULL) {
		size_t newLen = fread(source, sizeof(char), bufsize, fp);
	    if (newLen == 0) {
		    printf("\nCannot read file\n");
			exit(1);
		} else {
			source[newLen] = '\0'; /* Just to be safe. */
		}
	}
	fclose(fp);

}
void readFilesToArray(const char *filename, char* source, unsigned long bufsize, unsigned long size)
{
	int i;
	int times = (size -1) / bufsize;
	*source = 0;
	//printf("\nBufsize = %u", bufsize);
	//printf("\nReal size = %u", size);
	for(i = 0; i< times ; i++){
		FILE *fp = fopen(filename, "r");
		if (fp != NULL) {
			size_t newLen = fread(source + i*bufsize, sizeof(char), bufsize, fp);
			if (newLen == 0) {
			    printf("\nCannot read file\n");
				exit(1);
			} 
		}
		fclose(fp);
	}
	//printf("\nEnd = %u", i*bufsize);
	source[i*bufsize] ='\0';
}
