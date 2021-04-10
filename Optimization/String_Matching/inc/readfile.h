#include<stdio.h>
#include<string.h>

long getFileSize(const char *filename);
void readText(const char *filename, char* source, long bufsize);
void readFilesToArray(const char *filename, char* source, unsigned long bufsize, unsigned long size);
