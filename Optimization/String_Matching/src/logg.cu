#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdarg.h>
#include "../inc/logg.h"

int p_LogCreated = 0;
void P_LogString(const char *message)
{
	FILE *file;
	if(!p_LogCreated){
		file = fopen("log.txt", "w");
		p_LogCreated = 1;
	}
	else
		file  = fopen("log.txt", "a+");

	if(file==NULL){
		if(p_LogCreated)
			p_LogCreated =0;
		return;
	}
	
	fprintf(file, message);
	fclose(file);
}

void P_LogNum(char type, int numargs, ...)
{
	va_list listPointer;
	FILE *file;
	if(!p_LogCreated){
		file = fopen("log.txt", "w");
		p_LogCreated = 1;
	}
	else
		file = fopen("log.txt", "a+");

	if(file==NULL){
		if(p_LogCreated)
			p_LogCreated =0;
		return;
	}
	
	va_start(listPointer, numargs);	
	switch(type)
	{
	case 'c':
		{
		char arg;
		arg = va_arg(listPointer, char);
		fprintf(file,"%c", arg);
		break;
		}
	case 'd':
	case 'i':
		{
			int arg;
		arg = va_arg(listPointer, int);
		fprintf(file,"%4d", arg);
		break;
		}
	case 'u':
	case 'o':
	case 'x':
	case 'X':
		{
			unsigned int arg;
		arg = va_arg(listPointer, unsigned int);
		fprintf(file,"%4d" ,arg);
		break;
		}
	case 'f':   // float/double
  case 'e':   // scientific double/float
  case 'E':   // scientific double/float
  case 'g':   // scientific double/float
  case 'G':   // scientific double/float
  case 'a':   // signed hexadecimal double precision float
  case 'A':   // signed hexadecimal double precision float
		{
			double arg;
		arg = va_arg(listPointer, double);
		fprintf(file,"%5.2f", arg);
		break;
		}
	}
	va_end(listPointer);
		fclose(file);
}
void P_LogAll(const char *message, char type , int numargs, ...)
{
	va_list listPointer;
	
	FILE *file;
	if(!p_LogCreated){
		file = fopen("log.txt", "w");
		p_LogCreated = 1;
	}
	else
		file = fopen("log.txt", "a+");

	if(file==NULL){
		if(p_LogCreated)
			p_LogCreated =0;
		return;
	}
	fprintf(file,message);

	
	va_start(listPointer, numargs);
	switch(type)
	{
	case 'c':
		{
		char arg;
		arg = va_arg(listPointer, char);
		fprintf(file,"%c", arg);
		break;
		}
	case 'd':
	case 'i':
		{
		int arg;
		arg = va_arg(listPointer, int);
		fprintf(file,"%4d", arg);
		break;
		}
	case 'u':
	case 'o':
	case 'x':
	case 'X':
		{
			unsigned int arg;
		arg = va_arg(listPointer, unsigned int);
		fprintf(file,"%4d", arg);
		break;
		}
	case 'f':   // float/double
  case 'e':   // scientific double/float
  case 'E':   // scientific double/float
  case 'g':   // scientific double/float
  case 'G':   // scientific double/float
  case 'a':   // signed hexadecimal double precision float
  case 'A':   // signed hexadecimal double precision float
		{
			double arg;
		arg = va_arg(listPointer, double);
		fprintf(file,"%.8f", arg);
		break;
		}
	}
	va_end(listPointer);
	fclose(file);

}

void P_LogErr(const char *message){
	P_LogString(message);
	P_LogString("\n");
	exit(0);
}
void P_LogStart()
{
	remove(P_LOGFILE);
	p_LogCreated = 0;
}
