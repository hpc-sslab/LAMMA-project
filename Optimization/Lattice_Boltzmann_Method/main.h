/*############################################################################*/

#ifndef _MAIN_H_
#define _MAIN_H_

/*############################################################################*/

#include "config.h"
#define SPEC
#if !defined(SPEC)
#include <sys/times.h>
#endif

/*############################################################################*/

#if !defined(SPEC)
typedef struct {
	double timeScale;
	clock_t tickStart, tickStop;
	struct tms timeStart, timeStop;

} MAIN_Time;
#endif

typedef enum {NOTHING = 0, COMPARE, STORE} MAIN_Action;
typedef enum {LDC = 0, CHANNEL} MAIN_SimType;

typedef struct {
	int nTimeSteps;
	char* resultFilename;
	MAIN_Action action;
	MAIN_SimType simType;
	char* obstacleFilename;
} MAIN_Param;

/*############################################################################*/

void MAIN_parseCommandLine( int nArgs, char* arg[], MAIN_Param* param );
void MAIN_printInfo( const MAIN_Param* param );
void MAIN_initialize( const MAIN_Param* param );
void MAIN_finalize( const MAIN_Param* param );

#if !defined(SPEC)
void MAIN_startClock( MAIN_Time* time );
void MAIN_stopClock( MAIN_Time* time, const MAIN_Param* param );
#endif

/*############################################################################*/

#endif /* _MAIN_H_ */