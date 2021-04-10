
/*
**   ACSMX.H 
**
**
*/

//#ifdef HAVE_CONFIG_H
//#include "config.h"
//#endif

//#include "sf_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef ACSMX_H
#define ACSMX_H

/*
*   Prototypes
*/
typedef long int uint32_t; 

#define ALPHABET_SIZE    256     

#define ACSM_FAIL_STATE   -1     

typedef struct _acsm_userdata
{
    uint32_t ref_count;
    void *id;

} ACSM_USERDATA;

typedef struct _acsm_pattern {      
   
    unsigned char         *patrn;
    unsigned char         *casepatrn;
    int      n;
    int      nocase;
    int      offset;
    int      depth;
    int      negative;
    int      iid;
	ACSM_USERDATA *udata;
	struct  _acsm_pattern *next;
    void   * rule_option_tree;
    void   * neg_list;

} ACSM_PATTERN;


typedef struct  {    

    /* Next state - based on input character */
    int      NextState[ ALPHABET_SIZE ];  

    /* Failure state - used while building NFA & DFA  */
    int      FailState;   

    /* List of patterns that end here, if any */
    ACSM_PATTERN *MatchList;   

}ACSM_STATETABLE; 


/*
* State machine Struct
*/
typedef struct {
	short bcShift[256];
    int acsmMaxStates;  
    int acsmNumStates;  
    int   bcSize;
    int numPatterns;
	ACSM_PATTERN    * acsmPatterns;
    ACSM_STATETABLE * acsmStateTable;
    void (*userfree)(void *p);
    void (*optiontreefree)(void **p);
    void (*neg_list_free)(void **p);

}ACSM_STRUCT;

/*
*   Prototypes
*/
ACSM_STRUCT * acsmNew (void (*userfree)(void *p),
                       void (*optiontreefree)(void **p),
                       void (*neg_list_free)(void **p));

int acsmAddPattern( ACSM_STRUCT * p, unsigned char * pat, int n,
          int nocase, int offset, int depth, int negative, void * id, int iid );

int acsmCompile ( ACSM_STRUCT * acsm,
             int (*build_tree)(void * id, void **existing_tree),
             int (*neg_list_func)(void *id, void **list));

int acsmSearch ( ACSM_STRUCT * acsm,unsigned char * T, int n, 
                 int (*Match)(void * id, void *tree, int index, void *data, void *neg_list),
                 void * data, int* current_state );

void acsmFree ( ACSM_STRUCT * acsm );
int acsmPatternCount ( ACSM_STRUCT * acsm );

int acsmPrintDetailInfo(ACSM_STRUCT *);

int acsmPrintSummaryInfo(void);

#endif
