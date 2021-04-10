#define P_LOGFILE "log.txt"


void P_LogString(const char *message);
void P_LogNum(const char type,int numargs, ...);
void P_LogAll(const char *message, char type ,int numargs,  ...);
void P_LogErr(const char *message);
void P_LogStart();
