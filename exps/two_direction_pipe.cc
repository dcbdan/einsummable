#include <sys/types.h>  
#include <stdio.h>   
#include <stdlib.h>   
#include <string.h>   
#include <unistd.h>   
#include <sys/wait.h>

#define STDIN_FILENO    0       /* Standard input.  */ 
#define STDOUT_FILENO   1       /* Standard output.  */ 
#define STDERR_FILENO   2       /* Standard error output.  */ 
#define MAXLINE 4096 

// Source: https://stackoverflow.com/questions/3653521/how-to-execute-python-script-from-c-code-using-execl

int main(void){ 
    int  n, parent_child_pipe[2], child_parent_pipe[2]; 
    pid_t pid; 
    char line[MAXLINE]; 
    int rv;

    if (pipe(parent_child_pipe) < 0 || pipe(child_parent_pipe) < 0){
        puts("Error creating pipes...\n"); 
    }
    

    pid = fork();

    if(pid < 0){ // fork failed
        puts("Error forking...\n"); 
    }
    else if (pid > 0) { /* PARENT */ 
        while (fgets(line, MAXLINE, stdin) != NULL) { 
            write(parent_child_pipe[1], line, strlen(line)) != strlen(line);
            if ( (n = read(child_parent_pipe[0], line, MAXLINE)) < 0) {
                puts("read error from pipe...\n"); 
            }
            line[n] = 0; /* null terminate */ 
            if (fputs(line, stdout) == EOF) 
                puts("fputs error...\n"); 
        } 
        if (ferror(stdin)){
            puts("fgets error on stdin...\n"); 
        }
        exit(0); 

    } 
    else {  /* CHILD */ 
        dup2(parent_child_pipe[0], STDIN_FILENO);
        dup2(child_parent_pipe[1], STDOUT_FILENO);
        close(parent_child_pipe[1]); 
        close(child_parent_pipe[0]); 
        
        if (execl("./hello_parent.py", "./hello_parent.py", (char *) 0) < 0){
            puts("execl error...\n"); 
        }
    } 
} 