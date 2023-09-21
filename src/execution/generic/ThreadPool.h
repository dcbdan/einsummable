
#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <string>
#include <vector>
#include <memory>
#include <iostream>

using namespace std;

class ExecutableNode;

// this is an example resource: a threadpool.  Those nodes that need threads from the thread pool
// will wait on this resource
class ThreadPool {

	friend void *runForever (void *arg);

private:

	// the number of threads in our pool
	int numThreads;

	// the ID of this resource
	int myID;

	// these are used to queue up tasks to run
	vector <void *> args;
	vector <void *(*)(void *)> funcs;

	// the threads
	vector <pthread_t> allThreads;

	// used to start and stop the threads
	pthread_cond_t threadPoolCond;
	pthread_mutex_t threadPoolMutex;

	// this is true when it is time to end
	bool timeToDie = false;

	// when a thread eners the pool, it calls this
	void enter ();
public:

	// create a thread pool with the given ID, and the given number of threads
	ThreadPool (int numThreadsToRun, int myID);

	~ThreadPool (); 

	// all of the standard Resource functions
	unsigned  getID ();
	string getName ();
	bool isAvailable ();
	void weAreUsingYou (ExecutableNode &nodeUsingTheResource);
	void weAreDoneUsingYou (ExecutableNode &nodeUsingTheResource);
		
	// when someone wants to use a thread in the thread pool, they call this with the entry
	// function and the parameters
	void run (void *funcToRun(void *), void *argsIn);
};

#endif
