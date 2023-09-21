
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <unistd.h>
#include "ThreadPool.h"

using namespace std;

void *runForever (void *arg);

ThreadPool :: ThreadPool (int numThreadsToRun, int myID) : myID (myID) {
		
	pthread_mutex_init (&threadPoolMutex, NULL);
	pthread_cond_init (&threadPoolCond, NULL);
	numThreads = 0;
	for (int i = 0; i < numThreadsToRun; i++) {
		pthread_t foo;
		allThreads.push_back (foo);
		pthread_create (&(allThreads[allThreads.size () - 1]), NULL, runForever, this);
	}

	// busy-wait for all threads to start up, since I am too lazy to use a conidtion variable
	while (numThreads != numThreadsToRun) {}

}

ThreadPool :: ~ThreadPool () {

	// join everyone
	timeToDie = true;

	// first signal all of the threads that may be blocked
	for (auto &a : allThreads) {
		pthread_mutex_lock (&threadPoolMutex);
		pthread_cond_signal (&threadPoolCond);
		pthread_mutex_unlock (&threadPoolMutex);
	}

	// and join them
	for (auto &a : allThreads) {
		pthread_join (a, NULL);
	}

	pthread_mutex_destroy (&threadPoolMutex);
	pthread_cond_destroy (&threadPoolCond);

}

void ThreadPool :: enter () {

	while (timeToDie == false) {
		
		pthread_mutex_lock (&threadPoolMutex);
		numThreads++;
		int size = args.size ();
		if (size == 0 && !timeToDie) {
			pthread_cond_wait (&threadPoolCond, &threadPoolMutex);
		}

		// make sure there is work to do
		size = args.size ();
		if (size == 0 || timeToDie) {
			pthread_mutex_unlock (&threadPoolMutex);
			continue;
		}

		// get the work
		auto func = funcs[size - 1];
		auto arg = args[size - 1];
		funcs.pop_back ();
		args.pop_back ();

		// do the work
		numThreads--;
		pthread_mutex_unlock (&threadPoolMutex);
		func (arg);
	}
}


unsigned ThreadPool :: getID () {
	return myID;
}

string ThreadPool :: getName () {
	return "Thread Pool";
}

bool ThreadPool :: isAvailable () {
	return numThreads > 0;
}

void ThreadPool :: weAreUsingYou (ExecutableNode &nodeUsingTheResource) {}

void ThreadPool :: weAreDoneUsingYou (ExecutableNode &nodeUsingTheResource) {}

// uses the thread pool to run a thread
void ThreadPool :: run (void *funcToRun(void *), void *argsIn) {
	
	// put the args in the list
	pthread_mutex_lock (&threadPoolMutex);
	args.push_back (argsIn);
	funcs.push_back (funcToRun);
	pthread_cond_signal (&threadPoolCond);
	pthread_mutex_unlock (&threadPoolMutex);

}

void *runForever (void *arg) {
	ThreadPool *myThreadPool = (ThreadPool *) arg;
	myThreadPool->enter ();
	return NULL;
}

