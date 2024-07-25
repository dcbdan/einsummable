#include "piper.h"

#include <unistd.h>
#include <sys/wait.h>
#include <sys/prctl.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

piper_t::piper_t(string filename, vector<string> args)
{
    int _a = pipe(inpipefd);
    int _b = pipe(outpipefd);

    pid = fork();
    if (pid == 0) {
        // Child
        dup2(outpipefd[0], STDIN_FILENO);
        dup2(inpipefd[1], STDOUT_FILENO);
        dup2(inpipefd[1], STDERR_FILENO);

        // ask kernel to deliver SIGTERM in case the parent dies
        prctl(PR_SET_PDEATHSIG, SIGTERM);

        // close unused pipe ends
        close(outpipefd[1]);
        close(inpipefd[0]);

        // call the executable
        vector<char*> args_;
        args_.push_back(filename.data());
        for (auto& arg : args) {
            args_.push_back(arg.data());
        }
        args_.push_back(NULL);
        execvp(filename.c_str(), args_.data());

        // Nothing below this line should be executed by child process. If so,
        // it means that the execl function wasn't successfull, so lets exit:
        throw std::runtime_error("should not reach!");
    }
}

piper_t::~piper_t()
{
    kill(pid, SIGKILL); // send SIGKILL signal to the child process
    waitpid(pid, &status, 0);
    close(outpipefd[1]);
    close(inpipefd[0]);
}

string piper_t::read()
{
    string msg;
    while (true) {
        uint64_t n = msg.size();
        msg.resize(n + 1024);
        char* raw = msg.data() + n;
        auto  r = ::read(inpipefd[0], raw, 1024);
        if (r < 0) {
            throw std::runtime_error("could not read!");
        }
        if (r < 1024) {
            msg.resize(n + r);
            return msg;
        }
    }
}

void piper_t::write(string const& msg)
{
    auto r = ::write(outpipefd[1], msg.data(), msg.size());
    if (r != msg.size()) {
        throw std::runtime_error("invalid write size");
    }
}
