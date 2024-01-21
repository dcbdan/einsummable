#include "../src/base/setup.h"

// This class runs ./filename <args> and then
// messages can be sent to and from that executable.
// The implementation is repackaged from here
//   https://stackoverflow.com/questions/6171552/popen-simultaneous-read-and-write
struct piper_t {
  piper_t(
    string filename,
    vector<string> args);

  piper_t(
    string filename,
    string arg)
    : piper_t(filename, vector<string>{ arg })
  {}

  ~piper_t();

  string read();

  void write(string const& str);

private:
  pid_t pid = 0;
  int inpipefd[2];
  int outpipefd[2];
  int status;
};
