#include "../src/reference.h"

int main() {
  uint64_t ni = 3;
  uint64_t nj = 4;
  uint64_t nk = 5;

  einsummable_t matmul = einsummable_t::from_matmul(ni,nj,nk);

  buffer_t lhs = std::make_shared<buffer_holder_t>(ni*nj);
  buffer_t rhs = std::make_shared<buffer_holder_t>(nj*nk);

  lhs->iota(8);
  rhs->iota(100);

  // def f(s0, s1, ni, nj, nk):
  //     x = np.array([i for i in range(s0, s0+ni*nj)]).reshape((ni,nj))
  //     y = np.array([i for i in range(s1, s1+nj*nk)]).reshape((nj,nk))
  //     z = np.dot(x,y)
  //     print(z.reshape(-1))
  // f(0, 100, 3, 3, 3) = [18,321,1242,1254,1266,2169,2190,2211]
  // f(8, 100, 3, 4, 5) = [4110 4148 4186 4224 4262 5830 5884 5938 5992 6046 7550 7620 7690 7760 7830]

  buffer_t out = reference_einsummable(matmul, {lhs, rhs});

  std::cout << lhs << " " << rhs << " " << out << std::endl;
}
