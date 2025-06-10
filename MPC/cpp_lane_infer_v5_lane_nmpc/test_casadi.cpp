#include <casadi/casadi.hpp>

int main() {
    casadi::MX x = casadi::MX::sym("x");
    casadi::MX f = x*x + 1;
    std::cout << f << std::endl;
    return 0;
}