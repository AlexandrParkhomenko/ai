/*
https://en.wikipedia.org/wiki/Quadratic_equation
*/

#include <iostream> // std::cout
#include <utility> // std::pair
#include <cmath>  // sqrt

std::pair<bool, std::pair<double, double>> quadratic (int a, int b, int c) {
    double inside = b*b - 4*a*c;
    std::pair<double, double> blank;
    if (inside < 0) return std::make_pair(false, blank);
    std::pair<double, double> answer = std::make_pair( (-b-sqrt(inside))/(2*a), 
                                                       (-b+sqrt(inside))/(2*a) );
return std::make_pair(true, answer);
}

int main() {
   int a, b, c;
   std::cin >> a >> b >> c;   // this gets input
   std::pair<bool, std::pair<double, double>> result = quadratic(a, b, c);
   if (result.first) {
      std::pair<double, double> solutions = result.second;
      std::cout << solutions.first << "; " << solutions.second << std::endl;
   } else {
      std::cout << "No solutions found!" << std::endl;
   }
}
