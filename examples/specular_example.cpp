#include <newton.h>
#include <specular_functions.h>
#include <Eigen/Dense>
#include <functional>
#include <iostream>

int main(void)
{
	using T = double;

	// Earth parameters
    T a = 6378.1370;
    T b = 6356.7523;

	Eigen::Vector<T,3> t;
	Eigen::Vector<T,3> r;
	t << 6071, 400, 3000;
	r << 6771, 1000, -4000;

	auto f = [&t,&r](Eigen::Matrix<T, 4, 1> const& x){return specular_distance(x,t,r);};
	auto g = [&t,&r,a,b](Eigen::Vector<T, 4> const& x){return specular_gradient(x,t,r,a,b);};
	auto h = [&t,&r,a,b](Eigen::Vector<T, 4> const& x){return specular_hessian(x,t,r,a,b);};

	Eigen::Vector<T,4> x0 = get_x0(t,r,a,b);

	Result<T,4> result;
	for (int i = 0; i < 50*86400; i++)
		result = newtons_method(f,g,h,x0);

	std::cout << "Final Distance: " << result.f_opt << std::endl;
	std::cout << "Iterations: " << result.iter << std::endl;
}