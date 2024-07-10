#ifndef specular_h
#define specular_h

#include <newton.h>
#include <specular_functions.h>
#include <Eigen/Dense>
#include <functional>

template <typename T>
Eigen::Vector3<T> get_specular_point(Eigen::Vector3<T> const& t, Eigen::Vector3<T> const& r,
	T a, T b, T tol = 1e-6, int max_iter = 25)
{
	auto f = [&t,&r](Eigen::Matrix<T, 4, 1> const& x){return specular_distance(x,t,r);};
	auto g = [&t,&r,a,b](Eigen::Vector<T, 4> const& x){return specular_gradient(x,t,r,a,b);};
	auto h = [&t,&r,a,b](Eigen::Vector<T, 4> const& x){return specular_hessian(x,t,r,a,b);};

	Eigen::Vector<T,4> x0 = get_x0(t,r,a,b);
	Result<T,4> result = newtons_method(f,g,h,x0,tol,max_iter);

	return result.x_opt.template head<3>();
}
#endif