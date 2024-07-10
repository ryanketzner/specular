#include <newton.h>
#include <specular_gradient.h>
#include <Eigen/Dense>
#include <functional>

Eigen::Vector<T,3> get_specular_point(Eigen::Vector3<T,3> const& t, Eigen::Vector3<T,3> const& r
	T a, T b, T tol = 1e-6, int max_iter = 25)
{
	auto f = [&t,&r](Eigen::Matrix<T, 4, 1> const& x){return specular_distance(x,t,r);};
	auto g = [&t,&r](Eigen::Vector<T, 4> const& x){return specular_gradient(x,t,r);};
	auto h = [&t,&r](Eigen::Vector<T, 4> const& x){return specular_hessian(x,t,r);};

	Eigen::Vector<T,4> x0 = get_x0(t,r);
	Result<T,4> result = newtons_method(f,g,h,x0);

	return result.x_opt.template head<3>;
}
