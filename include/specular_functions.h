#ifndef specular_functions_h
#define specular_functions_h

#include <Eigen/Dense>
#include <cmath>

template <typename T>
Eigen::Matrix<T,4,1> get_x0(const Eigen::Matrix<T, 3, 1>& t, const Eigen::Matrix<T, 3, 1>& r, T a, T b)
{
    // Project onto ellipsoid
    T scale = sqrt(r(0)*r(0) / (a*a) + r(1)*r(1) / (a*a) + r(2)*r(2) / (b*b));

    Eigen::Vector<T,4> result;
    result << r/scale, 0.0;

    return result;
}

template <typename T>
Eigen::Matrix<T,4,1> get_x0_mid(const Eigen::Matrix<T, 3, 1>& t, const Eigen::Matrix<T, 3, 1>& r, T a, T b)
{
	Eigen::Vector<T,3> midpoint = (t+r)/2.0;
	T scale = sqrt(midpoint(0)*midpoint(0) / (a*a) + midpoint(1)*midpoint(1) / (a*a) + midpoint(2)*midpoint(2) / (b*b));

	Eigen::Vector<T,4> result;
	result << midpoint/scale, 0.0;

	return result;
}

template <typename T>
T specular_distance(const Eigen::Matrix<T, 4, 1>& x, const Eigen::Matrix<T, 3, 1>& t,
	const Eigen::Matrix<T, 3, 1>& r)
{
    Eigen::Matrix<T, 3, 1> s = x.template head<3>();

    return (t-s).norm() + (r-s).norm();
}

template <typename T>
Eigen::Matrix<T, 4, 1> specular_gradient(const Eigen::Matrix<T, 4, 1>& x, const Eigen::Matrix<T, 3, 1>& t,
	const Eigen::Matrix<T, 3, 1>& r, T a, T b)
{    
    // Extracting s and lam from x
    Eigen::Matrix<T, 3, 1> s = x.template head<3>();
    T lam = x(3);

    // Precompute distances for readability
    T d1 = (s - t).norm();
    T d2 = (s - r).norm();

    // Gradient computation
    Eigen::Matrix<T, 4, 1> grad;
    grad(0) = (s(0) - t(0)) / d1 - (r(0) - s(0)) / d2 + (2 * lam * s(0)) / (a * a);
    grad(1) = (s(1) - t(1)) / d1 - (r(1) - s(1)) / d2 + (2 * lam * s(1)) / (a * a);
    grad(2) = (s(2) - t(2)) / d1 - (r(2) - s(2)) / d2 + (2 * lam * s(2)) / (b * b);
    grad(3) = (s(0) * s(0)) / (a * a) + (s(1) * s(1)) / (a * a) + (s(2) * s(2)) / (b * b) - 1;

    return grad;
}

template <typename T>
void specular_gradient(const Eigen::Matrix<T, 4, 1>& x, const Eigen::Matrix<T, 3, 1>& t,
    const Eigen::Matrix<T, 3, 1>& r, T a, T b, Eigen::Matrix<T,4,1> & grad)
{    
    // Extracting s and lam from x
    Eigen::Matrix<T, 3, 1> s = x.template head<3>();
    T lam = x(3);

    // Precompute distances for readability
    T d1 = (s - t).norm();
    T d2 = (s - r).norm();

    // Gradient computation
    grad(0) = (s(0) - t(0)) / d1 - (r(0) - s(0)) / d2 + (2 * lam * s(0)) / (a * a);
    grad(1) = (s(1) - t(1)) / d1 - (r(1) - s(1)) / d2 + (2 * lam * s(1)) / (a * a);
    grad(2) = (s(2) - t(2)) / d1 - (r(2) - s(2)) / d2 + (2 * lam * s(2)) / (b * b);
    grad(3) = (s(0) * s(0)) / (a * a) + (s(1) * s(1)) / (a * a) + (s(2) * s(2)) / (b * b) - 1;
}


template <typename T>
Eigen::Matrix<T, 4, 4> specular_hessian(const Eigen::Matrix<T, 4, 1>& x, const Eigen::Matrix<T, 3, 1>& t,
	const Eigen::Matrix<T, 3, 1>& r, T a, T b)
{    
    // Extracting s and lam from x
    Eigen::Matrix<T, 3, 1> s = x.template head<3>();
    T lam = x(3);

    // Precompute distances and their squares for readability
    Eigen::Matrix<T, 3, 1> s_minus_t = s - t;
    Eigen::Matrix<T, 3, 1> s_minus_r = s - r;
    T d1_squared = s_minus_t.squaredNorm();
    T d2_squared = s_minus_r.squaredNorm();
    T d1 = std::sqrt(d1_squared);
    T d2 = std::sqrt(d2_squared);

    // Precompute factors for readability
    T d1_cubed = d1 * d1_squared;
    T d2_cubed = d2 * d2_squared;

    // Initialize Hessian matrix
    Eigen::Matrix<T, 4, 4> hess = Eigen::Matrix<T, 4, 4>::Zero();

    // Diagonal elements
    hess(0, 0) = (2 * lam) / (a * a)
                 - s_minus_t(0) * s_minus_t(0) / d1_cubed
                 - s_minus_r(0) * s_minus_r(0) / d2_cubed
                 + 1 / d1 + 1 / d2;
    hess(1, 1) = (2 * lam) / (a * a)
                 - s_minus_t(1) * s_minus_t(1) / d1_cubed
                 - s_minus_r(1) * s_minus_r(1) / d2_cubed
                 + 1 / d1 + 1 / d2;
    hess(2, 2) = (2 * lam) / (b * b)
                 - s_minus_t(2) * s_minus_t(2) / d1_cubed
                 - s_minus_r(2) * s_minus_r(2) / d2_cubed
                 + 1 / d1 + 1 / d2;

    // Off-diagonal elements
    hess(0, 1) = hess(1, 0) = - s_minus_r(0) * s_minus_r(1) / d2_cubed
                               - s_minus_t(0) * s_minus_t(1) / d1_cubed;
    hess(0, 2) = hess(2, 0) = - s_minus_r(0) * s_minus_r(2) / d2_cubed
                               - s_minus_t(0) * s_minus_t(2) / d1_cubed;
    hess(1, 2) = hess(2, 1) = - s_minus_r(1) * s_minus_r(2) / d2_cubed
                               - s_minus_t(1) * s_minus_t(2) / d1_cubed;

    // Last row and column
    hess(0, 3) = hess(3, 0) = (2 * s(0)) / (a * a);
    hess(1, 3) = hess(3, 1) = (2 * s(1)) / (a * a);
    hess(2, 3) = hess(3, 2) = (2 * s(2)) / (b * b);

    // The last element hess(3, 3) is already zero as initialized

    return hess;
}

// Version which avoids new memory allocation
template <typename T>
void specular_hessian(const Eigen::Matrix<T, 4, 1>& x, const Eigen::Matrix<T, 3, 1>& t,
    const Eigen::Matrix<T, 3, 1>& r, T a, T b, Eigen::Matrix<T,4,4> & hess)
{    
    // Extracting s and lam from x
    Eigen::Matrix<T, 3, 1> s = x.template head<3>();
    T lam = x(3);

    // Precompute distances and their squares for readability
    Eigen::Matrix<T, 3, 1> s_minus_t = s - t;
    Eigen::Matrix<T, 3, 1> s_minus_r = s - r;
    T d1_squared = s_minus_t.squaredNorm();
    T d2_squared = s_minus_r.squaredNorm();
    T d1 = std::sqrt(d1_squared);
    T d2 = std::sqrt(d2_squared);

    // Precompute factors for readability
    T d1_cubed = d1 * d1_squared;
    T d2_cubed = d2 * d2_squared;

    // Diagonal elements
    hess(0, 0) = (2 * lam) / (a * a)
                 - s_minus_t(0) * s_minus_t(0) / d1_cubed
                 - s_minus_r(0) * s_minus_r(0) / d2_cubed
                 + 1 / d1 + 1 / d2;
    hess(1, 1) = (2 * lam) / (a * a)
                 - s_minus_t(1) * s_minus_t(1) / d1_cubed
                 - s_minus_r(1) * s_minus_r(1) / d2_cubed
                 + 1 / d1 + 1 / d2;
    hess(2, 2) = (2 * lam) / (b * b)
                 - s_minus_t(2) * s_minus_t(2) / d1_cubed
                 - s_minus_r(2) * s_minus_r(2) / d2_cubed
                 + 1 / d1 + 1 / d2;

    // Off-diagonal elements
    hess(0, 1) = hess(1, 0) = - s_minus_r(0) * s_minus_r(1) / d2_cubed
                               - s_minus_t(0) * s_minus_t(1) / d1_cubed;
    hess(0, 2) = hess(2, 0) = - s_minus_r(0) * s_minus_r(2) / d2_cubed
                               - s_minus_t(0) * s_minus_t(2) / d1_cubed;
    hess(1, 2) = hess(2, 1) = - s_minus_r(1) * s_minus_r(2) / d2_cubed
                               - s_minus_t(1) * s_minus_t(2) / d1_cubed;

    // Last row and column
    hess(0, 3) = hess(3, 0) = (2 * s(0)) / (a * a);
    hess(1, 3) = hess(3, 1) = (2 * s(1)) / (a * a);
    hess(2, 3) = hess(3, 2) = (2 * s(2)) / (b * b);

    hess(3,3) = 0;
}
#endif
