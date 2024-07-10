#include <Eigen/Dense>
#include <functional>
#include <cmath>
#include <iostream>

template<typename T, int Dim>
struct Result
{
    Eigen::Matrix<T, Dim, 1> x_opt;
    T f_opt;
    int iter;
};

template<typename T, int Dim, typename Func, typename Grad, typename Hess>
Result<T, Dim> newtons_method(Func const& f, Grad const& grad_f, Hess const& hess_f, Eigen::Matrix<T, Dim, 1> const& x0,
    T tol = 1e-6, int max_iter = 25)
{
    Eigen::Matrix<T, Dim, 1> x = x0;
    Eigen::Matrix<T, Dim, 1> g;
    Eigen::Matrix<T, Dim, Dim> H;
    Eigen::Matrix<T, Dim, 1> p;
    int iter = 0;

    while (iter < max_iter)
    {
        g = grad_f(x);

        if (g.norm() < tol)
            break;

        H = hess_f(x);

        x.noalias() += -H.ldlt().solve(g);

        iter++;
    }

    Result<T, Dim> result;
    result.x_opt = x;
    result.f_opt = f(x);
    result.iter = iter;

    return result;
}

template<typename T, int Dim, typename Func, typename Grad, typename Hess>
Result<T, Dim> newtons_method_fast(Func const& f, Grad const& grad_f, Hess const& hess_f, Eigen::Matrix<T, Dim, 1> const& x0,
    T tol = 1e-6, int max_iter = 25)
{
    Eigen::Matrix<T, Dim, 1> x = x0;
    Eigen::Matrix<T, Dim, 1> g;
    Eigen::Matrix<T, Dim, Dim> H;
    Eigen::Matrix<T, Dim, 1> p;
    int iter = 0;

    while (iter < max_iter)
    {
        grad_f(x,g);

        if (g.norm() < tol)
            break;

        hess_f(x,H);

        x.noalias() += -H.ldlt().solve(g);

        iter++;
    }

    Result<T, Dim> result;
    result.x_opt = x;
    result.f_opt = f(x);
    result.iter = iter;

    return result;
}

