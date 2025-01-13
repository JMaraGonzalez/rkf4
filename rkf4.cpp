#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <cmath>
#include <limits>
#include <cinttypes>

#define REAL double
#define UINTG uint64_t
#define MAXSTEPS 100000000
#define DIVERGENCE_STR "Error: a potential divergence stopped the computation.\n"

using namespace std;

/* 
USER GUIDE:

0. The solver uses Runge-Kutta-Fehlberg of 4th order with adaptative step size (based on a 5th order
    computation). The precision is controlled with the maximum value of allowed steps and the 
    absolute tolerance allowed. The method requires an estimation of the number of steps needed
    (the value given is not very important)

1. There are 3 different routines [the order of the parameters is CRUCIAL]:
    ** rkf4: solves a first order ODE 
        y' = f(t, y) from t = [a, b], a < b, with boundary y(a)

    ** rkf4_2o: solves the 2-variable first order system
        y' = f(t, y, x) 
        x' = g(t, y, x) from t = [a, b], a < b, with boundary y(a), x(a)

    ** rkf4_vector: solves the n-variable first order system
        \vec y ' = \vec f(t, \vec y) from t = [a, b], a < b, with boundary \vec y(a)

2. Each returns a suitable 'cResult' structure, with 
    ** N: the number of points computed.
    ** errors: a C string of characters with informations about possible errors
    ** errors_len: length of the string 'errors'
    ** t: a vector containing the values of the independent variable
    ** THE DATA: depending on the routine, it can be either
        ++ one (or two) vector(s) y (x) with the computed values
        ++ an array of vectors containing the data as y[0], ..., y[n eqs.] 

3. It is IMPORTANT to release the data after usage, with a call to the appropriate
    ** 'clear_result' routine.

FOR DEVELOPER USAGE:
I. The code is prepared to be compiled as C code, so that it can be used from python for example.
II. By default, 'REAL' is defined as 'double', but this can be easily changed in the first lines
    of code. Same for 'UINTG'.
III. By default, there are some examples that can be removed from the compilation by 
commenting the first 'define' line after this guide

*/

#define __ALLOW_EXAMPLES_RK4V2_

extern "C"
{
        
    // IMPORTANT: the arrays used as y, t HAVE TO BE DYANMICALLY ALLOCATED
    struct cResult
    {
        char* errors;
        UINTG errors_len;
        REAL* y;
        REAL* t;
        UINTG N;
    };

    // IMPORTANT: the arrays used as y, t HAVE TO BE DYANMICALLY ALLOCATED
    struct cResult_2o
    {
        char* errors;
        UINTG errors_len;
        REAL* y;
        REAL* x;
        REAL* t;
        UINTG N;
    };

    // IMPORTANT: the arrays used as y, t HAVE TO BE DYANMICALLY ALLOCATED
    struct cResult_vector
    {
        char* errors;
        UINTG errors_len;
        REAL** y;
        REAL* t;
        UINTG n_vars;
        UINTG N;
    };

    void clear_result(cResult* res) {
        if (res->y != nullptr) {delete[] res->y; res->y=nullptr;}
        if (res->t != nullptr) {delete[] res->t; res->t=nullptr;}
        if (res->errors != nullptr) {delete[] res->errors; res->errors=nullptr;}
        res->N=0;
        res->errors_len=0;
    }

    void clear_result_2o(cResult_2o* res) {
        if (res->y != nullptr) {delete[] res->y; res->y=nullptr;}
        if (res->x != nullptr) {delete[] res->x; res->x=nullptr;}
        if (res->t != nullptr) {delete[] res->t; res->t=nullptr;}
        if (res->errors != nullptr) {delete[] res->errors; res->errors=nullptr;}
        res->N=0;
        res->errors_len=0;
    }

    void clear_result_vector(cResult_vector* res) {
        if (res->t != nullptr) {delete[] res->t; res->t=nullptr;}
        if (res->y != nullptr) {
            for (UINTG i=0; i<res->n_vars; i++) {
                if (res->y[i] != nullptr) {delete[] res->y[i]; res->y[i]=nullptr;}
            }
            delete[] res->y;
            res->y = nullptr;
        }
        if (res->errors != nullptr) {delete[] res->errors; res->errors=nullptr;}
        res->N=0;
        res->n_vars=0;
        res->errors_len=0;
    }

    
    cResult rkf4 (REAL (*f)(REAL, REAL), REAL a, REAL b, REAL ya, UINTG n_steps, UINTG max_steps, REAL absolute_tolerance)
    {
        const REAL big = 1e15;
        const REAL final_point_t_margin = 1e-5;
        REAL h = (b-a)/n_steps;

        const REAL a2 = 1.0/4, b2 = 1.0/4, a3 = 3.0/8, b3 = 3.0/32, c3 = 9.0/32, a4 = 12.0/13;
        const REAL b4 = 1932.0/2197, c4 = -7200.0/2197, d4 = 7296.0/2197, a5 = 1.0;
        const REAL b5 = 439.0/216, c5 = -8.0, d5 = 3680.0/513, e5 = -845.0/4104, a6 = 1.0/2;
        const REAL b6 = -8.0/27, c6 = 2.0, d6 = -3544.0/2565, e6 = 1859.0/4104;
        const REAL f6 = -11.0/40, r1 = 1.0/360, r3 = -128.0/4275, r4 = -2197.0/75240, r5 = 1.0/50;
        const REAL r6 = 2.0/55, n1 = 25.0/216, n3 = 1408.0/2565, n4 = 2197.0/4104, n5 = -1.0/5;

        string reserrors = "";
        REAL* rest = nullptr;
        REAL* resy = nullptr;
        UINTG resN = 0;

        if (n_steps > MAXSTEPS) {
            reserrors.append("Error: max number of memory allocation exceeded, reduce the \'n_steps\'.\n");
            cResult res = {nullptr, reserrors.length(), resy, rest, 0};
            res.errors = new char[reserrors.length()];
            memcpy(res.errors, reserrors.c_str(), reserrors.length());
            return res;
        }
        if (max_steps > MAXSTEPS) {
            reserrors.append("Warning: the value of \'max_steps\' if higher than the allowed one, in order to prevent memory issues. The limit will be set to the maximum value.\n");
            max_steps = MAXSTEPS;
        }

        // Figure this out
        REAL* T = new REAL[max_steps];
        REAL* Y = new REAL[max_steps];

        REAL k1, k2, k3, k4, k5, k6;
        REAL y1, y2, y3, y4, y5, y6;
        REAL err, ynew, s;

        // Check this type
        UINTG j = 0;
        UINTG step = 0;
        T[0] = a;
        Y[0] = ya;
        bool continue_flag = true; // controlled way of inerrupting the loop
        while (continue_flag)
        {
            if (T[j] + h > b) {
                h = b - T[j];
            }

            // Compute approximations
            k1 = h * f(T[j], Y[j]);
            y2 = Y[j] + b2 * k1;
            k2 = h * f(T[j] + a2 * h, y2);
            y3 = Y[j] + b3 * k1 + c3 * k2;
            k3 = h * f(T[j] + a3 * h, y3);
            y4 = Y[j] + b4 * k1 + c4 * k2 + d4 * k3;
            k4 = h * f(T[j] + a4 * h, y4);
            y5 = Y[j] + b5 * k1 + c5 * k2 + d5 * k3 + e5 * k4;
            k5 = h * f(T[j] + a5 * h, y5);
            y6 = Y[j] + b6 * k1 + c6 * k2 + d6 * k3 + e6 * k4 + f6 * k5;
            k6 = h * f(T[j] + a6 * h, y6);

            // Check for any divergence
            if (abs(y2) > big || abs(y3) > big || abs(y4) > big || abs(y5) > big || abs(y6) > big) {
                reserrors.append(DIVERGENCE_STR);
                break;
            }

            err = abs(r1*k1 + r3*k3 + r4*k4 + r5*k5 + r6*k6);
            ynew = Y[j] + n1 * k1 + n3 * k3 + n4 * k4 + n5 * k5;

            // Tolerance error and step size:
            step++;
            // If the new point is precise enough, we add it
            if (err < absolute_tolerance) {
                Y[j +1] = ynew;
                T[j +1] = T[j] + h;
                j++;
            }

            // Compute parameters for the next iteration
            if (err == 0.0) {
                s = 0;
            } else {
                s = 0.84 * pow(absolute_tolerance * h / err, 0.25);
                if (s<1.0) {
                    h = min(h*s, h / 2);
                } else {
                    h = max(h*s, 2 * h);
                }
            }

            // Check if can continue
            if (b - final_point_t_margin < T[j]) {
                continue_flag = false; // Evrything OK
            } else if (abs(Y[j]) > big) {
                reserrors.append(DIVERGENCE_STR);
                break;
            } else if (step > max_steps) {
                reserrors.append("Error: maximum number of steps reached. Computation stopped.\n");
                break;
            }
        }

        rest = new REAL[j+1];
        resy = new REAL[j+1];
        resN = j+1;
        for (UINTG i=0; i<j+1; i++) {
            rest[i] = T[i];
            resy[i] = Y[i];
        }

        // Cleanup and return
        delete[] T;
        delete[] Y;
        
        cResult res = {.errors= nullptr, .y= resy, .t= rest, .N= resN};
        res.errors = new char[reserrors.length()];
        memcpy(res.errors, reserrors.c_str(), reserrors.length());
        res.errors_len = reserrors.length();
        return res;

        // We will need some delete[] res.t cleanup;

    }
    

    cResult_2o rkf4_2o (REAL (*f)(REAL, REAL, REAL), REAL (*g)(REAL, REAL, REAL), REAL a, REAL b, REAL ya, REAL xa, UINTG n_steps, UINTG max_steps, REAL absolute_tolerance_y, REAL absolute_tolerance_x)
    {
        const REAL big = 1e15;
        const REAL final_point_t_margin = 1e-5;
        REAL h = (b-a)/n_steps;
        REAL h_x, h_y;

        const REAL a2 = 1.0/4, b2 = 1.0/4, a3 = 3.0/8, b3 = 3.0/32, c3 = 9.0/32, a4 = 12.0/13;
        const REAL b4 = 1932.0/2197, c4 = -7200.0/2197, d4 = 7296.0/2197, a5 = 1.0;
        const REAL b5 = 439.0/216, c5 = -8.0, d5 = 3680.0/513, e5 = -845.0/4104, a6 = 1.0/2;
        const REAL b6 = -8.0/27, c6 = 2.0, d6 = -3544.0/2565, e6 = 1859.0/4104;
        const REAL f6 = -11.0/40, r1 = 1.0/360, r3 = -128.0/4275, r4 = -2197.0/75240, r5 = 1.0/50;
        const REAL r6 = 2.0/55, n1 = 25.0/216, n3 = 1408.0/2565, n4 = 2197.0/4104, n5 = -1.0/5;

        string reserrors = "";
        REAL* rest = nullptr;
        REAL* resy = nullptr;
        REAL* resx = nullptr;
        UINTG resN = 0;

        if (n_steps > MAXSTEPS) {
            reserrors.append("Error: max number of memory allocation exceeded, reduce the \'n_steps\'.\n");
            cResult_2o res = {nullptr, reserrors.length(), resy, resx, rest, 0};
            res.errors = new char[reserrors.length()];
            memcpy(res.errors, reserrors.c_str(), reserrors.length());
            return res;
        }
        if (max_steps > MAXSTEPS) {
            reserrors.append("Warning: the value of \'max_steps\' if higher than the allowed one, in order to prevent memory issues. The limit will be set to the maximum value.\n");
            max_steps = MAXSTEPS;
        }

        // Figure this out
        REAL* T = new REAL[max_steps];
        REAL* Y = new REAL[max_steps];
        REAL* X = new REAL[max_steps];

        REAL k1, k2, k3, k4, k5, k6;
        REAL l1, l2, l3, l4, l5, l6;
        REAL y1, y2, y3, y4, y5, y6;
        REAL x1, x2, x3, x4, x5, x6;
        REAL err_x, err_y, ynew, xnew, s_x, s_y;

        // Check this type
        UINTG j = 0;
        UINTG step = 0;
        T[0] = a;
        Y[0] = ya;
        X[0] = xa;
        bool continue_flag = true; // controlled way of inerrupting the loop
        while (continue_flag)
        {
            if (T[j] + h > b) {
                h = b - T[j];
            }

            // Compute approximations
            k1 = h * f(T[j], Y[j], X[j]);
            l1 = h * g(T[j], Y[j], X[j]);
            y2 = Y[j] + b2 * k1;
            x2 = X[j] + b2 * l1;

            k2 = h * f(T[j] + a2 * h, y2, x2);
            l2 = h * g(T[j] + a2 * h, y2, x2);
            y3 = Y[j] + b3 * k1 + c3 * k2;
            x3 = X[j] + b3 * l1 + c3 * l2;

            k3 = h * f(T[j] + a3 * h, y3, x3);
            l3 = h * g(T[j] + a3 * h, y3, x3);
            y4 = Y[j] + b4 * k1 + c4 * k2 + d4 * k3;
            x4 = X[j] + b4 * l1 + c4 * l2 + d4 * l3;

            k4 = h * f(T[j] + a4 * h, y4, x4);
            l4 = h * g(T[j] + a4 * h, y4, x4);
            y5 = Y[j] + b5 * k1 + c5 * k2 + d5 * k3 + e5 * k4;
            x5 = X[j] + b5 * l1 + c5 * l2 + d5 * l3 + e5 * l4;

            k5 = h * f(T[j] + a5 * h, y5, x5);
            l5 = h * g(T[j] + a5 * h, y5, x5);
            y6 = Y[j] + b6 * k1 + c6 * k2 + d6 * k3 + e6 * k4 + f6 * k5;
            x6 = X[j] + b6 * l1 + c6 * l2 + d6 * l3 + e6 * l4 + f6 * l5;
            k6 = h * f(T[j] + a6 * h, y6, x6);
            l6 = h * g(T[j] + a6 * h, y6, x6);

            // Check for any divergence
            if (abs(y2) > big || abs(y3) > big || abs(y4) > big || abs(y5) > big || abs(y6) > big || abs(x2) > big || abs(x3) > big || abs(x4) > big || abs(x5) > big || abs(x6) > big) {
                reserrors.append(DIVERGENCE_STR);
                break;
            }

            err_y = abs(r1*k1 + r3*k3 + r4*k4 + r5*k5 + r6*k6);
            err_x = abs(r1*l1 + r3*l3 + r4*l4 + r5*l5 + r6*l6);
            ynew = Y[j] + n1 * k1 + n3 * k3 + n4 * k4 + n5 * k5;
            xnew = X[j] + n1 * l1 + n3 * l3 + n4 * l4 + n5 * l5;

            // Tolerance error and step size:
            step++;
            // If the new point is precise enough, we add it
            if (err_x < absolute_tolerance_x && err_y < absolute_tolerance_y) {
                Y[j +1] = ynew;
                X[j +1] = xnew;
                T[j +1] = T[j] + h;
                j++;
            }

            // Compute parameters for the next iteration
            if (err_x == 0.0) {
                s_x = 1.1;
            } else {
                s_x = 0.84 * pow(absolute_tolerance_x * h / err_x, 0.25);
                if (s_x<1.0) {
                    h_x = min(h*s_x, h / 2);
                } else {
                    h_x = max(h*s_x, 2 * h);
                }
            }

            if (err_y == 0.0) {
                s_y = 1.1;
            } else {
                s_y = 0.84 * pow(absolute_tolerance_y * h / err_y, 0.25);
                if (s_y<1.0) {
                    h_y = min(h*s_y, h / 2);
                } else {
                    h_y = max(h*s_y, 2 * h);
                }
            }
            h = min(h_x, h_y);

            // Check if can continue
            if (b - final_point_t_margin < T[j]) {
                continue_flag = false; // Evrything OK
            } else if (abs(Y[j]) > big || abs(X[j]) > big) {
                reserrors.append(DIVERGENCE_STR);
                break;
            } else if (step > max_steps) {
                reserrors.append("Error: maximum number of steps reached. Computation stopped.\n");
                break;
            }
        }

        rest = new REAL[j+1];
        resy = new REAL[j+1];
        resx = new REAL[j+1];
        resN = j+1;
        for (UINTG i=0; i<j+1; i++) {
            rest[i] = T[i];
            resy[i] = Y[i];
            resx[i] = X[i];
        }

        // Cleanup and return
        delete[] T;
        delete[] Y;
        delete[] X;
        
        cResult_2o res = {.errors= nullptr, .y= resy, .x= resx, .t= rest, .N= resN};
        res.errors = new char[reserrors.length()];
        memcpy(res.errors, reserrors.c_str(), reserrors.length());
        res.errors_len = reserrors.length();
        return res;

        // We will need some delete[] res.t cleanup;

    }
    
    // Seems to work, but needs further testing (compare to the other two methods)
    cResult_vector rkf4_vector (REAL (**f)(REAL, REAL*), REAL a, REAL b, UINTG n_vars, REAL* ya, UINTG n_steps, UINTG max_steps, REAL* absolute_tolerances)
    {
        const REAL big = 1e15;
        const REAL final_point_t_margin = 1e-5;
        REAL h = (b-a)/n_steps;

        const REAL a2 = 1.0/4, b2 = 1.0/4, a3 = 3.0/8, b3 = 3.0/32, c3 = 9.0/32, a4 = 12.0/13;
        const REAL b4 = 1932.0/2197, c4 = -7200.0/2197, d4 = 7296.0/2197, a5 = 1.0;
        const REAL b5 = 439.0/216, c5 = -8.0, d5 = 3680.0/513, e5 = -845.0/4104, a6 = 1.0/2;
        const REAL b6 = -8.0/27, c6 = 2.0, d6 = -3544.0/2565, e6 = 1859.0/4104;
        const REAL f6 = -11.0/40, r1 = 1.0/360, r3 = -128.0/4275, r4 = -2197.0/75240, r5 = 1.0/50;
        const REAL r6 = 2.0/55, n1 = 25.0/216, n3 = 1408.0/2565, n4 = 2197.0/4104, n5 = -1.0/5;

        string reserrors = "";
        REAL* rest = nullptr;
        REAL** res_vec = new REAL* [n_vars];

        UINTG i = 0; //For the loops
        for (i=0; i<n_vars; i++) {
            res_vec[i] = nullptr;
        }
        UINTG resN = 0;

        if (n_steps > MAXSTEPS) {
            reserrors.append("Error: max number of memory allocation exceeded, reduce the \'n_steps\'.\n");
            cResult_vector res;
            res.t = nullptr;
            res.y = nullptr;
            res.n_vars = n_vars;
            res.N = 0;
            res.errors = new char[reserrors.length()];
            memcpy(res.errors, reserrors.c_str(), reserrors.length());
            delete[] res_vec;
            return res;
        }
        if (max_steps > MAXSTEPS) {
            reserrors.append("Warning: the value of \'max_steps\' if higher than the allowed one, in order to prevent memory issues. The limit will be set to the maximum value.\n");
            max_steps = MAXSTEPS;
        }

        // Figure this out
        REAL* T = new REAL[max_steps];
        REAL** Y = new REAL* [n_vars];
        for (i=0; i<n_vars; i++) {
            Y[i] = new REAL[max_steps];
        }

        REAL K[6][n_vars];
        REAL ys[6][n_vars];
        REAL errs[n_vars];
        REAL ynews[n_vars];
        REAL ss[n_vars];
        REAL hs[n_vars];
        REAL hmin;

        /*REAL k1, k2, k3, k4, k5, k6;
        REAL l1, l2, l3, l4, l5, l6;
        REAL y1, y2, y3, y4, y5, y6;
        REAL x1, x2, x3, x4, x5, x6;
        REAL err_x, err_y, ynew, xnew, s_x, s_y;*/

        // Check this type
        UINTG j = 0;
        UINTG step = 0;
        T[0] = a;
        for (i=0; i<n_vars; i++) {
            Y[i][0] = ya[i];
        }
        bool continue_flag = true; // controlled way of inerrupting the loop
        bool successful_iteration = true;
        while (continue_flag)
        {
            if (T[j] + h > b) {
                h = b - T[j];
            }

            // Compute approximations
            for (i=0; i<n_vars; i++) {
                ys[0][i] = Y[i][j];
            }
            for (i=0; i<n_vars; i++) {
                K[0][i] = h * f[i](T[j], ys[0]);
                if( (ys[1][i] = Y[i][j] + b2 * K[0][i]) > big) {
                    reserrors.append(DIVERGENCE_STR);
                    break;
                }
            }
            /*k1 = h * f(T[j], Y[j], X[j]);
            l1 = h * g(T[j], Y[j], X[j]);
            y2 = Y[j] + b2 * k1;
            x2 = X[j] + b2 * l1;*/

            for (i=0; i<n_vars; i++) {
                K[1][i] = h * f[i](T[j] + a2 * h, ys[1]);
                if ( (ys[2][i] = Y[i][j] + b3 * K[0][i] + c3 * K[1][i]) > big) {
                    reserrors.append(DIVERGENCE_STR);
                    break;
                }
            }
            /*k2 = h * f(T[j] + a2 * h, y2, x2);
            l2 = h * g(T[j] + a2 * h, y2, x2);
            y3 = Y[j] + b3 * k1 + c3 * k2;
            x3 = X[j] + b3 * l1 + c3 * l2;*/

            for (i=0; i<n_vars; i++) {
                K[2][i] = h * f[i](T[j] + a3 * h, ys[2]);
                if ( (ys[3][i] = Y[i][j] + b4 * K[0][i] + c4 * K[1][i] + d4 * K[2][i]) > big) {
                    reserrors.append(DIVERGENCE_STR);
                    break;
                }
            }
            /*k3 = h * f(T[j] + a3 * h, y3, x3);
            l3 = h * g(T[j] + a3 * h, y3, x3);
            y4 = Y[j] + b4 * k1 + c4 * k2 + d4 * k3;
            x4 = X[j] + b4 * l1 + c4 * l2 + d4 * l3;*/

            for (i=0; i<n_vars; i++) {
                K[3][i] = h * f[i](T[j] + a4 * h, ys[3]);
                if ( (ys[4][i] = Y[i][j] + b5 * K[0][i] + c5 * K[1][i] + d5 * K[2][i] + e5 * K[3][i]) > big) {
                    reserrors.append(DIVERGENCE_STR);
                    break;
                }
            }
            /*k4 = h * f(T[j] + a4 * h, y4, x4);
            l4 = h * g(T[j] + a4 * h, y4, x4);
            y5 = Y[j] + b5 * k1 + c5 * k2 + d5 * k3 + e5 * k4;
            x5 = X[j] + b5 * l1 + c5 * l2 + d5 * l3 + e5 * l4;*/

            for (i=0; i<n_vars; i++) {
                K[4][i] = h * f[i](T[j] + a5 * h, ys[4]);
                if ( (ys[5][i] = Y[i][j] + b6 * K[0][i] + c6 * K[1][i] + d6 * K[2][i] + e6 * K[3][i] + f6 * K[4][i]) > big) {
                    reserrors.append(DIVERGENCE_STR);
                    break;
                }
            }
            for (i=0; i<n_vars; i++) {
                K[5][i] = h * f[i](T[j] + a6 * h, ys[5]);
            }
            /*k5 = h * f(T[j] + a5 * h, y5, x5);
            l5 = h * g(T[j] + a5 * h, y5, x5);
            y6 = Y[j] + b6 * k1 + c6 * k2 + d6 * k3 + e6 * k4 + f6 * k5;
            x6 = X[j] + b6 * l1 + c6 * l2 + d6 * l3 + e6 * l4 + f6 * l5;
            k6 = h * f(T[j] + a6 * h, y5, x5);
            l6 = h * g(T[j] + a6 * h, y5, x5);*/

            for (i=0; i<n_vars; i++) {
                errs[i] = abs(r1 * K[0][i] + r3 * K[2][i] + r4 * K[3][i] + r5 * K[4][i] + r6 * K[5][i]);
                ynews[i] = Y[i][j] + n1 * K[0][i] + n3 * K[2][i] + n4 * K[3][i] + n5 * K[4][i];
            }
            /*err_y = abs(r1*k1 + r3*k3 + r4*k4 + r5*k5 + r6*k6);
            err_x = abs(r1*l1 + r3*l3 + r4*l4 + r5*l5 + r6*l6);
            ynew = Y[j] + n1 * k1 + n3 * k3 + n4 * k4 + n5 * k5;
            xnew = X[j] + n1 * l1 + n3 * l3 + n4 * l4 + n5 * l5;*/

            // Tolerance error and step size:
            step++;
            // If the new point is precise enough, we add it
            successful_iteration = true;
            for (i=0; i<n_vars; i++) {
                if (errs[i] > absolute_tolerances[i]) {
                    successful_iteration = false;
                    break;
                }
            }
            if (successful_iteration) {
                for (i=0; i<n_vars; i++) {
                    Y[i][j +1] = ynews[i];
                }    
                T[j +1] = T[j] + h;
                j++;
            }

            // Compute parameters for the next iteration
            for (i=0; i<n_vars; i++) {
                if (errs[i] == 0.0) {
                    ss[i] = 1.1;
                } else {
                    ss[i] = 0.84 * pow(absolute_tolerances[i] * h / errs[i], 0.25);
                    if (ss[i]<1.0) {
                        hs[i] = min(h*ss[i], h / 2);
                    } else {
                        hs[i] = max(h*ss[i], 2 * h);
                    }
                }
            }
            hmin = hs[0];
            for (i=1; i<n_vars; i++) {
                hmin = (hs[i] < hmin) ? hs[i] : hmin;
            }
            h = hmin;

            // Check if can continue
            if (b - final_point_t_margin < T[j]) {
                continue_flag = false; // Evrything OK
            } 
            for (i=0; i<n_vars; i++) {
                if (abs(Y[i][j]) > big) {
                    reserrors.append(DIVERGENCE_STR);
                    break;
                } 
            }
            if (step > max_steps) {
                reserrors.append("Error: maximum number of steps reached. Computation stopped.\n");
                break;
            }
        }

        resN = j+1;
        rest = new REAL[j+1];
        for (i=0; i<n_vars; i++) {
            res_vec[i] = new REAL[j+1];
        }
        for (UINTG l=0; l<j+1; l++) {
            rest[l] = T[l];
            for (i=0; i<n_vars; i++) {
                res_vec[i][l] = Y[i][l];
            }
        }

        // Cleanup and return
        delete[] T;
        for (i=0; i<n_vars; i++) {
            delete[] Y[i];
        }
        delete[] Y;
        
        cResult_vector res = {.errors= nullptr, .y= res_vec, .t= rest, .n_vars = n_vars, .N= resN};
        res.errors = new char[reserrors.length()];
        memcpy(res.errors, reserrors.c_str(), reserrors.length());
        res.errors_len = reserrors.length();
        return res;

        // We will need some delete[] res.t cleanup;

    }
    

    // SOME EXAMPLE FUNCTIONS
#ifdef __ALLOW_EXAMPLES_RK4V2_
    /* y' = 30 - 5y */
    REAL test_f(REAL t, REAL y) {
        cout << "Calling test_fc";
        return 30.0 - 5*y;
    }

    /* test 2: y'' + y = 0 */
    REAL test2_f(REAL t, REAL y, REAL x) {
        return x;
    }

    REAL test2_g(REAL t, REAL y, REAL x) {
        return -y;
    }

    /* same as test 2, but with the vector routine*/
    REAL testv_f(REAL t, REAL* ys) {
        return ys[1];
    }

    REAL testv_g(REAL t, REAL* ys) {
        return -ys[0];
    }
#endif
}

#ifdef __ALLOW_EXAMPLES_RK4V2_
void main_old () {
    REAL ya = 1, xa = 0;
    REAL a=0, b=2 * 3.1415926535897932;

    cout << "Numeric limits (epsilon): " << numeric_limits<REAL>::epsilon() << endl;

    cResult_2o res = rkf4_2o(&test2_f, &test2_g, a, b, ya, xa, 100, 15000, 1e-6, 1e-6);
    cout << res.N << ", (" << res.t[res.N-1] << ", " << res.y[res.N-1] << ", " << res.x[res.N-1] << ").  " << endl << res.errors;

    // Explore potential bug: extra iterations (?)
    REAL (*fptrs[2])(REAL, REAL*);
    fptrs[0] = &testv_f;
    fptrs[1] = &testv_g;
    REAL yas[2], abstols[2];
    yas[0] = 1;
    yas[1] = 0;
    abstols[0] = abstols[1] = 1e-6;
    cResult_vector resv = rkf4_vector(fptrs, a, b, 2, yas, 100, 15000, abstols);
    cout << resv.N << ", (" << resv.t[resv.N-1] << ", " << resv.y[0][resv.N-1] << ", " << resv.y[1][resv.N-1] << ").  " << endl << resv.errors;
    clear_result_vector(&resv);
    clear_result_2o(&res);
}
#endif



// EXAMPLE OF A COMPLETE PROGRAM, THAT SOLVES A COMPLICATED 2ND ORDER ODE
// AND SAVES THE RESULTS IN A CSV FILE
#ifdef __ALLOW_EXAMPLES_RK4V2_


/* Classical ODE */
REAL cboulware_f(REAL t, REAL y, REAL x) {
    return x;
}

REAL cboulware_g(REAL t, REAL y, REAL x) {
    return 2 * x * (1 - x);
}

/* semiclassical ODE with eps = 0 */
REAL cboulware_g2(REAL t, REAL y, REAL x) {
    REAL m = 1;
    REAL beta = -1;
    return 2 * (4 - exp(2*t)*beta*(-1 + x)) * (4 + exp(2*t)*beta*(-1 + x) - 4*x) * x / (16 - 8*beta*exp(2*t) + beta*beta*exp(4*t));
}

/* semiclassical ODE with eps */
REAL cboulware_g3(REAL t, REAL y, REAL x) {
    REAL m = 1;
    REAL eps = 0.0;
    REAL beta = -1;
    return 2 * x * (4 - exp(2*t)*beta*(-1 + eps + x)) * (4 - 4*x + exp(2*t)*beta*(-1 + eps + (1 + eps)*x)) / (16 - 8*beta*(1 + eps)*exp(2*t) + beta*beta*exp(4*t)*(eps - 1)*(eps - 1));
}
/* semiclassical ODE with eps */
REAL cboulware_g4(REAL t, REAL y, REAL x) {
    REAL m = 1;
    REAL eps = 0.5;
    REAL beta = -1;
    return 2 * x * (4 - exp(2*t)*beta*(-1 + eps + x)) * (4 - 4*x + exp(2*t)*beta*(-1 + eps + (1 + eps)*x)) / (16 - 8*beta*(1 + eps)*exp(2*t) + beta*beta*exp(4*t)*(eps - 1)*(eps - 1));
}
/* semiclassical ODE with eps */
REAL cboulware_g5(REAL t, REAL y, REAL x) {
    REAL m = 1;
    REAL eps = 1;
    REAL beta = -1;
    return 2 * x * (4 - exp(2*t)*beta*(-1 + eps + x)) * (4 - 4*x + exp(2*t)*beta*(-1 + eps + (1 + eps)*x)) / (16 - 8*beta*(1 + eps)*exp(2*t) + beta*beta*exp(4*t)*(eps - 1)*(eps - 1));
}

void save_res(const cResult_2o& res, std::string file, REAL m, REAL beta, REAL epsilon) {
    // Save to CSV
    fstream fout;
    fout.open(file, ios::out);
    fout << m << "," << beta << "," << epsilon << endl;
    for(UINTG i=0; i < res.N-1; i++) {
        fout << res.t[i] << ",";
    }
    fout << res.t[res.N-1] << endl;
    for(UINTG i=0; i < res.N-1; i++) {
        fout << res.y[i] << ",";
    }
    fout << res.y[res.N-1] << endl;
    for(UINTG i=0; i < res.N-1; i++) {
        fout << res.x[i] << ",";
    }
    fout << res.x[res.N-1] << endl;
    fout.close();
}

int main() {

    // classical initial conditions
    REAL m = 1;
    REAL a=-10, b=10;
    REAL ya = a + 0.5 * log(exp(-2*a) - m);
    REAL xa = 1 - 1 / (1 - m * exp( 2 * a)); 

    cout << "Numeric limits (epsilon): " << numeric_limits<REAL>::epsilon() << endl;

    cResult_2o res = rkf4_2o(&cboulware_f, &cboulware_g, a, b, ya, xa, 100, 15000, 1e-12, 1e-12);
    cout << "ERRORS: " << res.errors << endl;
    save_res(res, "classical.csv", m, 0, 0);
    clear_result_2o(&res);

    cResult_2o res2 = rkf4_2o(&cboulware_f, &cboulware_g3, a, b, ya, xa, 10000, 15000, 1e-14, 1e-14);
    cout << "ERRORS: " << res2.errors << endl;
    save_res(res2, "sclassical.csv", m, -1, 0);
    clear_result_2o(&res2);

    res2 = rkf4_2o(&cboulware_f, &cboulware_g4, a, b, ya, xa, 10000, 15000, 1e-14, 1e-14);
    cout << "ERRORS: " << res2.errors << endl;
    save_res(res2, "sclassical1.csv", m, -1, 0.5);
    clear_result_2o(&res2);

    res2 = rkf4_2o(&cboulware_f, &cboulware_g5, a, b, ya, xa, 10000, 15000, 1e-14, 1e-14);
    cout << "ERRORS: " << res2.errors << endl;
    save_res(res2, "sclassical2.csv", m, -1, 1);
    clear_result_2o(&res2);
}

#endif
