# rkf4
Numerical solver for systems of ODEs. Method: Runge–Kutta–Fehlberg of order 4.5

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
