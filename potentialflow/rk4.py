"""Implements a method for fourth-order Runge-Kutta integration"""


import numpy as np


class RK4:
    """Contains the functionality for integrating a system of first-order
    linear differential equations using rk4.

    """
    def __init__(self, dy_dx, y_corrections=None):
        """Initializes the RK4 class with the equations that represent
        the system to be integrated.

        Parameters
        ----------
        dy_dx : function
            A function that implements the set of equations to be integrated.
            This functions should accept two parameters, the independent
            variable and a dependent variable or array of dependent variables.
            This function must then return the resulting value(s) that
            correspond to the y_dot equation(s). If the y_dot function is
            properly implemented than this routine can properly handle systems
            of equations in addition to a single equation.

        y_corrections : function
            An optional function for providing any corrections to the dependent
            variables between time steps. For example, renormalizing the
            quaternion vectors in a quaternion formulation.

        """
        self._dy_dx = dy_dx
        self._y_corrections = y_corrections

    def step_forward(self, x0, y0, dx):
        """Performs the RK4 integration over the range specified in n_steps.

        Parameters
        ----------
        x0 : float
            The initial value for the dependent variable.
        y0 : float or numpy array
            The initial condition(s) of the independent variable(s).

        Returns
        -------
        results : tuple
            A tuple of the dependent and independent variables at the next
            step.

        """
        x_new, y_new = self._rk4(x0, y0, dx)
        if self._y_corrections:
            self._y_corrections(y_new)

        return x_new, y_new

    def integrate_range(self, x0, y0, x_range, n_steps):
        """Performs the RK4 integration over the range specified in n_steps.

        Parameters
        ----------
        x0 : float
            The initial value for the dependent variable.
        y0 : float or numpy array
            The initial condition(s) of the independent variable(s).
        x_range : float
            The range over which the functions are integrated.
        n_steps : int
            The number of steps to be taken in integrations.

        Returns
        -------
        results : tuple
            A tuple of numpy arrays that contains the values of the dependent
            and independent variables at each step of the integration
            respectively.

        """
        x = np.zeros(n_steps+1)
        y = np.zeros((n_steps+1, len(y0)))
        x[0] = x0
        y[0] = y0

        dx = x_range/n_steps

        for i in np.arange(1, n_steps+1):
            x[i] = x[i-1] + dx
            y[i] = self._rk4(x[i-1], y[i-1], dx)
            if self._y_corrections:
                self._y_corrections(y[i])

        return x, y

    def _rk4(self, x0, y0, dx):
        """Integrates the given y_dot function to find the value(s) for
        y at the next step.

        Parameters
        ----------
        x0 : float or numpy array of floats
            The values(s) for the independent variable(s) at the current time
            step.
        y0 : float or numpy array of floats
            The value(s) for y at current time step, y(x0).
        dx : float
            Step size.

        Returns
        -------
        y : float or numpy array of floats
            The value(s) of the dependent variable(s) at the next step.

        """

        k1 = self._dy_dx(x0, y0)
        k2 = self._dy_dx(x0+dx/2., y0+dx*k1/2.)
        k3 = self._dy_dx(x0+dx/2., y0+dx*k2/2.)
        k4 = self._dy_dx(x0+dx, y0+dx*k3)

        y = y0 + dx*(k1+2.*k2+2.*k3+k4)/6.
        x = x0 + dx

        return x, y
