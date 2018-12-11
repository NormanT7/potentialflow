import numpy as np


def cosine_spacing(start, stop, num=50, offset=0):
    # calculates the cosine spacing
    index = np.linspace(0., 1., num)
    spacing = .5*(1.-np.cos(np.pi*(index-offset)))

    points = start+spacing*(stop-start)

    return points


def naca4digit(number, n_points, is_cosine_spaced=True, thickness_type="closed_te"):

    y_mc, x_mc, t_m = _parse4digit(number)

    x = _get_xpoints(n_points, is_cosine_spaced)

    # get camber line
    aft = x >= x_mc
    c = 1.*aft
    sign = np.power(-1, c)
    y_c = np.zeros(len(x))
    dy_c = np.zeros(len(x))
    y_c = y_mc*(2.*((c+sign*x)/(c+sign*x_mc))-((c+sign*x)/(c+sign*x_mc))**2)
    dy_c = 2.*y_mc*((sign/(c+sign*x_mc))-sign*((c+sign*x)/(c+sign*x_mc)**2))

    # get thickness
    t = _get_thickness(x, t_m, thickness_type)

    # get surface points
    xy_points = calc_surface(x, y_c, dy_c, t, n_points)


    return xy_points


def uniformload(CL, thick, n_points, is_cosine_spaced=True,
                thickness_type="closed_te"):

    x = _get_xpoints(n_points, is_cosine_spaced)

    # get camber line
    y_c = np.zeros(len(x))
    dy_c = np.zeros(len(x))
    xm = x[1:-1]
    y_c[1:-1] = -(CL/(4.*np.pi))*((1-xm)*np.log(1-xm)+xm*np.log(xm))
    dy_c[1:-1] = -(CL/(4.*np.pi))*(np.log(xm)-np.log(1-xm))
    dy_c[0] = -(CL/(4.*np.pi))*(np.log(1e-16)-np.log(1-1e-16))
    dy_c[-1] = -(CL/(4.*np.pi))*(np.log(1-1e-16)-np.log(1e-16))

    # get thickness
    t = _get_thickness(x, thick, thickness_type)

    # get surface points
    xy_points = calc_surface(x, y_c, dy_c, t, n_points)


    return xy_points


def _parse4digit(number):
    y_mc = float(number[0])/100.
    x_mc = float(number[1])/10.
    t_m = float(number[2:])/100.

    return y_mc, x_mc, t_m


def _get_xpoints(n_points, is_cosine_spaced):
    if is_cosine_spaced:
        spacing = cosine_spacing
    else:
        spacing = np.linspace

    if n_points % 2 == 1:
        n_x = n_points//2+1
        x = spacing(0., 1., n_x)
    else:
        n_x = n_points//2
        x = spacing(0., 1., 2*n_x)[1::2]

    return x


def _get_thickness(x, t_m, thickness_type):
    if thickness_type == "open_te":
        t = t_m*(2.969*np.sqrt(x)-1.26*x-3.516*x*x+2.843*x*x*x-1.015*x*x*x*x)
    elif thickness_type == "closed_te":
        t = t_m*(2.969*np.sqrt(x)-1.26*x-3.523*x*x+2.836*x*x*x-1.022*x*x*x*x)
    else:
        raise RuntimeError("thickness input not recognized")

    return t


def calc_surface(x, y_c, dy_c, t, n_points):
    t_mod = t/(2.*np.power(1+dy_c*dy_c, 0.5))

    x_u = x-t_mod*dy_c
    y_u = y_c+t_mod
    x_l = x+t_mod*dy_c
    y_l = y_c-t_mod

    xy_u = np.array([x_u, y_u]).T
    if n_points % 2 == 1:
        xy_l = (np.array([x_l, y_l]).T)[1:]
    else:
        xy_l = (np.array([x_l, y_l]).T)

    xy_points = np.concatenate((np.flipud(xy_l), xy_u))

    return xy_points
