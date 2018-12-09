import numpy as np


def cosine_spacing(start, stop, num=50, offset=0):
    # calculates the cosine spacing
    index = np.linspace(0., 1., num)
    spacing = .5*(1.-np.cos(np.pi*(index-offset)))

    points = start+spacing*(stop-start)

    return points


def naca4digit(number, n_points, cos_spacing=True, thickness="closed_te",
               camber="normal"):

    y_mc, x_mc, t_m = _parse4digit(number)

    if cos_spacing:
        spacing = cosine_spacing
    else:
        spacing = np.linspace

    if n_points % 2 == 1:
        n_x = n_points//2+1
        x = spacing(0., 1., n_x)
    else:
        n_x = n_points//2
        x = spacing(0., 1., 2*n_x)[1::2]

    # get camber line
    aft = x >= x_mc
    c = 1.*aft
    sign = np.power(-1, c)
    y_c = np.zeros(len(x))
    dy_c = np.zeros(len(x))
    if camber == "normal":
        y_c = y_mc*(2.*((c+sign*x)/(c+sign*x_mc))-((c+sign*x)/(c+sign*x_mc))**2)
        dy_c = 2.*y_mc*((sign/(c+sign*x_mc))-sign*((c+sign*x)/(c+sign*x_mc)**2))
    elif camber == "uniform_load":
        xm = x[1:-1]
        y_c[1:-1] = -(y_mc/np.log(2.))*((1-xm)*np.log(1-xm)+xm*np.log(xm))
        dy_c[1:-1] = -(y_mc/np.log(2.))*(np.log(xm)-np.log(1-xm))
    else:
        raise RuntimeError("camber input not recognized")

    # get thickness
    # t = t_m*(2.98*np.sqrt(x)-1.32*x-3.286*x*x+2.441*x*x*x-0.815*x*x*x*x)
    if thickness == "open_te":
        t = t_m*(2.969*np.sqrt(x)-1.26*x-3.516*x*x+2.843*x*x*x-1.015*x*x*x*x)
    elif thickness == "closed_te":
        t = t_m*(2.969*np.sqrt(x)-1.26*x-3.523*x*x+2.836*x*x*x-1.022*x*x*x*x)
    else:
        raise RuntimeError("thickness input not recognized")

    # get surface points
    x_u, y_u, x_l, y_l = calc_surface(x, y_c, dy_c, t)

    xy_u = np.array([x_u, y_u]).T
    if n_points % 2 == 1:
        xy_l = (np.array([x_l, y_l]).T)[1:]
    else:
        xy_l = (np.array([x_l, y_l]).T)

    xy_points = np.concatenate((np.flipud(xy_l), xy_u))

    return xy_points


def _parse4digit(number):
    y_mc = float(number[0])/100.
    x_mc = float(number[1])/10.
    t_m = float(number[2:])/100.

    return y_mc, x_mc, t_m


def calc_surface(x, y_c, dy_c, t):
    t_mod = t/(2.*np.power(1+dy_c*dy_c, 0.5))

    x_u = x-t_mod*dy_c
    y_u = y_c+t_mod
    x_l = x+t_mod*dy_c
    y_l = y_c-t_mod

    return x_u, y_u, x_l, y_l
