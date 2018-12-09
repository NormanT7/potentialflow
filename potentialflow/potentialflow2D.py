import numpy as np
import rk4
import matplotlib.pyplot as plt
import scipy.optimize as opt


def plot(solution, n_lines=10):
    # plot geometry
    s = np.linspace(0, 1, 100)
    x, y = solution.geometry(s)
    plt.plot(x, y, 'k', linewidth=1.75)
    # x_zeta, y_zeta = solution.zeta_geometry(s)
    # plt.plot(x_zeta, y_zeta, '--k', linewidth=1.)
    # z_sng, zeta_sng = solution.singularities()
    # plt.scatter(z_sng[0], z_sng[1], marker='.', color='k',
    #             label='singularities, z-plane')
    # plt.scatter(zeta_sng[0], zeta_sng[1], marker='.',
    #             facecolors='none', edgecolors='k',
    #             label=r'singularities, $\zeta$-plane')

    # find forward and rear stagnation points
    xy0_f = solution.geometry(0.5)
    xy_f, xy_r = find_stagnationpoint2(solution, xy0_f)
    x_f, y_f = xy_f
    x_r, y_r = xy_r
    # print(x_f, y_f)
    # # plt.scatter(x_f, y_f)
    # xy0_r = solution.geometry(0.)
    # x_r, y_r = find_stagnationpoint2(solution, xy0_r)
    # print(x_r, y_r)
    # plt.scatter(x_f, y_f)
    # plt.scatter(x_r, y_r)
    # plt.show()
    # exit()
    # plt.scatter(x_r, y_r)
    y_scale = 2.
    x_scale = 2.

    # plot streamlines
    dx = (np.max(x)-np.min(x))
    dy = (np.max(y)-np.min(y))
    ssf_x, ssf_y = streamline(solution.velocity, x_f-1e-4, y_f, -x_scale*dx, -.01)
    ssr_x, ssr_y = streamline(solution.velocity, x_r+1e-4, y_r, x_scale*dx, .01)
    plt.plot(ssf_x, ssf_y, 'k', linewidth=0.75)
    plt.plot(ssr_x, ssr_y, 'k', linewidth=0.75)
    y0_s = ssf_y[-1]
    # y0_s = 0.
    y_top = np.linspace(y0_s, y0_s+2.*y_scale*dy, n_lines//2)
    y_bot = np.linspace(y0_s-2.*y_scale*dy, y0_s, n_lines//2)
    y_initial = np.concatenate((y_bot[:-1], y_top[1:]))
    for y0 in y_initial:
        s_x, s_y = streamline(solution.velocity, -x_scale*dx, y0, x_scale*dx, .01)
        plt.plot(s_x, s_y, 'k', linewidth=0.75)

    axes = plt.gca()
    axes.set_xlim(-x_scale*dx, x_scale*dx)
    axes.set_ylim(-y_scale*dy, y_scale*dy)
    # plt.gcf().set_size_inches((6, 6))
    plt.xlabel('x/2R')
    plt.ylabel('y/2R')
    axes.set_aspect('equal', adjustable='box')
    plt.legend(loc='upper right')
    plt.show()
    plt.close()


def find_stagnationpoint(velocity, xy0):
    def v_mag(xy):
        x, y = xy
        u, v = velocity(x, y)
        return np.sqrt(u*u+v*v)

    results = opt.minimize(v_mag, xy0)
    x_stag, y_stag = results.x

    return x_stag, y_stag


def find_stagnationpoint2(solution, xy0):
    s = np.linspace(0, 1, 100)
    x, y = solution.geometry(s)
    u, v = solution.velocity(x, y)
    v_mag = np.sqrt(u*u+v*v)

    N = len(x)
    v_mag_front = v_mag[N//4:3*N//4]
    s_front = s[N//4:3*N//4]
    i_m_front = np.argmin(v_mag_front)
    s_m_front = s_front[i_m_front]
    xy_front = solution.geometry(s_m_front)

    v_mag_back = np.concatenate((v_mag[:N//4], v_mag[3*N//4:]))
    s_back = np.concatenate((s[:N//4], s[3*N//4:]))
    i_m_back = np.argmin(v_mag_back)
    s_m_back = s_back[i_m_back]
    # plt.show()
    # plt.plot(s_back, v_mag_back)
    # plt.scatter(s_m_back, v_mag_back[i_m_back])
    # plt.show()
    xy_back = solution.geometry(s_m_back)

    return xy_front, xy_back


class Cylinder():
    def __init__(self, R=1, eps=0.5, AOA=0., circulation=0., loc=(0., 0.)):
        self._radius = R
        self._epsilon = eps*2.*R
        self._v_inf = 1.
        self._AOA = AOA*np.pi/180.
        self._gamma = circulation*2*R*self._v_inf
        self._z0 = complex(loc[0]*2.*R, loc[1]*2.*R)

    def geometry(self, s):
        theta = s*2.*np.pi
        # z = self._z0+np.exp(1j*theta)
        R = self._radius
        eps = self._epsilon
        z = R*np.exp(1j*theta) + (R-eps)**2/(R*np.exp(1j*theta))
        x = np.real(z)
        y = np.imag(z)

        return x, y

    def velocity(self, x, y):
        R = self._radius
        eps = self._epsilon
        v_inf = self._v_inf
        alpha = self._AOA
        gamma = self._gamma
        z0 = self._z0
        z = x + y*1j
        if x < 0:
            zeta = (z-np.sqrt(z*z-4.*(R-eps)**2))/2.
        else:
            zeta = (z+np.sqrt(z*z-4.*(R-eps)**2))/2.

        w = v_inf*(np.exp(-1j*alpha)+1j*gamma/(2*np.pi*v_inf*(zeta)) -
                   R*R*np.exp(1j*alpha)/np.power(zeta, 2.))/(1.-(R-eps)**2/zeta**2)

        v_x = np.real(w)
        v_y = -np.imag(w)
        # theta = np.arctan2(y, x)
        # r = np.sqrt(x*x+y*y)
        # c_t = np.cos(theta)
        # s_t = np.sin(theta)
        # c_a = np.cos(alpha)
        # s_a = np.sin(alpha)

        # v_r = v_inf*(1.-R*R/(r*r))*np.cos(theta-alpha)
        # v_t = -v_inf*(1.+R*R/(r*r))*np.sin(theta-alpha)-gamma/(2.*np.pi*r)

        # v_x = v_r*np.cos(theta)-v_t*np.sin(theta)
        # v_y = v_r*np.sin(theta)+v_t*np.cos(theta)

        return v_x, v_y


class JoukowskiCylinder():
    def __init__(self, R=1, eps=0.5, AOA=0., circulation=0., loc=(0., 0.)):
        self._radius = R
        self._epsilon = eps*2.*R
        self._v_inf = 1.
        self._AOA = AOA*np.pi/180.
        self._gamma = circulation*2*R*self._v_inf
        self._z0 = complex(loc[0]*2.*R, loc[1]*2.*R)

    def geometry(self, s):
        theta = s*2.*np.pi
        z0 = self._z0
        R = self._radius*(1.+1e-6)
        eps = self._epsilon
        z = R*np.exp(1j*theta)+z0+(R-eps)**2/(R*np.exp(1j*theta)+z0)
        x = np.real(z)
        y = np.imag(z)

        return x, y

    def zeta_geometry(self, s):
        theta = s*2.*np.pi
        z0 = self._z0
        R = self._radius
        zeta = R*np.exp(1j*theta)+z0
        x = np.real(zeta)
        y = np.imag(zeta)

        return x, y

    def singularities(self):
        R = self._radius
        eps = self._epsilon
        zeta_sng = np.array([R-eps, eps-R])
        z_sng = zeta_sng+np.power(R-eps, 2)/zeta_sng
        xy_zeta_sng = (np.real(zeta_sng), np.imag(zeta_sng))
        xy_z_sng = (np.real(z_sng), np.imag(z_sng))

        return xy_z_sng, xy_zeta_sng


    def velocity(self, x, y):
        R = self._radius
        eps = self._epsilon
        v_inf = self._v_inf
        alpha = self._AOA
        gamma = self._gamma
        z0 = self._z0
        z = x + y*1j

        zeta = (z+np.sqrt(z*z-4.*(R-eps)**2))/2.
        zeta2 = (z-np.sqrt(z*z-4.*(R-eps)**2))/2.
        try:
            if np.abs(zeta2-z0) > np.abs(zeta-z0):
                zeta = zeta2
        except ValueError:
            i_bigger = np.abs(zeta2-z0) > np.abs(zeta-z0)
            zeta[i_bigger] = zeta2[i_bigger]

        w = v_inf*(np.exp(-1j*alpha)+1j*gamma/(2*np.pi*v_inf*(zeta-z0)) -
                   R*R*np.exp(1j*alpha)/np.power(zeta-z0, 2.))/(1.-(R-eps)**2/zeta**2)

        v_x = np.real(w)
        v_y = -np.imag(w)
        # theta = np.arctan2(y, x)
        # r = np.sqrt(x*x+y*y)
        # c_t = np.cos(theta)
        # s_t = np.sin(theta)
        # c_a = np.cos(alpha)
        # s_a = np.sin(alpha)

        # v_r = v_inf*(1.-R*R/(r*r))*np.cos(theta-alpha)
        # v_t = -v_inf*(1.+R*R/(r*r))*np.sin(theta-alpha)-gamma/(2.*np.pi*r)

        # v_x = v_r*np.cos(theta)-v_t*np.sin(theta)
        # v_y = v_r*np.sin(theta)+v_t*np.cos(theta)

        return v_x, v_y


class JoukowskiAirfoil:
    def __init__(self, cl_d, tau_d, alpha_d=0.):
        x0, y0 = self._calc_zeta0(cl_d, tau_d, alpha_d)
        self._x0 = x0
        self._y0 = y0

    def _calc_zeta0(self, cl, tau, alpha):
        x0 = -4.*tau/(3.*np.sqrt(3.))
        a = alpha*np.pi/180.
        y0 = (cl/(2.*np.pi*(1.-x0))-np.sin(a))/np.cos(a)

        return x0, y0

    def performance(self, alpha=0):
        a = alpha*np.pi/180.
        x0 = self._x0
        y0 = self._y0
        st = np.sqrt(1-y0**2)
        cl = 2.*np.pi*(np.sin(a)+y0*np.cos(a)/st)/(1.+x0/(st-x0))
        x_te = 2.*(st+x0)
        x_le = -2.*(1-y0**2+x0**2)/(st-x0)
        x_4 = x_le+(x_te-x_le)/4.
        y_4 = 0.
        Cm4 = (np.pi/4.*(1-y0**2-x0**2)/(1-y0**2)*np.sin(2.*a) +
               cl/4.*((x_4-x0)*np.cos(a)+(y_4-y0)*np.sin(a))/(1-y0**2)*(st-x0))

        return cl, Cm4

    def geometry(self, s):
        x0 = self._x0
        y0 = self._y0
        theta_te = np.arctan(-y0/np.sqrt(1-y0**2))
        theta = s*2.*np.pi+theta_te
        z0 = x0+y0*1j
        R = 1.  # self._radius*(1.+1e-6)
        eps = 1-R*np.sqrt(R**2-y0**2)-x0
        z = R*np.exp(1j*theta)+z0+(R-eps)**2/(R*np.exp(1j*theta)+z0)
        x_te = 2.*(np.sqrt(1-y0**2)+x0)
        x_le = -2.*(1-y0**2+x0**2)/(np.sqrt(1-y0**2)-x0)
        chord = x_te - x_le
        x = (np.real(z)-x_le)/chord
        y = np.imag(z)/chord

        return x, y


def streamline(v_function, x_start, y_start, x_lim, ds):
    def dpds(s, xy):
        x, y = xy
        v_x, v_y = v_function(x, y)
        v_mag = np.sqrt(v_x*v_x + v_y*v_y)
        dxds = v_x/v_mag
        dyds = v_y/v_mag

        return np.array([dxds, dyds])

    streamline_rk4 = rk4.RK4(dpds)
    x_values = [x_start]
    y_values = [y_start]
    s = 0.

    while np.sign(ds)*(x_lim-x_values[-1]) > 0.:
        xy = np.array([x_values[-1], y_values[-1]])
        s, xy_next = streamline_rk4.step_forward(s, xy, ds)
        x_values.append(xy_next[0])
        y_values.append(xy_next[1])

    return x_values, y_values


def plot_quiver(solution):
    x_range = np.linspace(-4., 4., 20)
    y_range = np.linspace(-4., 4., 20)

    points_x, points_y = np.meshgrid(x_range, y_range)
    points_u, points_v = solution.velocity(points_x, points_y)

    plt.quiver(points_x, points_y, points_u, points_v)
    plt.show()
