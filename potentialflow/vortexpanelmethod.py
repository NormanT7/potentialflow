import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt


class VPM:
    """
    """
    def __init__(self, geometry):
        self._nodes = geometry
        self._influence_matrix = self._calc_influences()
        self._lu_piv = scipy.linalg.lu_factor(self._influence_matrix)
        self._gamma = None
        self._results = None

    def solve(self, aoa=[0.], v_mag=1.):
        self._alpha = np.array(aoa)*np.pi/180.
        self._v_mag = v_mag
        cl = []
        cm = []
        cm4 = []

        for i, a in enumerate(self._alpha):
            rhs = self._calc_rhs(a, v_mag)
            gamma = scipy.linalg.lu_solve(self._lu_piv, rhs)
            cl_a, cm_a, cm4_a = self._calc_results(gamma, a, v_mag)
            self._gamma = gamma
            cl.append(cl_a)
            cm.append(cm_a)
            cm4.append(cm4_a)

        self._results = (cl, cm, cm4)

        return self._results

    def velocity(self, x, y):
        gamma = self._gamma
        try:
            x_p = x[..., None]
            y_p = y[..., None]
        except TypeError:
            x_p = x
            y_p = y
        p11, p12, p21, p22 = self._calc_pmatrix(x_p, y_p)

        vx_induced = np.sum(p11*gamma[:-1]+p12*gamma[1:], axis=-1)
        vy_induced = np.sum(p21*gamma[:-1]+p22*gamma[1:], axis=-1)

        try:
            aoa = self._alpha[-1]
        except TypeError:
            aoa = self._alpha

        vx = self._v_mag*np.cos(aoa)+vx_induced
        vy = self._v_mag*np.sin(aoa)+vy_induced

        if vx.size > 1:
            return vx, vy
        else:
            return vx[0], vy[0]

    def geometry(self, s):
        # TODO: Fix this. It doesn't do what I thought it did.
        # Look at airfoil with small number of points
        x, y = self._nodes.T
        lengths = self._lengths
        s_n = np.concatenate((np.array([0.]), np.cumsum(lengths)))
        s_n /= s_n[-1]

        x_g = np.interp(s, s_n, x)
        y_g = np.interp(s, s_n, y)

        return x_g, y_g

    def _calc_influences(self):
        x, y = self._nodes.T
        x1 = x[:-1]  # first x points of panels
        x2 = x[1:]  # second x points of panels
        y1 = y[:-1]  # first y points of panels
        y2 = y[1:]  # second y points of panels

        N = len(x)//2
        x_cp = (x1+x2)/2.
        y_cp = (y1+y2)/2.

        dx = x2-x1
        dy = y2-y1
        lengths = np.sqrt(dx*dx+dy*dy)
        self._dx = dx
        self._dy = dy
        self._lengths = lengths

        p11, p12, p21, p22 = self._calc_pmatrix(x_cp[:, None], y_cp[:, None])

        # build influence matrix
        a_ji = (dx/lengths)[:, None]*p21-(dy/lengths)[:, None]*p11
        b_ji = (dx/lengths)[:, None]*p22-(dy/lengths)[:, None]*p12

        influence_matrix = np.zeros((len(x), len(x)))
        influence_matrix[:-1, :-1] = a_ji
        influence_matrix[:-1, 1:] += b_ji
        influence_matrix[-1, 0] = 1.
        influence_matrix[-1, -1] = 1.
        # print(influence_matrix)

        return influence_matrix

    def _calc_pmatrix(self, x_p, y_p):
        x, y = self._nodes.T
        x1 = x[:-1]  # first x points of panels
        y1 = y[:-1]  # first y points of panels
        dx = self._dx
        dy = self._dy
        lengths = self._lengths

        # each point in terms of every panel coordinate system
        # i.e., each row is a point in terms of every other system
        dx_p = x_p-x1[None, :]
        dy_p = y_p-y1[None, :]

        xi_ji = (1./lengths[None, :])*(dx[None, :]*dx_p+dy[None, :]*dy_p)
        eta_ji = (1./lengths[None, :])*(-dy[None, :]*dx_p+dx[None, :]*dy_p)

        # calculate P matrix
        xi_sq = xi_ji*xi_ji
        eta_sq = eta_ji*eta_ji
        phi = np.arctan2(eta_ji*lengths[None, :],
                         eta_sq+xi_sq-xi_ji*lengths[None, :])
        psi = 0.5*np.log((xi_sq+eta_sq)/((xi_ji-lengths[None, :])**2+eta_sq))

        s11 = (lengths[None, :]-xi_ji)*phi+eta_ji*psi
        s12 = xi_ji*phi-eta_ji*psi
        s21 = eta_ji*phi-(lengths[None, :]-xi_ji)*psi-lengths[None, :]
        s22 = -eta_ji*phi-xi_ji*psi+lengths[None, :]

        l2pi = 2.*np.pi*lengths[None, :]*lengths[None, :]
        p11 = (dx*s11-dy*s21)/l2pi
        p12 = (dx*s12-dy*s22)/l2pi
        p21 = (dy*s11+dx*s21)/l2pi
        p22 = (dy*s12+dx*s22)/l2pi

        return p11, p12, p21, p22

    def _calc_rhs(self, aoa, v_mag):
        lengths = self._lengths
        dx = self._dx
        dy = self._dy

        v_normal = v_mag*(dx*np.sin(aoa)/lengths-dy*np.cos(aoa)/lengths)

        rhs = np.zeros(len(dx)+1)
        rhs[:-1] = -v_normal

        return rhs

    def _calc_results(self, gamma, aoa, v_mag):
        x, y = self._nodes.T
        x1 = x[:-1]  # first x points of panels
        x2 = x[1:]  # second x points of panels
        y1 = y[:-1]  # first y points of panels
        y2 = y[1:]  # second y points of panels
        lengths = self._lengths
        g1 = gamma[:-1]
        g2 = gamma[1:]

        cl = np.sum(lengths*(g1+g2))/v_mag
        cmi = lengths*((2*x1*g1+x1*g2+x2*g1+2*x2*g2)*np.cos(aoa) +
                       (2*y1*g1+y1*g2+y2*g1+2*y2*g2)*np.sin(aoa))
        cm = -np.sum(cmi)/(3.*v_mag)
        cmc4 = cm+cl*np.cos(aoa)/4.

        return cl, cm, cmc4

    def plot_Cp_surface(self):
        x, y = self._nodes.T
        x1 = x[:-1]  # first x points of panels
        x2 = x[1:]  # second x points of panels
        y1 = y[:-1]  # first y points of panels
        y2 = y[1:]  # second y points of panels
        x_cp = (x1+x2)/2.
        y_cp = (y1+y2)/2.
        dx = x2-x1
        dy = y2-y1
        l_ = self._lengths
        nx = -dy/l_
        ny = dx/l_
        x_p = x_cp+1.e-12*nx
        y_p = y_cp+1.e-12*ny

        vx, vy = self.velocity(x_p, y_p)
        v_mag = np.sqrt(vx*vx+vy*vy)
        cp = 1.-v_mag*v_mag

        plt.plot(x_p, -cp)
        plt.xlabel('x/c')
        plt.ylabel('Pressure Coefficient')
        plt.tight_layout()
        plt.show()
