import math
import numpy as np
from numpy import linalg as la
from scipy.optimize import root


from globals import HRJ, Q, EG_GAAS, M0, ME_QW, MZHH_QW




'''
    written by Krassimir Panajotov in june 2014
    adapted and modified to python by Lucas Laplanche january 2021

    based on
    Lengyel jqe26 p296
    Chow, Koch, Sargent III "Semiconductor Laser Physics" ch.2
'''




def qw_even(ez, vcv, mz, mb, qw_width):
    u = np.sqrt(2*mz*Q*ez)/HRJ
    x = u*qw_width/2

    # floating point rounding error
    if ez>vcv:
        ez = vcv
    fu = np.sqrt(ez)*np.sin(x) -np.sqrt(mb/mz*(vcv -ez))*np.cos(x)


    return fu




def qw_odd(ez, vcv, mz, mb, qw_width):
    u = np.sqrt(2*mz*Q*ez)/HRJ
    x = u*qw_width/2

    # floating point rounding error
    if ez>vcv:
        ez = vcv
    fu = np.sqrt(ez)*np.cos(x) +np.sqrt(mb/mz*(vcv -ez))*np.sin(x)


    return fu




def zbrak(function, xl, vcv, n, mz, mb, qw_width):
    xstep = (vcv -xl)/n
    f0 = function(xl, vcv, mz, mb, qw_width)
    eqw = np.array([])


    for i in range(n):
        x2 = xl +(i +1)*xstep
        f2 = function(x2, vcv, mz, mb, qw_width)

        if f0*f2 < 0:
            xm = x2 -xstep/2
            r = root(function, xm, args=(vcv, mz, mb, qw_width), method='lm').x[0]

            eqw = np.append(eqw, r)

        f0 = f2


    return eqw




def norm_even(k, g, qw_width):
    cc = np.divide( np.cos(k*qw_width/2.), np.exp(-g*qw_width/2.) )
    nrm = qw_width/2. +np.divide( np.sin(k*qw_width)/2., k ) +np.multiply(np.divide(np.square(cc), g), np.exp(-g*qw_width))
    a = 1./np.sqrt(nrm)
    b = np.multiply(cc, a)


    return [a, b]




def norm_odd(k, g, qw_width):
    cc = np.divide( np.cos(k*qw_width/2), np.exp(-g*qw_width/2) )
    nrm = qw_width/2 -np.divide( np.sin(k*qw_width)/2, k ) +np.multiply(np.divide(np.square(cc), g), np.exp(-g*qw_width))
    a = 1./np.sqrt(nrm)
    b = np.multiply(cc, a)


    return [a, b]




def psi_even(kz, gz, az, bz, z, qw_width):
    psi = np.NAN

    if abs(z) <= qw_width/2:
        psi = az*np.cos(kz*z, dtype=np.double)

    if z < -qw_width/2:
        psi = bz*np.exp(gz*z, dtype=np.double)

    if z > qw_width/2:
        psi = bz*np.exp(-gz*z, dtype=np.double)


    return psi




def psi_odd(kz, gz, az, bz, z, qw_width):
    psi = np.NAN

    if abs(z) <= qw_width/2:
      psi = az*np.sin(kz*z, dtype=np.double)

    if z < -qw_width/2:
      psi = bz*np.exp(gz*z, dtype=np.double)

    if z > qw_width/2:
      psi = -bz*np.exp(-gz*z, dtype=np.double)


    return psi




def gamm(el, electric_field, nee, k, g, a, b, qw_width):
    dz = 2e-10
    zz = np.arange(-25e-9, 25e-9 + dz, dz, dtype=np.double)
    nz = np.size(zz)
    gz = np.zeros(nz, dtype=np.double)
    nb = np.size(k)
    gs = np.zeros([nb, nb], dtype=np.double)


    if not el:
        electric_field = -electric_field


    for ik in range(nb):
        for jk in range(nb):
            iz = 0

            for z in zz:
                if ik < nee:
                    psi = psi_even(k[ik], g[ik], a[ik], b[ik], z, qw_width)
                else:
                    psi = psi_odd(k[ik], g[ik], a[ik], b[ik], z, qw_width)

                if jk < nee:
                    psj = psi_even(k[jk], g[jk], a[jk], b[jk], z, qw_width)
                else:
                    psj = psi_odd(k[jk], g[jk], a[jk], b[jk], z, qw_width)

                pgz = psi*electric_field*z*psj
                gz[iz] = psi*electric_field*z*psj
                iz += 1




            gs[ik, jk] = math.fsum(gz)*dz


    return gs




def perturb(el, electric_field, e, n, k, g, a, b, qw_width):
    g = gamm(el, electric_field, n, k, g, a, b, qw_width)
    [row, col] = g.shape
    nb = np.max([row, col])


    for i in range(nb):
        g[i, i] = g[i, i] +e[i]


    [d, v] = la.eig(g.astype(np.single))
    imin = np.where(d==np.amin(d))[0]
    e1 = d[imin]
    v1 = v[:, imin]


    return [v1, e1]




def gaas_sqw_absorption_at_wavelength(x_al, qw_width, electric_field, wavelength):
    ll, alpha, psie, psih, zz = gaas_sqw_absorption(x_al, qw_width, electric_field)
    idx = find_nearest_index(ll, wavelength*1e9/1e3)
    alpha = alpha[idx]


    return alpha




def gaas_sqw_absorption(x_al, qw_width, electric_field):
    # x_al in [1]
    # qw_width in [m]
    # electric field in [V/m]
    eg_cb = EG_GAAS +1.247*x_al
    me_cb = (0.067 +0.083*x_al)*M0
    mzhh_cb = (0.48 +0.31*x_al)*M0

    vc = 0.6*(eg_cb -EG_GAAS)
    vv = 0.4*(eg_cb -EG_GAAS)


    # electron levels
    ee_even = zbrak(qw_even, 1e-9, vc -1e-9, 100, ME_QW, me_cb, qw_width)
    ke_even = np.sqrt(2*ME_QW*Q*ee_even)/HRJ
    ge_even = np.sqrt(2*me_cb*Q*(vc -ee_even))/HRJ
    [ae_even, be_even] = norm_even(ke_even, ge_even, qw_width)

    ee_odd = zbrak(qw_odd, 1e-9, vc -1e-9, 100, ME_QW, me_cb, qw_width)
    ke_odd = np.sqrt(2*ME_QW*Q*ee_odd)/HRJ
    ge_odd = np.sqrt(2*me_cb*Q*(vc -ee_odd))/HRJ
    [ae_odd, be_odd] = norm_odd(ke_odd, ge_odd, qw_width)


    # heavy holes levels
    ehh_even = zbrak(qw_even, 1e-9, vv -1e-9, 1000, MZHH_QW, mzhh_cb, qw_width)
    khh_even = np.sqrt(2*MZHH_QW*Q*ehh_even)/HRJ
    ghh_even = np.sqrt(2*mzhh_cb*Q*(vv -ehh_even))/HRJ
    [ahh_even, bhh_even] = norm_even(khh_even, ghh_even, qw_width)

    ehh_odd = zbrak(qw_odd, 1e-9, vv -1e-9, 1000, MZHH_QW, mzhh_cb, qw_width)
    khh_odd = np.sqrt(2*MZHH_QW*Q*ehh_odd)/HRJ
    ghh_odd = np.sqrt(2*mzhh_cb*Q*(vv -ehh_odd))/HRJ
    [ahh_odd, bhh_odd] = norm_odd(khh_odd, ghh_odd, qw_width)


    # perturbative method
    ee = np.append(ee_even, ee_odd)
    ke = np.append(ke_even, ke_odd)
    ge = np.append(ge_even, ge_odd)
    ae = np.append(ae_even, ae_odd)
    be = np.append(be_even, be_odd)
    nee = np.size(ee_even)
    ne = np.size(ee)

    ehh = np.append(ehh_even, ehh_odd)
    khh = np.append(khh_even, khh_odd)
    ghh = np.append(ghh_even, ghh_odd)
    ahh = np.append(ahh_even, ahh_odd)
    bhh = np.append(bhh_even, bhh_odd)
    nhh = np.size(ehh_even)
    nh = np.size(ehh)

    dz = 1e-10
    zz = np.arange(-25e-9, 25e-9, dz)
    nz = np.size(zz)
    psie = np.zeros(nz)
    psih = np.zeros(nz)

    cc = 160000                # [nm/cm] 10 times larger than Lengyel paper
    alpha_bulk = 5500          # [cm-1]


    [ve, ee_pert] = perturb(True, electric_field, ee, nee, ke, ge, ae, be, qw_width)
    [vhh, ehh_pert] = perturb(False, electric_field, ehh, nhh, khh, ghh, ahh, bhh, qw_width)


    if electric_field == 0:
       eexc = EG_GAAS +ee[0] +ehh[0]
    else:
       eexc = EG_GAAS +ee_pert +ehh_pert


    s = 0
    iz = 0
    for z in zz:
       if electric_field == 0:
           psie[iz] = psi_even(ke[0], ge[0], ae[0], be[0], z, qw_width)
       else:
           for ie in range(ne):
               if ie < nee:
                   psi = psi_even(ke[ie], ge[ie], ae[ie], be[ie], z, qw_width)
               else:
                   psi = psi_odd(ke[ie], ge[ie], ae[ie], be[ie], z, qw_width)

               psie[iz] = psie[iz] +ve[ie]*psi


       if electric_field == 0:
           psih[iz] = psi_even(khh[0], ghh[0], ahh[0], bhh[0], z, qw_width)
       else:
           for ih in range(nh):
               if ih < nhh:
                   psi = psi_even(khh[ih], ghh[ih], ahh[ih], bhh[ih], z, qw_width)
               else:
                   psi = psi_odd(khh[ih], ghh[ih], ahh[ih], bhh[ih], z, qw_width)

               psih[iz] = psih[iz] +vhh[ih]*psi


       s += psie[iz]*np.conj(psih[iz])
       iz += 1


    s = s*dz
    alpha_exmax = cc/(qw_width*1e9)*abs(s)**2 +alpha_bulk           # alpha_hh [cm-1]


    qw_width_nm = qw_width*1e9
    electric_field_mv_nm = electric_field * 1e-6
    ghh = (7.374 -0.511*qw_width_nm +0.0182*qw_width_nm**2 -0.054*electric_field_mv_nm +0.0161*electric_field_mv_nm**2)*1e-3    # [ev]

    lex = 1.24/eexc
    ll = np.arange(lex -0.001, lex +0.05, 0.0001)


    alpha = alpha_exmax/(1. +((eexc -1.24/ll)/ghh)**2)


    return ll, alpha, psie, psih, zz




def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()


    return idx
