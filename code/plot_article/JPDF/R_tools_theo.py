# Compute JDPF of the product
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd 
import sys 
import scipy.stats as st

#: Approximate earth angular speed
EARTH_ANG_SPEED = 7.292115e-5
#: Approximate earth radius
EARTH_RADIUS = 6370e3
#: Approximate gravity
GRAVITY = 9.81
#: Degrees / radians conversion factor
P0 = np.pi / 180

def sphere_distance(_lats, _late, _lons, _lone):
    dlat, dlon = P0 * (_late - _lats), P0 * (_lone - _lons)
    return EARTH_RADIUS * np.sqrt(dlat ** 2 + np.cos(P0 * _lats) * np.cos(P0 * _late) * dlon ** 2)

def compute_coriolis_factor(lat):
    return 2 * EARTH_ANG_SPEED * np.sin(lat * P0)

def v2rho(var_v):

    [Mp,L]=var_v.shape
    Lp=L+1
    Lm=L-1
    var_rho=np.zeros((Mp,Lp))
    var_rho[:,1:L]=0.5*(var_v[:,0:Lm]+var_v[:,1:L])
    var_rho[:,0]=var_rho[:,1]
    var_rho[:,Lp-1]=var_rho[:,L-1]
    return var_rho

def u2rho(var_u):

    [M,Lp]=var_u.shape
    Mp=M+1
    Mm=M-1
    var_rho=np.zeros((Mp,Lp))
    var_rho[1:M,:]=0.5*(var_u[0:Mm,:]+var_u[1:M,:])
    var_rho[0,:]=var_rho[1,:]
    var_rho[Mp-1,:]=var_rho[M-1,:]

    return var_rho


def rho2psi(var_rho):

    var_psi = 0.25*(var_rho[1:,1:]+var_rho[1:,:-1]+var_rho[:-1,:-1]+var_rho[:-1,1:])

    return var_psi

def psi2rho(var_psi):

    [M,L]=var_psi.shape
    Mp=M+1
    Lp=L+1
    Mm=M-1
    Lm=L-1

    var_rho=np.zeros((Mp,Lp))
    var_rho[1:M,1:L]=0.25*(var_psi[0:Mm,0:Lm]+var_psi[0:Mm,1:L]+var_psi[1:M,0:Lm]+var_psi[1:M,1:L])
    var_rho[0,:]=var_rho[1,:]
    var_rho[Mp-1,:]=var_rho[M-1,:]
    var_rho[:,0]=var_rho[:,1]
    var_rho[:,Lp-1]=var_rho[:,L-1]

    return var_rho

def psi2rho(var_psi):

    [M,L]=var_psi.shape
    Mp=M+1
    Lp=L+1
    Mm=M-1
    Lm=L-1

    var_rho=np.zeros((Mp,Lp))
    var_rho[1:M,1:L]=0.25*(var_psi[0:Mm,0:Lm]+var_psi[0:Mm,1:L]+var_psi[1:M,0:Lm]+var_psi[1:M,1:L])
    var_rho[0,:]=var_rho[1,:]
    var_rho[Mp-1,:]=var_rho[M-1,:]
    var_rho[:,0]=var_rho[:,1]
    var_rho[:,Lp-1]=var_rho[:,L-1]

    return var_rho

def rho2u(var_rho):

    var_u = 0.5*(var_rho[1:,:]+var_rho[:-1,:])

    return var_u

def rho2v(var_rho):

    var_v = 0.5*(var_rho[:,1:]+var_rho[:,:-1])

    return var_v

def diffy(var,pn,dn=1):

    if (np.ndim(pn)==2) and (var.shape[1]==pn.shape[1]):
        dvardy = (var[:,dn:]-var[:,:-dn])*0.5*(pn[:,dn:]+pn[:,:-dn])/dn
    else: 
        dvardy = (var[:,dn:]-var[:,:-dn])*pn/dn

    return dvardy

def diffx(var,pm,dn=1):

    if (np.ndim(pm)==2) and (var.shape[0]==pm.shape[0]): 
        dvardx = (var[dn:,:]-var[:-dn,:])*0.5*(pm[dn:,:]+pm[:-dn,:])/dn
    else: 
        dvardx = (var[dn:,:]-var[:-dn,:])*pm/dn

    return dvardx

def get_strain(u,v,pm,pn,z_r=None,z_w=None,mask=None):


    dvdx = diffx(v,rho2v(pm))
    dudy = diffy(u,rho2u(pn))
    dvdy = diffy(v,rho2v(pm))
    dudx = diffx(u,rho2u(pn))
        
    # everything on rho-grid
    dvdx = psi2rho(dvdx)
    dudy = psi2rho(dudy)
    

    #print(dvdy.shape)
    dvdy = np.pad(dvdy,pad_width = ((0, 0), (1, 1)),mode='edge')
    dudx = np.pad(dudx,pad_width = ((1, 1), (0, 0)),mode='edge')
    #print(dvdy.shape)
    
    strain = np.sqrt((dudx-dvdy)**2 + (dudy+dvdx)**2)

    return strain

def get_vrt(u,v,pm,pn,z_r=None,z_w=None,mask=None):


    dvdx = diffx(v,rho2v(pm))
    dudy = diffy(u,rho2u(pn))  
        
    #vrt on psi grid
    vrt = dvdx - dudy    
    
    return vrt


def add_contour_per(H_sum_tot,proba_list,xedges,yedges,c):
    #proba_list = [99.99]
    n = 0

    level = np.zeros(len(proba_list))
    p = np.zeros(len(proba_list))

    for proba in proba_list:

        i = dichotomie(f,-4,6,0.01,proba,H_sum_tot)
        #print(i)

        H_tot = np.sum(H_sum_tot)

        H_filter = np.where(H_sum_tot > 10**i, H_sum_tot,0)
        p[n] = np.sum(H_filter)*100/H_tot

        level[n] = 10**i
        n=n+1

    fmt = {}
    for l, s in zip(level, p):
        #fmt[l] = str(round(s,1))+'%'
        fmt[l] = ''

    CS = plt.contour(xedges,yedges,H_sum_tot,level,linewidths=2,alpha=1,colors=c,linestyles='-')
    plt.clabel(CS,level, inline=1, fontsize=18,fmt = fmt)
    
    return level,fmt


def compute_fraction(H,H_wc):
        
    H_wc_neg = np.where(H_wc<0,H_wc,0)

    xx, yy = np.meshgrid(xedges[1:], yedges[1:])
    #H_80 £*= np.where(H < 10**x_80, H,0)
    H_f = np.where(xx > 0.5, H,0)
    H_f = np.where(yy > xx, H_f,0)
    
    H_f_wc = np.where(xx > 0.5, H_wc,0)
    H_f_wc = np.where(yy > xx, H_f_wc,0)

    H_f_per = np.sum(H_f)*100/np.sum(H)
    H_f_wc_per = np.sum(H_f_wc)*100/np.sum(H_wc_neg)
    
    return(H_f_per,H_f_wc_per,np.sum(H_f_wc))

def f(x,p,H_sum_tot):
    
    H_filter = np.where(H_sum_tot > 10**x, H_sum_tot,0)
    H_tot = np.sum(H_sum_tot)
    return(np.sum(H_filter)*100/H_tot-p) 

def dichotomie(f,a,b,e,p,H_sum_tot):
    delta = 1

    while delta > e:
        m = a + (b - a) / 2
        delta = abs(b - a)
        #print("{:15} , {:15} , {:15} , {:15} , {:15} , {:15} , {:15} ".format(a,b,m,f(a),f(b),f(m),delta) )
        if f(m,p,H_sum_tot) == 0:
            return m
        elif f(a,p,H_sum_tot) * f(m,p,H_sum_tot)  > 0:
            a = m
        else:
            b = m
    return m