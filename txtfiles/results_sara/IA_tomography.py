#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 01:55:20 2023

@author: sara
"""
import fitsio
from astropy.cosmology import FLRW, LambdaCDM
from astropy.io import fits
import numpy as np
import astropy
from astropy.coordinates import SkyCoord
from astropy.table import Table, Column
from astropy.constants import G, c
from array import *
from astropy import units as u
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import treecorr
import os.path


#########################################################################################
################################## Functions ############################################
#########################################################################################

def tableMaker(photozMean=0):
    
    file_path_shapes = '/home/astro/aliqoliz/Downloads/y1a1-im3shape_v5_unblind_v2_matched_v4(1).fits'
    data_shapes = fits.open(file_path_shapes, memmap= True)
    data_shapes.info()
    print(data_shapes[1].columns)
    radecz_table = Table(data_shapes[1].data)
    radecz_table.sort(['coadd_objects_id'])
    file_path_z = '/home/astro/aliqoliz/Downloads/y1a1-gold-mof-badregion_BPZ.fits'
    data_z = fits.open(file_path_z, memmap= True)
    data_z.info()
    print(data_z[1].columns)
    ztable = Table(data_z[1].data)
    z_table= Table([ztable['coadd_objects_id'],ztable['mean_z'], ztable['z_mc'], ztable['z_sigma']],names=('coadd_objects_id','mean_z','z_mc','z_sigma'))
    z_table.sort(['coadd_objects_id'])
    print("Tables are ready...")
    
    flag_arg=0
    dec_arg=0
    zmin=0
    zmax0=0
    zmax1=0
    zmax2=0
    zmc= z_table['z_mc']
    zmean = z_table['mean_z']+photozMean
    zsig = z_table['z_sigma']
    f = radecz_table['flags_select']
    rao =(radecz_table['ra']+180)%360
    deco = radecz_table['dec']
    weo = radecz_table['weight']
    eo1 = radecz_table['e1']-radecz_table['c1']
    eo2 = radecz_table['e2']-radecz_table['c2']
    m=radecz_table['m']+1
    
    table1 = Table([f, rao, deco, zmc, zmean, zsig, eo1, eo2, weo,m], names=('flag','ra','dec','zmc','zmean','zsig', 'e1', 'e2', 'w','m'))
    table1.sort(['flag'])
    i=0
    for i in range(len(table1['flag'])):    
        if table1['flag'][i] > 0:
            flag_arg = i-1
            break
    print(flag_arg)
    T0=table1['ra'][:flag_arg]
    T1=table1['dec'][:flag_arg]
    T2=table1['zmc'][:flag_arg]
    T3=table1['zmean'][:flag_arg]
    T4=table1['zsig'][:flag_arg]
    T5 = table1['e1'][:flag_arg]
    T6 = table1['e2'][:flag_arg]
    T7 = table1['w'][:flag_arg]
    T8 = table1['m'][:flag_arg]    
    table2 = Table([T0,T1,T2,T3,T4,T5,T6,T7,T8], names=('ra','dec','zmc','zmean','zsig', 'e1', 'e2', 'w','m'))
    table2.sort(['dec'])
    i=0
    for i in range(len(table2['dec'])):    
        if table2['dec'][i] > -30:
            dec_arg = i-1
            break
    U0=table2['ra'][:dec_arg]
    U1=table2['dec'][:dec_arg]
    U2=table2['zmc'][:dec_arg]
    U3=table2['zmean'][:dec_arg]
    U4=table2['zsig'][:dec_arg]
    U5=table2['e1'][:dec_arg]
    U6=table2['e2'][:dec_arg]
    U7=table2['w'][:dec_arg]
    U8=table2['m'][:dec_arg]    
    table3= Table([U0,U1,U2,U3,U4,U5,U6,U7,U8], names=('ra','dec','zmc','zmean','zsig', 'e1', 'e2', 'w','m'))
    
    
    table3.sort(['zmean'])
    i=0
    for i in range(len(table3['zmean'])):
        if table3['zmean'][i] >= 0.3:       
            zmin = i
            break
    print(zmin)
    
    for i in range( len(table3['zmean'])):  
        if table3['zmean'][i] >= 0.45:
            zmax0 = i
            break
    print(zmax0)
    
    
    for i in range( len(table3['zmean'])):  
        if table3['zmean'][i] >= 0.55:
            zmax1 = i
            break
    print(zmax1)
    
    for i in range( len(table3['zmean'])):  
        if table3['zmean'][i] >= 0.7:
            zmax2 = i
            break
    print(zmax2)
    
    X0=table3['ra'][zmin:zmax0]
    X00=table3['dec'][zmin:zmax0]
    X1=table3['zmc'][zmin:zmax0]
    X2=table3['zmean'][zmin:zmax0]
    X3=table3['zsig'][zmin:zmax0]
    X4=table3['e1'][zmin:zmax0]
    X5=table3['e2'][zmin:zmax0]
    X6=table3['w'][zmin:zmax0]
    X7=table3['m'][zmin:zmax0]
    X8=np.multiply(table3['w'][zmin:zmax0],table3['m'][zmin:zmax0]+1)
    
    Y0=table3['ra'][zmax1:zmax2]
    Y00=table3['dec'][zmax1:zmax2]
    Y1=table3['zmc'][zmax1:zmax2]
    Y2=table3['zmean'][zmax1:zmax2]
    Y3=table3['zsig'][zmax1:zmax2]
    Y4=table3['e1'][zmax1:zmax2]
    Y5=table3['e2'][zmax1:zmax2]
    Y6=table3['w'][zmax1:zmax2]
    Y7=table3['m'][zmax1:zmax2]
    Y8=np.multiply(table3['w'][zmax1:zmax2],table3['m'][zmax1:zmax2]+1)
    
    table4=Table([X0,X00,X1,X2,X3,X4,X5,X6,X7,X8], names=('ra','dec','zmc','zmean','zsig','e1','e2','w','m+1','wm'))
    table5=Table([Y0,Y00,Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8], names=('ra','dec','zmc','zmean','zsig','e1','e2','w','m+1','wm'))
    table4.write('/home/astro/aliqoliz/Documents/IA_pipeline/DESzmean0_meanShift_'+str(photozMean)+'.fits', format='fits')
    table5.write('/home/astro/aliqoliz/Documents/IA_pipeline/DESzmean1_meanShift_'+str(photozMean)+'.fits', format='fits')
    data_z.close()
    data_shapes.close()
    return ['/home/astro/aliqoliz/Documents/IA_pipeline/DESzmean0_meanShift_'+str(photozMean)+'.fits', '/home/astro/aliqoliz/Documents/IA_pipeline/DESzmean1_meanShift_'+str(photozMean)+'.fits']

#Runs TreeCorr on the whole footprint, returns boost and gamma_t
#for the source samples
def BoostNshear(lens_path, rand_path, source_path, smin, smax, nbins, label):
    B=[]
    Xi=[]
    r=[]
    lens=treecorr.Catalog(lens_path,ra_col = 'ra', dec_col = 'dec',ra_units = 'deg', dec_units ='deg')
    rand=treecorr.Catalog(rand_path,ra_col = 'ra', dec_col = 'dec',ra_units = 'deg', dec_units ='deg')
    BC=boostCorr(lens_path, rand_path)
    for s in source_path:
        if weights=='ones':
            source = treecorr.Catalog(s,ra_col = 'ra', dec_col = 'dec', g1_col = 'e1', g2_col = 'e2',ra_units = 'deg', dec_units ='deg')
        if weights!='ones':
            source = treecorr.Catalog(s,w_col='w',ra_col = 'ra', dec_col = 'dec', g1_col = 'e1', g2_col = 'e2',ra_units = 'deg', dec_units ='deg')
        rg = treecorr.NGCorrelation(min_sep = smin, max_sep = smax, nbins = nbins, sep_units = 'arcmin', verbose = 3,bin_slop=0.)
        ng = treecorr.NGCorrelation(min_sep = smin, max_sep = smax, nbins = nbins, sep_units = 'arcmin', verbose = 3,bin_slop=0.)
        NG=ng.process(lens,source)
        RG=rg.process(rand,source)
        ng.write('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/lens_source_'+str(s[-6]))
        rg.write('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/rand_source_'+str(s[-6]))
        xi_rel, xi_im, varxi = ng.calculateXi(RG)
        boost=np.divide(ng.weight*BC, rg.weight)
        np.savetxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/AvgBoost'+str(s[-6]),boost)
        np.savetxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/AvgGammaT'+str(s[-6]),xi_rel)
    array=np.loadtxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/lens_source_0')
    R=array[:,1]
    return R
        
#Returns radial comoving dostance between 2 redshift bins 
        
def radialComSep(zL, zS, H0, Om0, Ode0):
    i=j=0
    D=[[None for x in range(len(zS))] for y in range(len(zL))]    
    XL=LambdaCDM(H0=H0, Om0=Om0, Ode0=Ode0).comoving_distance(zL)
    XS=LambdaCDM(H0=H0, Om0=Om0, Ode0=Ode0).comoving_distance(zS)
    for i in range(len(XL)):
        for j in range(len(XS)):
            D[i][j]=np.abs(XL[i]-XS[j])
    return D 


#Divides the whole footprint of source, lens, and rands into the given number of patches.

def Patches(npatch, zList, lens_path, rand_path, source_path, photozMean):
    pointer=0
    file_path = lens_path
    data = fits.open(file_path, memmap= True)
    data.info()
    print(data[1].columns)
    table = Table(data[1].data)
    table.sort(['ra'])

    RAmax=np.amax(table['ra'])
    RAmin=np.amin(table['ra'])

    DECmax=np.amax(table['dec'])
    DECmin=np.amin(table['dec'])
            
    n = int(len(table['ra'].data)/npatch)
    patchArea = (RAmax-RAmin)*(DECmax-DECmin)/npatch
    patchEdge = np.sqrt(patchArea)
    RAbins = np.arange(RAmin, RAmax, patchEdge)
    binsNum = len(RAbins)
    RAhist, binsEdge, patches = plt.hist(table['ra'].data, RAbins) 
    patchInBin = np.rint(np.multiply(1/n,RAhist[:]))
    edgeArgs = [[None for x in range(2)] for y in range(binsNum)]
    counter=0        

    for i in range(binsNum-1):
        for j in range(len(table['ra'].data)):
            if table['ra'][j]>=binsEdge[i]:
                leftIndex = j 
                break
        for j in range(len(table['ra'])):
            if table['ra'][j]>=(binsEdge[i]+patchEdge):
                rightIndex = j 
                break
             
        temp = Table([table['ra'][leftIndex:rightIndex],
                   table['dec'][leftIndex:rightIndex],
                   table['z'][leftIndex:rightIndex]], 
                   names=('ra','dec','z'))
         
        temp.sort(['dec'])    
         

        for k in range(int(patchInBin[i])):
            if patchInBin[i]>=1.:
                indexdown = int(len(temp['ra'].data)*k/patchInBin[i])
                indexup = int(len(temp['ra'].data)*(k+1)/patchInBin[i])
                dec= temp['dec'][indexdown:indexup]
                ra= temp['ra'][indexdown:indexup]
                z= temp['z'][indexdown:indexup]
                theTable = Table([ra,dec,z], names=('ra','dec','z'))
                theTable.write('/home/astro/aliqoliz/Documents/IA_tomography_results/Patches_'+str(photozMean)+'/lensPatch'+str(counter)+'.fits', format='fits')
                counter+=1
            else:
                theTable = Table([temp['ra'],temp['dec'],temp['z']], names=('ra','dec','z'))
                theTable.write('/home/astro/aliqoliz/Documents/IA_tomography_results/Patches_'+str(photozMean)+'/lensPatch'+str(counter)+'.fits', format='fits')
                counter+=1
    for i in range(counter):
        file_path = rand_path
        lp=lens_path
        dataS = fits.open(file_path, memmap= True)
        dataS.info()
        T=Table(dataS[1].data)
        file_path = '/home/astro/aliqoliz/Documents/IA_tomography_results/Patches_'+str(photozMean)+'/lensPatch'+str(i)+'.fits'
        data = fits.open(file_path, memmap= True)
        data.info()
        table = Table(data[1].data)
        ramin = np.amin(table['ra'].data)
        ramax = np.amax(table['ra'].data)
        decmin = np.amin(table['dec'].data)
        decmax = np.amax(table['dec'].data)
        A=[]
        for j in range(len(T['ra'])):
            if (ramin>T['ra'][j]) or (T['ra'][j]>ramax) or (decmin>T['dec'][j]) or (T['dec'][j]>decmax):
                A.append(j)
        T.remove_rows(A)
        T.write('/home/astro/aliqoliz/Documents/IA_tomography_results/Patches_'+str(photozMean)+'/randPatch'+str(i)+'.fits', format='fits')

        
    for s in source_path:
        for i in range(counter):
            file_path = s
            lp=lens_path
            dataS = fits.open(file_path, memmap= True)
            dataS.info()
            T=Table(dataS[1].data)
            file_path = '/home/astro/aliqoliz/Documents/IA_tomography_results/Patches_'+str(photozMean)+'/randPatch'+str(i)+'.fits'
            data = fits.open(file_path, memmap= True)
            data.info()
            table = Table(data[1].data)
            ramin = np.amin(table['ra'].data)
            ramax = np.amax(table['ra'].data)
            decmin = np.amin(table['dec'].data)
            decmax = np.amax(table['dec'].data)
            A=[]
            for j in range(len(T['ra'])):
                if (ramin>T['ra'][j]) or (T['ra'][j]>ramax) or (decmin>T['dec'][j]) or (T['dec'][j]>decmax):
                    A.append(j)
            T.remove_rows(A)
            T.write('/home/astro/aliqoliz/Documents/IA_tomography_results/Patches_'+str(photozMean)+'/source'+str(s[51])+'Patch'+str(i)+'.fits', format='fits')

#Runs TreeCorr on combined npatch-1 Jackknife regions for source, lens, and rands.
def shearPosClassic(npatch, smin, smax, nbins, z, kmeans, label):

    spaths=[]
    lpaths=[]
    rlpaths=[]
    for j in range(npatch):
        lpaths.append('/home/astro/aliqoliz/Documents/IA_tomography_results/Patches_'+str(photozMean)+'/lensPatch'+str(j)+'.fits')
        spaths.append('/home/astro/aliqoliz/Documents/IA_tomography_results/Patches_'+str(photozMean)+'/source'+str(z)+'Patch'+str(j)+'.fits')
        rlpaths.append('/home/astro/aliqoliz/Documents/IA_tomography_results/Patches_'+str(photozMean)+'/randPatch'+str(j)+'.fits')

    for i in range(npatch):
        lenspaths = np.delete(lpaths, i)
        sourcepaths= np.delete(spaths, i)
        randompaths= np.delete(rlpaths, i)
        RAl=[]
        DECl=[]
        RAs=[]
        DECs=[]
        RArl = []
        DECrl = []
        E1=[]
        E2=[]
        W=[]
        for file_path in lenspaths:
            data_shapes = fits.open(file_path, memmap= True)
            data_shapes.info()
            Lradecz = Table(data_shapes[1].data)
            for k in range(len(Lradecz['ra'].data)):
                Rl=Lradecz['ra'][k]
                RAl.append(Rl)
                Dl=Lradecz['dec'][k]
                DECl.append(Dl)
        for file_path in sourcepaths:
            data_shapes = fits.open(file_path, memmap= True)
            data_shapes.info()
            Sradecz = Table(data_shapes[1].data)
            for k in range(len(Sradecz['ra'].data)):
                Rs=Sradecz['ra'][k]
                RAs.append(Rs)
                Ds=Sradecz['dec'][k]
                DECs.append(Ds)
                E1s=Sradecz['e1'][k]
                E1.append(E1s)
                E2s=Sradecz['e2'][k]
                E2.append(E1s)
                if weights!='ones':
                    Ws=Sradecz['w'][k]
                    W.append(Ws)
        for file_path in randompaths:
            data_shapes = fits.open(file_path, memmap= True)
            data_shapes.info()
            RMradecz = Table(data_shapes[1].data)
            for k in range(len(RMradecz['ra'].data)):
                Rrl=RMradecz['ra'][k]
                RArl.append(Rrl)
                Drl=RMradecz['dec'][k]
                DECrl.append(Drl)
        l = treecorr.Catalog(ra = RAl, dec = DECl, ra_units = 'deg', dec_units ='deg')
        if weights=='ones':
            s = treecorr.Catalog(ra = RAs, dec = DECs, g1 = E1, g2 = E2, ra_units = 'deg', dec_units ='deg')
        if weights!='ones':
            s = treecorr.Catalog(ra = RAs, dec = DECs, g1 = E1, g2 = E2, ra_units = 'deg', dec_units ='deg',w=W)
        r = treecorr.Catalog(ra = RArl, dec = DECrl, ra_units = 'deg', dec_units ='deg')    
        rg = treecorr.NGCorrelation(min_sep = smin, max_sep = smax, nbins = nbins, sep_units = 'arcmin', verbose = 3,bin_slop=0.)
        ng = treecorr.NGCorrelation(min_sep = smin, max_sep = smax, nbins = nbins, sep_units = 'arcmin', verbose = 3,bin_slop=0.)
        NG=ng.process(l,s)
        RG=rg.process(r,s)
        ng.savetxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/NG_345_'+str(z)+'_patch'+str(i)+'deleted', rg=RG)
        rg.savetxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/RG_345_'+str(z)+'_patch'+str(i)+'deleted')
            
#Runs TreeCorr on each Jackknife regions for source, lens, and rands.            
def shearPosSLR(npatch, z, smin, smax, nbins,weights, kmeans, label, photozMean):
    for i in range(npatch):
        if kmeans==False:
            lpath='/home/astro/aliqoliz/Documents/IA_tomography_results/Patches_'+str(photozMean)+'/lensPatch'+str(i)+'.fits'
            spath='/home/astro/aliqoliz/Documents/IA_tomography_results/Patches_'+str(photozMean)+'/source'+str(z)+'Patch'+str(i)+'.fits'  
            rlpath='/home/astro/aliqoliz/Documents/IA_tomography_results/Patches_'+str(photozMean)+'/randPatch'+str(i)+'.fits'
        if kmeans==True:
            lpath='/home/astro/aliqoliz/Documents/IA_tomography_kmeansPatches/Patches/rM345_'+str(i)+'.fits'
            rlpath='/home/astro/aliqoliz/Documents/IA_tomography_kmeansPatches/Patches/rM345rand_'+str(i)+'.fits'
        l = treecorr.Catalog(lpath, ra_col = 'ra', dec_col = 'dec', ra_units = 'deg', dec_units ='deg')
        if weights=='ones':
            s = treecorr.Catalog(spath, ra_col = 'ra', dec_col = 'dec', g1_col = 'e1', g2_col = 'e2', ra_units = 'deg', dec_units ='deg')
        if weights!='ones':
            s = treecorr.Catalog(spath, w_col='w',ra_col = 'ra', dec_col = 'dec', g1_col = 'e1', g2_col = 'e2', ra_units = 'deg', dec_units ='deg')
        r = treecorr.Catalog(rlpath, ra_col = 'ra', dec_col = 'dec', ra_units = 'deg', dec_units ='deg')    
        rg = treecorr.NGCorrelation(min_sep = smin, max_sep = smax, nbins = nbins, sep_units = 'arcmin', verbose = 3,bin_slop=0.)
        ng = treecorr.NGCorrelation(min_sep = smin, max_sep = smax, nbins = nbins, sep_units = 'arcmin', verbose = 3,bin_slop=0.)
        NG=ng.process(l,s)
        RG=rg.process(r,s)
        rg.write('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/RG_345_'+str(z)+'_patch'+str(i))            
        ng.write('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/NG_345_'+str(z)+'_patch'+str(i),rg=RG)
            
#Runs TreeCorr on each Jackknife regions for lens and rands but complete source catalog.    
def shearPosLR(npatch, z, source_path, smin,smax, nbins,weights, kmeans,label, photozMean):
    spath=source_path
    for i in range(npatch):
        if kmeans==False:
            lpath='/home/astro/aliqoliz/Documents/IA_tomography_results/Patches_'+str(photozMean)+'/lensPatch'+str(i)+'.fits'
            rlpath='/home/astro/aliqoliz/Documents/IA_tomography_results/Patches_'+str(photozMean)+'/randPatch'+str(i)+'.fits'
        if kmeans==True:
            lpath='/home/astro/aliqoliz/Documents/IA_tomography_kmeansPatches/Patches/rM345_'+str(i)+'.fits'
            rlpath='/home/astro/aliqoliz/Documents/IA_tomography_kmeansPatches/Patches/rM345rand_'+str(i)+'.fits'            
        l = treecorr.Catalog(lpath, ra_col = 'ra', dec_col = 'dec', ra_units = 'deg', dec_units ='deg')
        if weights=='ones':
            s = treecorr.Catalog(spath, ra_col = 'ra', dec_col = 'dec', g1_col = 'e1', g2_col = 'e2', ra_units = 'deg', dec_units ='deg')
            k= treecorr.Catalog(spath,k_col='m+1', ra_col = 'ra', dec_col = 'dec', ra_units = 'deg', dec_units ='deg')
        if weights!='ones':
#            s = treecorr.Catalog(spath,w_col='w', ra_col = 'ra', dec_col = 'dec', g1_col = 'e1', g2_col = 'e2', ra_units = 'deg', dec_units ='deg')
            s = treecorr.Catalog(spath,w_col='w', ra_col = 'ra', dec_col = 'dec', g1_col = 'e1', g2_col = 'e2', ra_units = 'deg', dec_units ='deg')
            k= treecorr.Catalog(spath,k_col='m+1',w_col='w', ra_col = 'ra', dec_col = 'dec', ra_units = 'deg', dec_units ='deg')
        
        r = treecorr.Catalog(rlpath, ra_col = 'ra', dec_col = 'dec', ra_units = 'deg', dec_units ='deg')    
        rg = treecorr.NGCorrelation(min_sep = smin, max_sep = smax, nbins = nbins, sep_units = 'arcmin', verbose = 3,bin_slop=0.)
        ng = treecorr.NGCorrelation(min_sep = smin, max_sep = smax, nbins = nbins, sep_units = 'arcmin', verbose = 3,bin_slop=0.)
        kg=treecorr.NKCorrelation(min_sep = smin, max_sep = smax, nbins = nbins, sep_units = 'arcmin', verbose = 3,bin_slop=0.)
        KG=kg.process(l,k)
        NG=ng.process(l,s)
        RG=rg.process(r,s)
        rg.write('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/RG_345_'+str(z)+'_patch'+str(i))            
        ng.write('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/NG_345_'+str(z)+'_patch'+str(i),rg=RG)    
        kg.write('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/KG_345_'+str(z)+'_patch'+str(i))
    
def errorBarClassic(npatch, zList, F0, F1, Sig1, Sig0, label, boostCorrection):

    for i in range(npatch):
        gammaA = np.loadtxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/NG_345_'+str(zList[0])+'_patch'+str(i)+'deleted')
        gammaB = np.loadtxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/NG_345_'+str(zList[1])+'_patch'+str(i)+'deleted')
        RandA = np.loadtxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/RG_345_'+str(zList[0])+'_patch'+str(i)+'deleted')
        RandB = np.loadtxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/RG_345_'+str(zList[1])+'_patch'+str(i)+'deleted')
        BoostA= np.divide(gammaA[:,6]*boostCorrection[i], RandA[:,6])
        BoostB=np.divide(gammaB[:,6]*boostCorrection[i], RandB[:,6])
        estimator = []
        for j in range(len(gammaB[:,3])):
            numerator = gammaB[:,3][j]*Sig1-gammaA[:,3][j]*Sig0
            denominator = (BoostB[j]+1-F1[i][j])*Sig1-(BoostA[j]+1-F0[i][j])*Sig0
            estimator.append(numerator/denominator)
        np.savetxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/GammaIA'+str(i),estimator)
    sigma=[]
    for k in range(len(gammaA[:,0])):
        a=[]    
        for i in range(npatch):        
            A = np.loadtxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/GammaIA'+str(i))        
            a.append(A[k])
        x=np.sqrt((npatch-1)*np.var(a))
        sigma.append(x)
    np.savetxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/ErrorBars',sigma)



#Returns Covariance and Correlation matrix along with diagonal terms of covariance matrix as a 1D array;
#Be careful with what "path" means here, it's gammaIAs' paths without iterators
def CovCorrMat(npatch, nbins, label):
    
    avg = [None for k in range(nbins)]
    for i in range(nbins):
        X=0
        for j in range(npatch):
            array = np.loadtxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/GammaIA'+str(j))        
            X += array[i]
        avg[i] = X/npatch
        
    Covariance = [[None for k in range(nbins)]for l in range(nbins)]
    for i in range(nbins):
        for j in range(nbins):
            X=0        
            for k in range(npatch):
                array = np.loadtxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/GammaIA'+str(k))
                X+=((array[i]-avg[i])*(array[j]-avg[j]))
            Covariance[i][j]=((npatch-1)/npatch)*X
    np.save('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/CovMat',Covariance)
    diag=[]    
    for i in range(nbins):
        diag.append(Covariance[i][i])
    Correlation = [[None for k in range(nbins)]for l in range(nbins)]
    for i in range(nbins):
        for j in range(nbins):
            Correlation[i][j]=Covariance[i][j]/np.sqrt(Covariance[i][i]*Covariance[j][j])
    np.save('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/CorrMat',Correlation)
    
    gamma=np.loadtxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/GammaIAfinal')    
    CovInv= np.linalg.inv(Covariance)

    chi2=np.matmul(np.matmul(gamma,CovInv),gamma)

    hartlapFactor=(npatch-nbins-2)/(npatch-1)
    snr=hartlapFactor*chi2
    SNR=[snr]
    np.savetxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/chi2', SNR)    
    X=[]
    Y=[]
    flatCorr=[]
    for i in range(nbins):
        for j in range(nbins):
            X.append(i)
            Y.append(j)
    tempCov=np.array(Covariance)
    flatCov=tempCov.flatten()
    for i in range(nbins): 
        for j in range(nbins):
            flatCorr.append(Covariance[i][j]/np.sqrt(Covariance[i][i]*Covariance[j][j]))
    
    '''cr=plt.scatter(X,
            Y,
            c =flatCorr, cmap = 'Purples', 
            lw = 0, marker = 's', s = 1000)
    plt.xlabel(r'bin number', fontsize= 15)
    plt.xticks(ticks=range(nbins),fontsize= 10)
    plt.yticks(ticks=range(nbins),fontsize= 10)
    plt.ylabel(r'bin number', fontsize= 15)
    plt.colorbar(cr)
    plt.show()
    plt.savefig('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/correlation.png', dpi=300)
    cv=plt.scatter(X,
            Y,
            c =np.log10(np.abs(flatCov)), cmap = 'Purples', 
            lw = 0, marker = 's', s = 1000)
    plt.xlabel(r'bin number', fontsize= 15)
    plt.xticks(ticks=range(nbins),fontsize= 10)
    plt.yticks(ticks=range(nbins),fontsize= 10)
    plt.ylabel(r'bin number', fontsize= 15)
    plt.title('$\\chi^2$ ='+str(SNR))
    plt.colorbar(cv)
    plt.show()
    plt.savefig('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/covariance.png', dpi=300)'''
    return Covariance, Correlation, diag, SNR


def Map(lp, sp, H0, omegaM, omegaDE, redshiftBins_s, redshiftBins_l,label, comovingLimit,z):
    
    lens_path= lp
    source_path=sp
    
    data_lens = fits.open(lens_path, memmap= True)
    data_lens.info()
    tableL = Table(data_lens[1].data)
    data_source = fits.open(source_path, memmap= True)
    data_source.info()
    tableS = Table(data_source[1].data)    
    zl=tableL['z']
    zs=tableS['zmc']
#    binL = np.linspace(np.amin(zl), np.amax(zl), num=redshiftBins+1)
    binL = np.linspace(np.amin(zl), np.amax(zl), num=redshiftBins_l)
    Lmid= 0.5*(binL[1:]+binL[:-1])
    binLmid= np.append(0.,Lmid)
    
#    binS = np.linspace(np.amin(zs), np.amax(zs), num=redshiftBins+1)
    binS = np.linspace(np.amin(zs), np.amax(zs), num=redshiftBins_s)
    Smid = 0.5*(binS[1:]+binS[:-1])
    binSmid = np.append(0., Smid)
    
    Dist=radialComSep(Lmid,Smid, H0, omegaM, omegaDE)
    
    Xs=LambdaCDM(H0=H0, Om0=omegaM, Ode0=omegaDE).comoving_distance(Smid)
    Xl=LambdaCDM(H0=H0, Om0=omegaM, Ode0=omegaDE).comoving_distance(Lmid)
    
    
    SigInvMap=[[None for x in range(len(Xs))] for y in range(len(Xl))]    
    Fmap= [[None for x in range(len(Xs))] for y in range(len(Xl))]
    
    for i in range(len(Xl)):
        for j in range(len(Xs)):
            if ((Xl[i].value)<=(Xs[j].value)):
#                SigInvMap[i][j]= ((((1/(Xs[j]/(np.abs(Dist[i][j])*((Xl[i]))*(1+binLmid[i])))).decompose())*((4*np.pi*G)/(c**2))).to(u.pc**2/u.M_sun)).value
                SigInvMap[i][j]= ((((1/(Xs[j]/(np.abs(Dist[i][j])*((Xl[i]))*(1+Lmid[i])))).decompose())*((4*np.pi*G)/(c**2))).to(u.pc**2/u.M_sun)).value
            else:
                SigInvMap[i][j] =0.
                
            if (np.abs((Xl[i]-Xs[j]).value)<comovingLimit):
                Fmap[i][j]=1
            else:
                Fmap[i][j]=0
    np.savetxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/Fmap'+str(z),Fmap)
    np.savetxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/SigMap'+str(z),SigInvMap)
    return Fmap, SigInvMap


def MapArray(lp, sp, H0, omegaM, omegaDE, redshiftBins_s, redshiftBins_l,label, comovingL,z):
    
    lens_path= lp
    source_path=sp
    
    data_lens = fits.open(lens_path, memmap= True)
    data_lens.info()
    tableL = Table(data_lens[1].data)
    data_source = fits.open(source_path, memmap= True)
    data_source.info()
    tableS = Table(data_source[1].data)    
    zl=tableL['z']
    zs=tableS['zmc']
#    binL = np.linspace(np.amin(zl), np.amax(zl), num=redshiftBins+1)
    binL = np.linspace(np.amin(zl), np.amax(zl), num=redshiftBins_l)
    Lmid= 0.5*(binL[1:]+binL[:-1])
    binLmid= np.append(0.,Lmid)
    
#    binS = np.linspace(np.amin(zs), np.amax(zs), num=redshiftBins+1)
    binS = np.linspace(np.amin(zs), np.amax(zs), num=redshiftBins_s)
    Smid = 0.5*(binS[1:]+binS[:-1])
    binSmid = np.append(0., Smid)
    print((LambdaCDM(H0=72, Om0=0.3, Ode0=0.7).comoving_distance(np.amax(zs))-LambdaCDM(H0=72, Om0=0.3, Ode0=0.7).comoving_distance(np.amin(zs)))/redshiftBins_s)
    Dist=radialComSep(Lmid,Smid, H0, omegaM, omegaDE)
    
    Xs=LambdaCDM(H0=H0, Om0=omegaM, Ode0=omegaDE).comoving_distance(Smid)
    Xl=LambdaCDM(H0=H0, Om0=omegaM, Ode0=omegaDE).comoving_distance(Lmid)
    
    
    SigInvMap=[[None for x in range(len(Xs))] for y in range(len(Xl))]
    Fmap=[]    
    
    for i in range(len(Xl)):
        for j in range(len(Xs)):
            if ((Xl[i].value)<=(Xs[j].value)):
#                SigInvMap[i][j]= ((((1/(Xs[j]/(np.abs(Dist[i][j])*((Xl[i]))*(1+binLmid[i])))).decompose())*((4*np.pi*G)/(c**2))).to(u.pc**2/u.M_sun)).value
                SigInvMap[i][j]= ((((1/(Xs[j]/(np.abs(Dist[i][j])*((Xl[i]))*(1+Lmid[i])))).decompose())*((4*np.pi*G)/(c**2))).to(u.pc**2/u.M_sun)).value
            else:
                SigInvMap[i][j] =0.
    for k in comovingL:
        f =[[None for x in range(len(Xs))] for y in range(len(Xl))]
        for i in range(len(Xl)):
            for j in range(len(Xs)):        
                if (np.abs((Xl[i]-Xs[j]).value)<k):
                    f[i][j]=1
                else:
                    f[i][j]=0
        Fmap.append(f)
    np.save('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/Fmap'+str(z),Fmap)
    np.savetxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/SigMap'+str(z),SigInvMap)
    return Fmap, SigInvMap


def histogramming(sourcepath, lenspath, z,FandSigma_cMode , redshiftBins_s, redshiftBins_l,weights, label, photozNoise,photozDev, photozMean, k=0):
    
    lens_path=lenspath
    data_lens = fits.open(lens_path, memmap= True)
    data_lens.info()
    tableL = Table(data_lens[1].data)

    file_path_shapes =sourcepath
    data_shapes = fits.open(file_path_shapes, memmap= True)
    data_shapes.info()
    tableS = Table(data_shapes[1].data)
    
    zl=tableL['z']
    zs=tableS['zmc']
    minZmc=np.amin(zs)
    print('min: ',minZmc)
    maxZmc=np.amax(zs)
    print('max: ',maxZmc)
    
    w = np.zeros(len(zs))
    if weights == 'ones':
        w += 1.
    else:
        w += tableS['w'] 
    noise=np.zeros(len(zs))
    
    if photozNoise == True:
        noise+=np.random.normal(loc=photozMean, scale=photozDev, size=len(zs))
        zs+=noise
    if (FandSigma_cMode=='once'):
        binS = np.linspace(minZmc, maxZmc, num=redshiftBins_s+1)
        Smid = 0.5*(binS[1:]+binS[:-1])

        numSweight, binS= np.histogram(zs, Smid, weights=w)
        np.savetxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/source'+str(z)+'InBin', numSweight)
        np.savetxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/source'+str(z)+'Edges', binS)
        if (os.path.exists('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/lensInBin')==False):    
            binL = np.linspace(np.amin(zl), np.amax(zl), num=redshiftBins_l+1)
            Lmid= 0.5*(binL[1:]+binL[:-1])

            numLweight, binsL= np.histogram(zl, Lmid)
            np.savetxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/lensInBin', numLweight)
            np.savetxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/lensEdges', binsL)    
            
    if (FandSigma_cMode=='LR'):
        binL = np.linspace(np.amin(zl), np.amax(zl), num=redshiftBins_l+1)
        Lmid= 0.5*(binL[1:]+binL[:-1])
        binLmid= np.append(0.,Lmid)
#        numLweight, binsL= np.histogram(zl, binLmid)
        numLweight, binsL= np.histogram(zl, Lmid)
        np.savetxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/lensInBin'+str(k), numLweight)
        if (os.path.exists('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/source'+str(z)+'InBin')==False):
            binS = np.linspace(minZmc, maxZmc, num=redshiftBins_s+1)
            Smid = 0.5*(binS[1:]+binS[:-1])
            binSmid = np.append(0., Smid)
#            numSweight, binS= np.histogram(zs, binSmid, weights=w)
            numSweight, binS= np.histogram(zs, Smid, weights=w)
            np.savetxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/source'+str(z)+'InBin', numSweight)
            
    if (FandSigma_cMode=='SLR'):
        binL = np.linspace(np.amin(zl), np.amax(zl), num=redshiftBins_l+1)
        Lmid= 0.5*(binL[1:]+binL[:-1])
        binLmid= np.append(0.,Lmid)
#        numLweight, binsL= np.histogram(zl, binLmid)
        numLweight, binsL= np.histogram(zl, Lmid)
        np.savetxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/lensInBin'+str(k), numLweight)
        binS = np.linspace(minZmc, maxZmc, num=redshiftBins_s+1)
        Smid = 0.5*(binS[1:]+binS[:-1])
        binSmid = np.append(0., Smid)
#        numSweight, binS= np.histogram(zs, binSmid, weights=w)
        numSweight, binS= np.histogram(zs, Smid, weights=w)
        np.savetxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/source'+str(z)+'InBin'+str(k), numSweight)
                

def boostCorr(lp, rp):
    data_lens = fits.open(lp, memmap= True)
    data_lens.info()
    tableL = Table(data_lens[1].data)
    data_rand = fits.open(rp, memmap= True)
    data_rand.info()
    tableR = Table(data_rand[1].data)
    bc= len(tableR['z'])/len(tableL['z'])
    return bc
        
def analysis(patchRun=True,  kmeans=False, npatch=40, nbins=10, minSep=0.2, maxSep=180, weights= 'ones',redshiftBins_s =1000, redshiftBins_l=1000, F_mode='flat', F_min_threshold=2, comovingLimit=100,Mode='LR',
 FandSigma_cMode='once', boostCorrectionMode='once', H0=72, omegaM=0.3, omegaDE=0.7, photozNoise=True,
 photozMean = 0.05, photozDev=0.1, sourcepath=None, lenspath=None, randpath=None, label='test'):
    R= [  0.30541 ,  0.60341 ,  1.1918 ,   2.3571  ,  4.6615  ,  9.2158  , 18.198,
      35.928 ,   70.893  , 139.8    ]            
#    R=BoostNshear(lens_path=lenspath, rand_path=randpath, source_path=sourcepath, smin=minSep, smax=maxSep, nbins=nbins, label=label)
    zList=[0,1]            
    if F_mode != 'flat':
        transScale=LambdaCDM(H0=H0, Om0=omegaM, Ode0=omegaDE).comoving_transverse_distance(0.5*(0.3+0.45))
        print('transScale: ',transScale.value)
        separationL=np.multiply(R,int(transScale.value))/3437.75
        comovingL=[]
        for x in separationL:
            if x <= F_min_threshold:
                comovingL.append(F_min_threshold)
            else:
                comovingL.append(x)
        print(comovingL)
    if F_mode == 'flat':                
        FMap0, SigMap0 = Map(lp=lenspath, sp=sourcepath[0], H0=H0, omegaM=omegaM, omegaDE=omegaDE, redshiftBins_s=redshiftBins_s, redshiftBins_l=redshiftBins_l,comovingLimit=comovingLimit, label=label,z=0)
        FMap1, SigMap1= Map(lp=lenspath, sp=sourcepath[1], H0=H0, omegaM=omegaM, omegaDE=omegaDE, redshiftBins_s=redshiftBins_s,redshiftBins_l=redshiftBins_l,comovingLimit=comovingLimit,label=label, z=1)
    else:
        FMap0, SigMap0 = MapArray(lp=lenspath, sp=sourcepath[0], H0=H0, omegaM=omegaM, omegaDE=omegaDE, redshiftBins_s=redshiftBins_s,redshiftBins_l=redshiftBins_l,comovingL=comovingL, label=label,z=0)
        FMap1, SigMap1= MapArray(lp=lenspath, sp=sourcepath[1], H0=H0, omegaM=omegaM, omegaDE=omegaDE, redshiftBins_s=redshiftBins_s,redshiftBins_l=redshiftBins_l,comovingL=comovingL,label=label, z=1)   
#    FMap0 = np.load('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/Fmap0.npy')
#    FMap1=np.load('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/Fmap1.npy')
#    SigMap0=np.loadtxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/SigMap0')
#    SigMap1=np.loadtxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/SigMap1')
    if patchRun == True:
        if kmeans == False:
            Patches(npatch=npatch, zList=zList, lens_path=lensPath, rand_path=randPath, source_path=sourcePath, photozMean=photozMean)
        if kmeans == True:
            rand=treecorr.Catalog(randpath, ra_col = 'ra', dec_col = 'dec', z='z',ra_units = 'deg', dec_units ='deg', npatch=npatch,
             save_patch_dir='/home/astro/aliqoliz/Documents/IA_tomography_results/kmeansPatches')
            treecorr.Catalog(lenspath, ra_col = 'ra', dec_col = 'dec', z='z',ra_units = 'deg', dec_units ='deg', 
            patch_centers=rand.patch_centers,
             save_patch_dir='/home/astro/aliqoliz/Documents/IA_tomography_results/kmeansPatches')    
            treecorr.Catalog(sourcepath, ra_col = 'ra', dec_col = 'dec', z='zmc',ra_units = 'deg', dec_units ='deg',w='w',
             patch_centers=rand.patch_centers,
             save_patch_dir='/home/astro/aliqoliz/Documents/IA_tomography_results/kmeansPatches')
    F0s=[]
    F1s=[]
    Sig0s=[]
    Sig1s=[]    
    for z in zList:
        F = np.zeros(1)
        Sig = np.zeros(1)
        Lhist = np.zeros(redshiftBins_l-1)
        Shist= np.zeros(redshiftBins_s-1)
#        Lhist = np.zeros(redshiftBins)
#        Shist= np.zeros(redshiftBins)
        fnum= np.zeros(1)
        snum = np.zeros(1)
        fden= np.zeros(1)
        sden= np.zeros(1)
        NL=np.zeros(1)
        NS=np.zeros(1)
        pathL=[]
        pathS=[]
        Farray=[]
        

            
        if FandSigma_cMode=='LR':
            for l in range(npatch):        
                histogramming(sourcepath='/home/astro/aliqoliz/Documents/IA_pipeline/DESzmean'+str(z)+'_meanShift_'+str(photozMean)+'.fits',
                 lenspath='/home/astro/aliqoliz/Documents/IA_tomography_results/Patches_'+str(photozMean)+'/randPatch'+str(i)+'.fits'
                , z=z, FandSigma_cMode='LR', redshiftBins_s=redshiftBins_s,redshiftBins_l=redshiftBins_l,weights=weights, label=label, photozNoise=photozNoise,photozDev=photozDev,
                photozMean=photozMean, k=l)
            for i in range(npatch):
                LhistInPatch=np.loadtxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/lensInBin'+str(i))
                Lhist+=LhistInPatch
                pathL.append('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/lensInBin'+str(i))        
            ShistInPatch=np.loadtxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/source'+str(z)+'InBin')
            Shist+=ShistInPatch    
            for j in range(npatch):
                lensDir=np.delete(pathL,j)
                fen = np.zeros(1)
                sen = np.zeros(1)
                fum=np.zeros(1)
                summ=np.zeros(1)
                lrHistL=np.zeros(1)
                lrHistS=np.zeros(1)
                for path in lensDir:
                    a=np.loadtxt(path)
                    lrHistL+=a
                NL=np.sum(a)
                NS=np.sum(Shist)
                fen +=  NL*NS
                sen += NL*NS                
            if z==0:
                fum += np.matmul(lrHistL,np.matmul(FMap0,Shist))
                summ += np.matmul(lrHistL,np.matmul(SigMap0,Shist))
                f0=np.divide(fum, fen)
                np.append(F0s,f0)
                s0=np.divide(sen, summ)
                np.append(Sig0s, s0)
            if z==1:
                fum += np.matmul(lrHistL,np.matmul(FMap1,Shist))
                summ += np.matmul(lrHistL,np.matmul(SigMap1,Shist))
                f1=np.divide(fum, fen)
                np.append(F1s,f1)
                s1=np.divide(sen,summ)
                np.append(Sig1s, s1)
                
                
                
        if FandSigma_cMode=='SLR':
            for l in range(npatch):        
                histogramming(sourcepath='/home/astro/aliqoliz/Documents/IA_tomography_results/Patches_'+str(photozMean)+'/source'+str(z)+'Patch'+str(l)+'.fits',
                 lenspath='/home/astro/aliqoliz/Documents/IA_tomography_results/Patches_'+str(photozMean)+'/lensPatch'+str(l)+'.fits',
                  z=z, FandSigma_cMode='SLR', redshiftBins_s=redshiftBins_s,redshiftBins_l=redshiftBins_l, weights=weights, photozNoise=photozNoise, label=label, photozDev=photozDev
                  ,photozMean=photozMean ,k=l)
            for i in range(npatch):
                LhistInPatch=np.loadtxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/lensInBin'+str(i))
                Lhist+=LhistInPatch
                ShistInPatch=np.loadtxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/source'+str(z)+'InBin'+str(i))
                Shist+=ShistInPatch    
                pathL.append('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/lensInBin'+str(i))    
                pathS.append('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/source'+str(z)+'InBin'+str(i))                                    
            for j in range(npatch):
                lensDir=np.delete(pathL,j)
                srDir=np.delete(pathS,j)
                fen = np.zeros(1)
                sen = np.zeros(1)
                fum=np.zeros(1)
                summ=np.zeros(1)
                NS=np.zeros(1)
                NL=np.zeros(1)
                slrHistL=np.zeros(redshiftBins_l-1)
                slrHistS=np.zeros(redshiftBins_s-1)
#                slrHistL=np.zeros(redshiftBins)
#                slrHistS=np.zeros(redshiftBins)
                for k in range(npatch-1):
                    a=np.loadtxt(lensDir[k])
                    slrHistL+=a
                    b=np.loadtxt(srDir[k])
                    slrHistS+=b                    
                    NL+=np.sum(a)
                    NS+=np.sum(b)
                fen +=  NL*NS
                sen += NL*NS                
                if z==0:
                    fum += np.matmul(slrHistL,np.matmul(FMap0,slrHistS))
                    summ += np.matmul(slrHistL,np.matmul(SigMap0,slrHistS))
                    f0=np.divide(fum, fen)
                    F0s.append(f0)
                    s0=np.divide(sen, summ)
                    Sig0s.append(s0)                
                if z==1:
                    fum += np.matmul(slrHistL,np.matmul(FMap1,slrHistS))
                    summ += np.matmul(slrHistL,np.matmul(SigMap1,slrHistS))
                    f1=np.divide(fum, fen)
                    F1s.append(f1)
                    s1=np.divide(sen,summ)
                    Sig1s.append(s1)                
            print('F1s:',F1s)
            print('F0s:',F0s)
            print('S1s:', Sig1s)
            print('S0s:', Sig0s)
            
        else:
            fum=np.zeros(1)
            summ=np.zeros(1)
            fen=np.zeros(1)
            f00=[]
            f11=[]
            sen=np.zeros(1)
            histogramming(sourcepath='/home/astro/aliqoliz/Documents/IA_pipeline/DESzmean'+str(z)+'_meanShift_'+str(photozMean)+'.fits',
             lenspath='/home/astro/aliqoliz/Documents/IA_pipeline/rM345.fits',
              z=z,FandSigma_cMode =FandSigma_cMode, redshiftBins_s=redshiftBins_s,redshiftBins_l=redshiftBins_l, weights=weights, label=label,photozDev=photozDev,
              photozMean=photozMean , photozNoise=photozNoise)
            LhistInPatch=np.loadtxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/lensInBin')
            ShistInPatch=np.loadtxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/source'+str(z)+'InBin')            
            print(len(LhistInPatch))
            print(len(ShistInPatch))
            Lhist+=LhistInPatch
            Shist+=ShistInPatch
            NL=np.sum(Lhist)
            NS=np.sum(Shist)
            fen +=  NL*NS
            sen += NL*NS    
            print('sen:',sen)
            print('fen:',fen)        
            if z==0:
                if F_mode!='flat':
                    for x in FMap0:
                        fum==np.zeros(1)                            
                        fum += np.matmul(Lhist,np.matmul(x,Shist))
                        F00=fum/fen
                        f00.append(F00)
                    F0s=np.full((npatch, 10, 1),f00)
#                    print(F0s)
                else:        
                    fum += np.matmul(Lhist,np.matmul(FMap0,Shist))
                    F00=fum/fen
                    F0s=np.full(npatch,F00)
                summ += np.matmul(Lhist,np.matmul(SigMap0,Shist))
                Sig00=sen/summ
                Sig0s=np.full(npatch,Sig00)    
#                print(F0s)
#                print(Sig0s)                
            if z==1:
                if F_mode!='flat':
                    for x in FMap1:
                        fum==np.zeros(1)                            
                        fum += np.matmul(Lhist,np.matmul(x,Shist))
                        F01=fum/fen
                        f11.append(F01)
                    F1s=np.full((npatch, 10, 1),f11)             
                else:            
                    fum += np.matmul(Lhist,np.matmul(FMap1,Shist))
                    F01=fum/fen
                    F1s=np.full(npatch,F01)
                summ += np.matmul(Lhist,np.matmul(SigMap1,Shist))
                
                Sig01=sen/summ
                
                Sig1s=np.full(npatch,Sig01)    
#                print(F1s)
#                print(Sig1s)
        
        Nl= np.sum(Lhist)
        Ns= np.sum(Shist)
        fden +=  Nl*Ns
        if z==0:
            if F_mode == 'flat':
                fnum += np.matmul(Lhist,np.matmul(FMap0,Shist))
            else:
                for x in FMap0:
                    fnum = np.zeros(1)
                    fnum += np.matmul(Lhist,np.matmul(x,Shist))
                    
                    Farray.append(np.divide(fnum,fden))
                    np.savetxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/AvgF'+str(z),Farray)
            snum += np.matmul(Lhist,np.matmul(SigMap0,Shist))
        if z==1:
            if F_mode=='flat':            
                fnum += np.matmul(Lhist,np.matmul(FMap1,Shist))
            else:
                for x in FMap1:
                    fnum = np.zeros(1)
                    fnum += np.matmul(Lhist,np.matmul(x,Shist))
                    
                    Farray.append(np.divide(fnum,fden))
                    np.savetxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/AvgF'+str(z),Farray)
            snum += np.matmul(Lhist,np.matmul(SigMap1,Shist))
            
        
        sden += Nl*Ns
        if F_mode=='flat':
            F+=np.divide(fnum,fden)
            np.savetxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/AvgF'+str(z),F)
        Sig+=np.divide(sden, snum)    
        
        np.savetxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/AvgSigInv'+str(z),Sig)    
        
        if boostCorrectionMode == 'once':
            
            data_lens = fits.open(lenspath, memmap= True)
            data_lens.info()
            tableL = Table(data_lens[1].data)
            data_rand = fits.open(randpath, memmap= True)
            data_rand.info()
            tableR = Table(data_rand[1].data)
            bc= len(tableR['z'])/len(tableL['z'])
            print(bc)
            boostCorrection=np.full(npatch,bc)
            
        if boostCorrectionMode != 'once':
            boostCorrection=[]*(npatch)
            for i in range(npatch):
                data_lens = fits.open('/home/astro/aliqoliz/Documents/IA_tomography_results/Patches_'+str(photozMean)+'/lensPatch'+str(i)+'.fits', memmap= True)
                data_lens.info()
                tableL = Table(data_lens[1].data)                
                data_rand = fits.open('/home/astro/aliqoliz/Documents/IA_tomography_results/Patches_'+str(photozMean)+'/randPatch'+str(i)+'.fits', memmap= True)
                data_rand.info()
                tableR = Table(data_rand[1].data)    
                ll=len(tableL['z'])
                lr=len(tableR['z'])
                print(lr)
                print(ll)
                A= lr/ll
                print(A)
                boostCorrection.append(A)

                        
        if Mode == 'LR':
            shearPosLR(npatch=npatch, z=z, source_path=sourcepath[z], smin=minSep,smax=maxSep, nbins=nbins, weights=weights,kmeans=kmeans, label=label, photozMean=photozMean)

        if Mode == 'SLR':
            shearPosSLR(npatch=npatch, z=z, smin=minSep,smax=maxSep, nbins=nbins, weights=weights, kmeans=kmeans, label=label)
        
        if Mode == 'classical':
            shearPosClassic(npatch=npatch, smin=minSep, smax=maxSep, nbins=nbins, z=z, kmeans=kmeans, label=label)      
        
    if Mode != 'classical':    
        for z in zList:
            gammaT=np.zeros(nbins)
            w_ls=np.zeros(nbins)
            w_rs=np.zeros(nbins)
            w_b=np.zeros(nbins)
#            print('boostCorrection:', boostCorrection)
            for i in range(npatch):
#                bpath='/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/RG_345_'+str(z)+'_patch'+str(i)
#                shpath='/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/NG_345_'+str(z)+'_patch'+str(i)
#                kpath='/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/KG_345_'+str(z)+'_patch'+str(i)
               
                bpath='/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/cosmology_72_OmegaM/RG_345_'+str(z)+'_patch'+str(i)
                shpath='/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/cosmology_72_OmegaM/NG_345_'+str(z)+'_patch'+str(i)
                kpath='/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/cosmology_72_OmegaM/KG_345_'+str(z)+'_patch'+str(i)
                barray=np.loadtxt(bpath)
                sharray=np.loadtxt(shpath)
#                gammaT+= np.multiply(sharray[:,3], sharray[:,6])
                karray=np.loadtxt(kpath)
                gammaSens=np.divide(sharray[:,3],karray[:,3])
                gammaT+=np.multiply(gammaSens,sharray[:,6])
                w_ls+=sharray[:,6]
                w_b+=boostCorrection[i]*sharray[:,6]
                w_rs+=barray[:,6]
            boost=np.divide(w_b,w_rs)
            shear=np.divide(gammaT,w_ls)    
            np.savetxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/AvgGammaT'+str(z),shear)
            np.savetxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/AvgBoost'+str(z),boost)        
        for j in range(npatch):
#            bpath='/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/RG_345_'+str(z)+'_patch'+str(j)
#            shpath='/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/NG_345_'+str(z)+'_patch'+str(j)
#            kpath='/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/KG_345_'+str(z)+'_patch'+str(j)
            
            bpath='/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/cosmology_72_OmegaM/RG_345_'+str(z)+'_patch'+str(j)
            shpath='/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/cosmology_72_OmegaM/NG_345_'+str(z)+'_patch'+str(j)
            kpath='/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/cosmology_72_OmegaM/KG_345_'+str(z)+'_patch'+str(j)
            
            barray=np.loadtxt(bpath)
            sharray=np.loadtxt(shpath)
            karray=np.loadtxt(kpath)
#            gammaT+= np.multiply(sharray[:,3], sharray[:,6])
            gammaSens=np.divide(sharray[:,3],karray[:,3])
            gammaT+=np.multiply(gammaSens, sharray[:,6])
            w_ls+=boostCorrection[j]*sharray[:,6]
            w_rs+=barray[:,6]    
        boost=np.divide(w_ls,w_rs)
        shear=np.divide(gammaT,w_ls)
        for z in zList:
            bpaths=[]
            shpaths=[]
            kpaths=[]
            for i in range(npatch):
                bpaths.append('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/cosmology_72_OmegaM/RG_345_'+str(z)+'_patch'+str(i))
                shpaths.append('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/cosmology_72_OmegaM/NG_345_'+str(z)+'_patch'+str(i))
                kpaths.append('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/cosmology_72_OmegaM/KG_345_'+str(z)+'_patch'+str(i))
                indices=list(range(npatch))
            for i in range(npatch):
                boosts=np.delete(bpaths,i)
                shears=np.delete(shpaths,i)
                ks=np.delete(kpaths,i)
                ind=np.delete(indices,i)
                N=np.loadtxt(boosts[0])
                nbin=len(N[:,0])
                w_ls_tot=np.zeros(nbins)
                w_rs_tot=np.zeros(nbins)
                w_ls_b=np.zeros(nbins)
                shear_tot=np.zeros(nbins)
                counter=0
                for file_path in shears:
                    datash=np.loadtxt(file_path)
                    w_ls_tot+=datash[:,6]
                    datak=np.loadtxt(ks[counter])
                    gammaSensInBin=np.divide(datash[:,3],datak[:,3])
#                    gammaTInBin=np.multiply(datash[:,3], datash[:,6])
                    gammaTInBin=np.multiply(gammaSensInBin, datash[:,6])
                    shear_tot+=gammaTInBin
                    w_ls_b+=datash[:,6]*boostCorrection[ind[counter]]
                    counter+=1
                for file_path in boosts:

                    datab=np.loadtxt(file_path)
                    w_rs_tot+=datab[:,6]

                gammaT_tot=np.divide(shear_tot, w_ls_tot)
                Boost=np.divide(w_ls_b, w_rs_tot)            
                np.savetxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/'+str(z)+'BoostInPatch_without'+str(i),Boost)
                np.savetxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/'+str(z)+'GammaTinPatch_without'+str(i),gammaT_tot)
        
    for i in range(npatch):
        gamma0 = np.loadtxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/'+str(zList[0])+'GammaTinPatch_without'+str(i))        
        gamma1 = np.loadtxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/'+str(zList[1])+'GammaTinPatch_without'+str(i))        
        Boost0 = np.loadtxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/'+str(zList[0])+'BoostInPatch_without'+str(i))    
        Boost1 = np.loadtxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/'+str(zList[1])+'BoostInPatch_without'+str(i))
        F0=np.loadtxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/AvgF0')
        F1=np.loadtxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/AvgF1')      

#        estimator = []
#        for j in range(nbins):
#            numerator = gamma1[j]*Sig1s[i]-gamma0[j]*Sig0s[i]
#            denominator = (Boost1[j]-1+F1s[i])*Sig1s[i]-(Boost0[j]-1+F0s[i])*Sig0s[i]
#            estimator.append(numerator/denominator)
        print('F1: ',F1)
        print('F0: ',F0)      
        numerator=gamma1*Sig1s[i]-gamma0*Sig0s[i]
        denominator=(Boost1-1+F1)*Sig1s[i]-(Boost0-1+F0)*Sig0s[i]
        
        estimator=np.divide(numerator, denominator)
        np.savetxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/GammaIA'+str(i),estimator)    
    sigma=[]
    for k in range(nbins):
        a=[]        
        for i in range(npatch):        
            A = np.loadtxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/GammaIA'+str(i))        
            a.append(A[k])
        x=np.sqrt((npatch-1)*np.var(a))
        sigma.append(x)
    np.savetxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/ErrorBar',sigma)
    if Mode=='classical':
        Sig0=np.loadtxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/AvgSigInv0')
        Sig1=np.loadtxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/AvgSigInv1')
        F0=np.loadtxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/AvgF0')
        F1=np.loadtxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/AvgF1')
        errorBarClassic(npatch=npatch, zList=zList, F0=F0, F1=F1, Sig1=Sig1, Sig0=Sig0, label=label, boostCorrection=boostCorrection)
        
        
    boost0=np.loadtxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/AvgBoost0')
    boost1=np.loadtxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/AvgBoost1')
    shear0=np.loadtxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/AvgGammaT0')
    shear1=np.loadtxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/AvgGammaT1')
    Sig0=np.loadtxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/AvgSigInv0')
    Sig1=np.loadtxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/AvgSigInv1')
    F0=np.loadtxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/AvgF0')
    F1=np.loadtxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/AvgF1')
    print('F0: ',F0)
    print('F1: ',F1)
    numerator= np.multiply(shear0, Sig0)-np.multiply(shear1, Sig1)
    denominator= np.multiply(boost0-1+F0, Sig0) - np.multiply(boost1-1+F1, Sig1)
    gamma_IA=np.divide(numerator, denominator)
    np.savetxt('/home/astro/aliqoliz/Documents/IA_tomography_results/ConsistentCosmo/'+str(label)+'/GammaIAfinal', gamma_IA)

    
    Cov, Cor, diag, chi2=CovCorrMat(npatch=npatch, nbins=nbins, label=label)
    return gamma_IA, diag, chi2
    
lensPath = '/home/astro/aliqoliz/Documents/IA_pipeline/rM345.fits'
randPath = '/home/astro/aliqoliz/Documents/IA_pipeline/rM345rand.fits'

#########################################################################################
####################################### TODO ############################################
#########################################################################################
sourcePath = ['/home/astro/aliqoliz/Documents/IA_pipeline/DESzmean0_meanShift_0.fits','/home/astro/aliqoliz/Documents/IA_pipeline/DESzmean1_meanShift_0.fits']

#sourcePath=tableMaker(photozMean=0)

IAsignal, errorbars, chi2=analysis(patchRun=False, kmeans=False, npatch=40, nbins=10, minSep=0.2, maxSep=180, weights= 'ones',
redshiftBins_s =10001,redshiftBins_l=1001, F_mode='nonflat', F_min_threshold=2, comovingLimit=100 ,Mode='LR', FandSigma_cMode='once', boostCorrectionMode='once', H0=67.32, omegaM=0.3158, omegaDE=1.-0.3158,
 photozNoise=False, photozMean =0, photozDev=0, sourcepath=sourcePath, lenspath=lensPath, randpath=randPath, label='Planck18_w1')


