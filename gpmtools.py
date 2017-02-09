"""

=====================

This is a GPM DPR toolkit for reading and plotting

Developed by Stephen Nesbitt and Randy Chase at The University of Illinois, Urbana-Champaign, 2017

=====================

"""


import h5py
import numpy as np
import pyresample as pr
from pyresample import geometry, data_reduce
import matplotlib.pyplot as plt
from pyart.graph import cm

def GPMDPRread(filename):

    """
    ==============
    GPMDPRread reads in the standard GPM DPR radar file and returs a dictionary full of the datasets
    labeled. 
    

    ==============
    """

    f = h5py.File(filename,"r")
                
    fields=['SLV/zFactorCorrectedNearSurface','SLV/zFactorCorrected','PRE/zFactorMeasured',\
        'SRT/pathAtten','SLV/piaFinal','SLV/epsilon','PRE/binClutterFreeBottom','PRE/binRealSurface',\
        'CSF/flagBB','CSF/binBBPeak','CSF/binBBTop',\
        'CSF/binBBBottom','CSF/qualityBB','CSF/typePrecip','CSF/qualityTypePrecip',\
        'Longitude','Latitude']

    NS={}
    for i in fields:
        fullpath='/NS/'+i
        key=i.split('/')[-1]
        NS[key]=f[fullpath][:]
    HS={}
    for i in fields:
        fullpath='/HS/'+i
        key=i.split('/')[-1]
        HS[key]=f[fullpath][:]
    MS={}
    for i in fields:
        fullpath='/MS/'+i
        key=i.split('/')[-1]
        MS[key]=f[fullpath][:]
        
    ##Add Z at 3km (index 151 for Ku)
    Z = NS['zFactorCorrected']
    Z_3km = Z[:,:,151]
    Z_3km = np.squeeze(Z_3km)
    NS['zFactorCorrected_3km'] = Z_3km
    ##
        
    GPM_DPR = {}
    GPM_DPR['NS_Ku'] = NS #Normal Ku scan
    GPM_DPR['MS_Ka'] = MS #Mached Ka to Ku scan
    GPM_DPR['HS_Ka'] = HS #High Sensativity scan Ka
    
    return GPM_DPR


def GPMDPR_planview(filename,camp='RELAMPAGO',savefig=False,ray = 23,fontsize=14,fontsize2=12,vmin=-20,vmax=75,alpha= 0.8,lw = 3):
    
    """

    =============
    
    Creates a plan view map of the GPM DPR Ku swath reflectiviy 
    
    =============
    
    """
    if camp == 'RELAMPAGO':
        area_def = pr.geometry.AreaDefinition('areaD', 'IPHEx', 'areaD',
            {'a': '6378144.0', 'b': '6356759.0','lat_0': '-31', 'lat_ts': '31','lon_0': '-60', 'proj': 'stere'},400, 400,
            [-400000., -400000.,400000., 400000.])
    elif camp == 'OLYMPEX':
        area_def = pr.geometry.AreaDefinition('areaD', 'IPHEx', 'areaD',
            {'a': '6378144.0', 'b': '6356759.0','lat_0': '47.6', 'lat_ts': '47.6','lon_0': '-124.5', 'proj': 'stere'},400, 400,
            [-400000., -400000.,400000., 400000.])
    else:
        print('script not built for this campaign')
        return

    fig=plt.figure(figsize=(18,10.5))
   
    data = GPMDPRread(filename)
    NS = data['NS_Ku']
    grid_lons, grid_lats = area_def.get_lonlats()

    maxlon=grid_lons.max()
    maxlat=grid_lats.max()
    minlon=grid_lons.min()
    minlat=grid_lats.min()


    swath_def = pr.geometry.SwathDefinition(lons=NS['Longitude'], lats=NS['Latitude'])
    Z = data['NS_Ku']['zFactorCorrected_3km']
    Z[Z < vmin] = np.nan
    result=pr.kd_tree.resample_nearest(swath_def,Z, area_def,radius_of_influence=5000, fill_value=np.NAN)

    cmap = cm.NWSRef
    bmap = pr.plot.area_def2basemap(area_def,resolution='l')
    col = bmap.imshow(result, origin='upper',vmin=vmin,vmax=vmax,cmap=cmap,zorder=10,alpha=alpha)
    bmap.drawcoastlines(linewidth=2)
    bmap.drawstates(linewidth=2)
    bmap.drawcountries(linewidth=2)

    #Map Stuff
    parallels = np.arange(-90.,90,2)
    bmap.drawparallels(parallels,labels=[1,0,0,0],fontsize=fontsize)
    meridians = np.arange(180.,360.,2)
    bmap.drawmeridians(meridians,labels=[0,0,0,1],fontsize=fontsize)
    bmap.drawmapboundary(fill_color='aqua')
    bmap.fillcontinents(color='coral',lake_color='aqua')
    #
    
    cbar = plt.colorbar()
    cbar.set_label('Reflectivity, $[dbZe]$',fontsize=fontsize,labelpad = 5)
    cax = cbar.ax
    cax.tick_params(labelsize=fontsize2)

    x,y=bmap(NS['Longitude'][:,ray],NS['Latitude'][:,ray])
    plt.plot(x,y,'k--',lw=lw,zorder=11)
    x2,y2=bmap(NS['Longitude'][:,0],NS['Latitude'][:,0])
    plt.plot(x2,y2,'k-',lw=lw-.5,zorder=12)
    x3,y3=bmap(NS['Longitude'][:,48],NS['Latitude'][:,48])
    plt.plot(x3,y3,'k-',lw=lw-.5,zorder=13)
    plt.title('KuNS 3km Reflectivity',fontsize=fontsize+8)
    plt.xlabel('Longitude', fontsize=fontsize+8,labelpad=30)
    plt.ylabel('Latitude', fontsize=fontsize+8,labelpad=50)
    
    if savefig:
        print('Save file is: '+'3km_Ze'+camp+'.png')
        plt.savefig('3km_Ze'+camp+'.png',dpi=300)
    plt.show()
    
    return

def GPMDPR_profile(filename,camp ='OLYMPEX',band='Ku',savefig=False, Kuray = 23, Karay=12,  fontsize=14, fontsize2=12, vmin=-20,
                   vmax=75,lw = 3,xmin=0,xmax=1000,ymin=0,ymax=12,cmap='seismic'):

    if camp == 'RELAMPAGO':
        area_def = pr.geometry.AreaDefinition('areaD', 'IPHEx', 'areaD',
            {'a': '6378144.0', 'b': '6356759.0','lat_0': '-31', 'lat_ts': '31','lon_0': '-60', 'proj': 'stere'},400, 400,
            [-400000., -400000.,400000., 400000.])
    elif camp == 'OLYMPEX':
        area_def = pr.geometry.AreaDefinition('areaD', 'IPHEx', 'areaD',
            {'a': '6378144.0', 'b': '6356759.0','lat_0': '47.6', 'lat_ts': '47.6','lon_0': '-124.5', 'proj': 'stere'},400, 400,
            [-400000., -400000.,400000., 400000.])
    else:
        print('script not built for this campaign')
        return
    if cmap == 'NWSRef':
        cmap = cm.NWSRef
        
    fig=plt.figure(figsize=(18,10.5))
    
    data = GPMDPRread(filename)
    NS = data['NS_Ku']
    grid_lons, grid_lats = area_def.get_lonlats()

    maxlon=grid_lons.max()
    maxlat=grid_lats.max()
    minlon=grid_lons.min()
    minlat=grid_lats.min()
    
    
    
    data = GPMDPRread(filename)
    NS = data['NS_Ku']
    MS = data['MS_Ka']
    HS = data['HS_Ka']
    
    indx=np.where((NS['Latitude'][:,Kuray] > minlat) & (NS['Latitude'][:,Kuray] < maxlat) \
              & (NS['Longitude'][:,Kuray] > minlon) & (NS['Longitude'][:,Kuray] < maxlon))
    indxMS=np.where((MS['Latitude'][:,Karay] > minlat) & (MS['Latitude'][:,Karay] < maxlat) \
              & (MS['Longitude'][:,Karay] > minlon) & (MS['Longitude'][:,Karay] < maxlon))
    
    if band=='Ku':
        fig,axes = plt.subplots(1,1,figsize=(15,4))
        ax1 = axes
        ##Ku
        Z =np.transpose(np.squeeze(NS['zFactorCorrected'][indx,Kuray,:]))
        ind = np.where(Z <= vmin )
        Z[ind] = np.inf
        Z = np.ma.masked_invalid(Z, copy=True)

        pm = ax1.pcolormesh(5.*np.arange(len(indx[0])),.125*np.arange(175,-1,-1),Z,cmap=cmap,vmin=vmin,vmax=vmax)
        ax1.fill_between(5.*np.arange(len(indx[0])),.125*(176-np.squeeze(NS['binClutterFreeBottom'][indx,Kuray])),
                         color=[.5,.5,.5],alpha=.5)
        ax1.fill_between(5.*np.arange(len(indx[0])),.125*(176-np.squeeze(NS['binRealSurface'][indx,Kuray])),color='k')
        ax1.set_ylim([0,10])
        ax1.set_xlim,([0,max(5.*np.arange(len(indx[0])))])
        ax1.set_ylabel('Slant range from ellipsoid, $[km]$',fontsize=fontsize)
        ax1.set_title('KuNS',fontsize=fontsize)
        ax1.set_xlim([xmin,xmax])
        ax1.set_ylim([ymin,ymax])
        ax1.tick_params(axis='both',direction='in',labelsize=fontsize2,width=2,length=5)

        cbar = plt.colorbar(pm,aspect=10,ax=ax1)
        cbar.set_label('Reflectivity, $[dBZ]$')
        cax = cbar.ax
        cax.tick_params(labelsize=fontsize2)
        ax1.set_xlabel('Distance, $[km]$',fontsize=fontsize)
        plt.show()
        
    elif band=='Ka':
        fig,axes = plt.subplots(1,1,figsize=(15,4))
        
        
        ax2 = axes
        ##Ka
        rayMS = 10
        Z =np.transpose(np.squeeze(MS['zFactorCorrected'][indx,Karay,:]))
        ind = np.where(Z <= vmin )
        Z[ind] = np.inf
        Z = np.ma.masked_invalid(Z, copy=True)

        pm = ax2.pcolormesh(5.*np.arange(len(indxMS[0])),.125*np.arange(175,-1,-1),Z,cmap=cmap,vmin=vmin,vmax=vmax)
        ax2.fill_between(5.*np.arange(len(indxMS[0])),.125*(176-np.squeeze(MS['binClutterFreeBottom'][indxMS,Karay])),
                         color= [.5,.5,.5],alpha=0.5)
        ax2.fill_between(5.*np.arange(len(indxMS[0])),.125*(176-np.squeeze(MS['binRealSurface'][indxMS,Karay])),color='k')

        ax2.set_ylim([0,10])
        ax2.set_xlim,([0,max(5.*np.arange(len(indx[0])))])
        ax2.set_ylabel('Slant range from ellipsoid, $[km]$',fontsize=fontsize)
        ax2.set_title('KaMS',fontsize=fontsize)
        ax2.set_xlim([xmin,xmax])
        ax2.set_ylim([ymin,ymax])
        ax2.tick_params(axis='both',direction='in',labelsize=fontsize2,width=2,length=5)

        cbar = plt.colorbar(pm,aspect=10,ax=ax2)
        cbar.set_label('Reflectivity, $[dBZ]$')
        cax = cbar.ax
        cax.tick_params(labelsize=fontsize2)
        ax2.set_xlabel('Distance, $[km]$',fontsize=fontsize)
        plt.show()

    elif band=='KuKa':
        fig,axes = plt.subplots(2,1,figsize=(15,8))
        ax1 = axes[0]
        ax2 = axes[1]
        
        ##Ku
        Z =np.transpose(np.squeeze(NS['zFactorCorrected'][indx,Kuray,:]))
        ind = np.where(Z <= vmin )
        Z[ind] = np.inf
        Z = np.ma.masked_invalid(Z, copy=True)

        pm = ax1.pcolormesh(5.*np.arange(len(indx[0])),.125*np.arange(175,-1,-1),Z,cmap=cmap,vmin=vmin,vmax=vmax)
        ax1.fill_between(5.*np.arange(len(indx[0])),.125*(176-np.squeeze(NS['binClutterFreeBottom'][indx,Kuray])),
                         color=[.5,.5,.5],alpha=.5)
        ax1.fill_between(5.*np.arange(len(indx[0])),.125*(176-np.squeeze(NS['binRealSurface'][indx,Kuray])),color='k')
        ax1.set_ylim([0,10])
        ax1.set_xlim,([0,max(5.*np.arange(len(indx[0])))])
        ax1.set_ylabel('Slant range from ellipsoid, $[km]$',fontsize=fontsize)
        ax1.set_title('KuNS',fontsize=fontsize)
        ax1.set_xlim([xmin,xmax])
        ax1.set_ylim([ymin,ymax])
        ax1.tick_params(axis='both',direction='in',labelsize=fontsize2,width=2,length=5)

        cbar = plt.colorbar(pm,aspect=10,ax=ax1)
        cbar.set_label('Reflectivity, $[dBZ]$')
        cax = cbar.ax
        cax.tick_params(labelsize=fontsize2)
        
        ##Ka
        ax2 = axes[1]
        rayMS = 10
        Z =np.transpose(np.squeeze(MS['zFactorCorrected'][indx,Karay,:]))
        ind = np.where(Z <= vmin )
        Z[ind] = np.inf
        Z = np.ma.masked_invalid(Z, copy=True)

        pm = ax2.pcolormesh(5.*np.arange(len(indxMS[0])),.125*np.arange(175,-1,-1),Z,cmap=cmap,vmin=vmin,vmax=vmax)
        ax2.fill_between(5.*np.arange(len(indxMS[0])),.125*(176-np.squeeze(MS['binClutterFreeBottom'][indxMS,Karay])),
                         color= [.5,.5,.5],alpha=0.5)
        ax2.fill_between(5.*np.arange(len(indxMS[0])),.125*(176-np.squeeze(MS['binRealSurface'][indxMS,Karay])),color='k')

        ax2.set_ylim([0,10])
        ax2.set_xlim,([0,max(5.*np.arange(len(indx[0])))])
        ax2.set_ylabel('Slant range from ellipsoid, $[km]$',fontsize=fontsize)
        ax2.set_title('KaMS',fontsize=fontsize)
        ax2.set_xlim([xmin,xmax])
        ax2.set_ylim([ymin,ymax])
        ax2.tick_params(axis='both',direction='in',labelsize=fontsize2,width=2,length=5)

        cbar = plt.colorbar(pm,aspect=10,ax=ax2)
        cbar.set_label('Reflectivity, $[dBZ]$')
        cax = cbar.ax
        cax.tick_params(labelsize=fontsize2)
        ax2.set_xlabel('Distance, $[km]$',fontsize=fontsize)
        plt.tight_layout()
        plt.show()
        
    return
