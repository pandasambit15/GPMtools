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

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

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
        
    ##Add Z at 3km Ku
    Z = NS['zFactorCorrected']

    x2 = 2. * 17
    re = 6378.
    theta = -1 *(x2/2.) + (x2/48.)*np.arange(0,49)
    theta = theta * (np.pi/180.)
    prh = np.zeros([176,49])
    for i in np.arange(0,175):
        for j in np.arange(0,49):
            a = np.arcsin(((re+407)/re)*np.sin(theta[j]))-theta[j]

            prh[i,j] = (176-i)*0.125*np.cos(theta[j]+a)


    Z_3km = np.zeros(Z.shape[0:2])
    for j in np.arange(0,prh.shape[1]):
        temp = prh[:,j]
        ind = find_nearest(temp,3)

        Z_3km[:,j] = np.squeeze(Z[:,j,ind])
    NS['zFactorCorrected_3km'] = Z_3km
    
    NS['Height'] = prh
     ##Add Z at 3km Ka
    Z = MS['zFactorCorrected']
    x2 = 2. * 8.5
    re = 6378.
    theta = -1 *(x2/2.) + (x2/24.)*np.arange(0,25)
    theta = theta * (np.pi/180.)
    prh = np.zeros([176,25])
    for i in np.arange(0,175):
        for j in np.arange(0,25):
            a = np.arcsin(((re+407)/re)*np.sin(theta[j]))-theta[j] #orbital height == 407 km
            prh[i,j] = (176-i)*0.125*np.cos(theta[j]+a)
    Z_3km = np.zeros(Z.shape[0:2])
    for j in np.arange(0,prh.shape[1]):
        temp = prh[:,j]
        ind = find_nearest(temp,3)

        Z_3km[:,j] = np.squeeze(Z[:,j,ind])
        
    MS['zFactorCorrected_3km'] = Z_3km
    
    MS['Height'] = prh
    
    
        
    GPM_DPR = {}
    GPM_DPR['NS_Ku'] = NS #Normal Ku scan
    GPM_DPR['MS_Ka'] = MS #Mached Ka to Ku scan
    GPM_DPR['HS_Ka'] = HS #High Sensativity scan Ka
    
    return GPM_DPR


def GPMDPR_planview(filename,camp=' ',savefig=False,Kuray = 23,
                    fontsize=14,fontsize2=12,vmin=-20,vmax=75,alpha= 0.8,
                    figsize=(9,4.75),lw = 3,cmap = 'NWSRef',zoom=1,lat_0 = '0',lon_0 = '90'):
    
    """

    =============
    
    Creates a plan view map of the GPM DPR Ku swath reflectiviy 
    
    =============
    
    """
    
    a1 = 400000.*zoom
    
    if camp == ' ' and lat_0 == 0 and lon_0 == 90: #check for basemap inputs
        print('please designate a campaign (camp = str) or lat_0 and lon_0 (lat_0 = str,lon_0=str)')
        return
    
    if camp == 'RELAMPAGO':
        if lat_0 == '0' and lon_0 == '90': #use defults
            lat_0 = '-31'
            lon_0 = '-60'
        area_def = pr.geometry.AreaDefinition('areaD', 'IPHEx', 'areaD',
            {'a': '6378144.0', 'b': '6356759.0','lat_0': lat_0, 'lat_ts': lat_0,'lon_0': lon_0, 'proj': 'stere'},400, 400,
            [-a1, -a1,a1, a1])
    elif camp == 'OLYMPEX':
         if lat_0 == '0' and lon_0 == '90': #use defults
            lat_0 = '47.6'
            lon_0 = '-124.5'
         area_def = pr.geometry.AreaDefinition('areaD', 'IPHEx', 'areaD',
            {'a': '6378144.0', 'b': '6356759.0','lat_0': lat_0, 'lat_ts': lat_0,'lon_0': lon_0, 'proj': 'stere'},400, 400,
            [-a1, -a1,a1, a1])
    else:
        area_def = pr.geometry.AreaDefinition('areaD', 'IPHEx', 'areaD',
        {'a': '6378144.0', 'b': '6356759.0','lat_0': lat_0, 'lat_ts': lat_0,'lon_0': lon_0, 'proj': 'stere'},400, 400,
        [-a1, -a1,a1, a1])

    fig=plt.figure(figsize=figsize)
    if cmap == 'NWSRef':
        cmap = cm.NWSRef
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
    
    bmap = pr.plot.area_def2basemap(area_def,resolution='l')
    col = bmap.imshow(result, origin='upper',vmin=vmin,vmax=vmax,cmap=cmap,zorder=10,alpha=alpha)
    bmap.drawcoastlines(linewidth=2)
    bmap.drawstates(linewidth=2)
    bmap.drawcountries(linewidth=2)

    #Map Stuff
    parallels = np.arange(-90.,90,zoom*2)
    bmap.drawparallels(parallels,labels=[1,0,0,0],fontsize=fontsize)
    meridians = np.arange(180.,360.,zoom*2)
    bmap.drawmeridians(meridians,labels=[0,0,0,1],fontsize=fontsize)
    bmap.drawmapboundary(fill_color='aqua')
    bmap.fillcontinents(color='coral',lake_color='aqua')
    #
    
    cbar = plt.colorbar()
    cbar.set_label('Reflectivity, $[dbZe]$',fontsize=fontsize,labelpad = 5)
    cax = cbar.ax
    cax.tick_params(labelsize=fontsize2)

    x,y=bmap(NS['Longitude'][:,Kuray],NS['Latitude'][:,Kuray])
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

def GPMDPR_profile(filename,camp =' ',band='Ku',savefig=False, Kuray = 23, Karay=11,
                   fontsize=14, fontsize2=12, vmin=-20,vmax=75,lw = 3, xmin=0, xmax=1000, 
                   ymin=0,ymax=12,cmap='NWSRef',zoom=1,figsize=(15,4),lat_0 = '0',lon_0 = '90'):
    
    a1 = 400000.*zoom
    
    if camp == ' ' and lat_0 == 0 and lon_0 == 90: #check for basemap inputs
        print('please designate a campaign (camp = str) or lat_0 and lon_0 (lat_0 = str,lon_0=str)')
        return
    
    if camp == 'RELAMPAGO':
        if lat_0 == '0' and lon_0 == '90': #use defults
            lat_0 = '-31'
            lon_0 = '-60'
        area_def = pr.geometry.AreaDefinition('areaD', 'IPHEx', 'areaD',
            {'a': '6378144.0', 'b': '6356759.0','lat_0': lat_0, 'lat_ts': lat_0,'lon_0': lon_0, 'proj': 'stere'},400, 400,
            [-a1, -a1,a1, a1])
    elif camp == 'OLYMPEX':
         if lat_0 == '0' and lon_0 == '90': #use defults
            lat_0 = '47.6'
            lon_0 = '-124.5'
         area_def = pr.geometry.AreaDefinition('areaD', 'IPHEx', 'areaD',
            {'a': '6378144.0', 'b': '6356759.0','lat_0': lat_0, 'lat_ts': lat_0,'lon_0': lon_0, 'proj': 'stere'},400, 400,
            [-a1, -a1,a1, a1])
    else:
        area_def = pr.geometry.AreaDefinition('areaD', 'IPHEx', 'areaD',
        {'a': '6378144.0', 'b': '6356759.0','lat_0': lat_0, 'lat_ts': lat_0,'lon_0': lon_0, 'proj': 'stere'},400, 400,
        [-a1, -a1,a1, a1])
    
    if cmap == 'NWSRef':
        cmap = cm.NWSRef
    
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
        height = NS['Height']
        fig,axes = plt.subplots(1,1,figsize=figsize)
        ax1 = axes
        ##Ku
        Z =np.transpose(np.squeeze(NS['zFactorCorrected'][indx,Kuray,:]))
        ind = np.where(Z <= vmin )
        Z[ind] = np.inf
        Z = np.ma.masked_invalid(Z, copy=True)

        pm = ax1.pcolormesh(5.*np.arange(len(indx[0])),height[:,Kuray],Z,cmap=cmap,
                            vmin=vmin,vmax=vmax)
        ax1.fill_between(5.*np.arange(len(indx[0])),.125*(176-np.squeeze(NS['binClutterFreeBottom'][indx,Kuray])),
                         color=[.5,.5,.5],alpha=.5)
        ax1.fill_between(5.*np.arange(len(indx[0])),.125*(176-np.squeeze(NS['binRealSurface'][indx,Kuray])),color='k')
        ax1.set_ylim([0,10])
        ax1.set_xlim,([0,max(5.*np.arange(len(indx[0])))])
        ax1.set_ylabel('Height, $[km]$',fontsize=fontsize)
        ax1.set_title('KuNS',fontsize=fontsize)
        ax1.set_xlim([xmin,xmax])
        ax1.set_ylim([ymin,ymax])
        ax1.tick_params(axis='both',direction='in',labelsize=fontsize2,width=2,length=5)

        cbar = plt.colorbar(pm,aspect=10,ax=ax1)
        cbar.set_label('Reflectivity, $[dBZ]$')
        cax = cbar.ax
        cax.tick_params(labelsize=fontsize2)
        ax1.set_xlabel('Distance, $[km]$',fontsize=fontsize)
        
    elif band == 'raw':
        
        fig,axes = plt.subplots(1,1,figsize=figsize)
        ax1 = axes
        ##Ku
        height = NS['Height']
        Z =np.transpose(np.squeeze(NS['zFactorMeasured'][indx,Kuray,:]))
        ind = np.where(Z <= vmin )
        Z[ind] = np.inf
        Z = np.ma.masked_invalid(Z, copy=True)

        pm = ax1.pcolormesh(5.*np.arange(len(indx[0])),height[:,Kuray],Z,cmap=cmap,
                            vmin=vmin,vmax=vmax)
        ax1.fill_between(5.*np.arange(len(indx[0])),.125*(176-np.squeeze(NS['binClutterFreeBottom'][indx,Kuray])),
                         color=[.5,.5,.5],alpha=.5)
        ax1.fill_between(5.*np.arange(len(indx[0])),.125*(176-np.squeeze(NS['binRealSurface'][indx,Kuray])),color='k')
        ax1.set_ylim([0,10])
        ax1.set_xlim,([0,max(5.*np.arange(len(indx[0])))])
        ax1.set_ylabel('Height, $[km]$',fontsize=fontsize)
        ax1.set_title('KuNS',fontsize=fontsize)
        ax1.set_xlim([xmin,xmax])
        ax1.set_ylim([ymin,ymax])
        ax1.tick_params(axis='both',direction='in',labelsize=fontsize2,width=2,length=5)

        cbar = plt.colorbar(pm,aspect=10,ax=ax1)
        cbar.set_label('Reflectivity, $[dBZ]$')
        cax = cbar.ax
        cax.tick_params(labelsize=fontsize2)
        ax1.set_xlabel('Distance, $[km]$',fontsize=fontsize)
        
        
    elif band=='Ka':
        fig,axes = plt.subplots(1,1,figsize=figsize)
        ax2 = axes
        ##Ka
        height = MS['Height']
        rayMS = 10
        Z =np.transpose(np.squeeze(MS['zFactorCorrected'][indx,Karay,:]))
        ind = np.where(Z <= vmin )
        Z[ind] = np.inf
        Z = np.ma.masked_invalid(Z, copy=True)

        pm= ax2.pcolormesh(5.*np.arange(len(indxMS[0])),height[:,Karay],Z,cmap=cmap,
                           vmin=vmin,vmax=vmax)
        ax2.fill_between(5.*np.arange(len(indxMS[0])),.125*(176-np.squeeze(MS['binClutterFreeBottom'][indxMS,Karay])),
                         color= [.5,.5,.5],alpha=0.5)
        ax2.fill_between(5.*np.arange(len(indxMS[0])),.125*(176-np.squeeze(MS['binRealSurface'][indxMS,Karay])),color='k')

        ax2.set_ylim([0,10])
        ax2.set_xlim,([0,max(5.*np.arange(len(indx[0])))])
        ax2.set_ylabel('Height, $[km]$',fontsize=fontsize)
        ax2.set_title('KaMS',fontsize=fontsize)
        ax2.set_xlim([xmin,xmax])
        ax2.set_ylim([ymin,ymax])
        ax2.tick_params(axis='both',direction='in',labelsize=fontsize2,width=2,length=5)

        cbar = plt.colorbar(pm,aspect=10,ax=ax2)
        cbar.set_label('Reflectivity, $[dBZ]$')
        cax = cbar.ax
        cax.tick_params(labelsize=fontsize2)
        ax2.set_xlabel('Distance, $[km]$',fontsize=fontsize)

    elif band=='KuKa':
        fig,axes = plt.subplots(2,1,figsize=figsize)
        ax1 = axes[0]
        ax2 = axes[1]
        
        ##Ku
        Z =np.transpose(np.squeeze(NS['zFactorCorrected'][indx,Kuray,:]))
        ind = np.where(Z <= vmin )
        Z[ind] = np.inf
        Z = np.ma.masked_invalid(Z, copy=True)

        pm = ax1.pcolormesh(5.*np.arange(len(indx[0])),height[:,Kuray],Z,cmap=cmap,
                            vmin=vmin,vmax=vmax)
        ax1.fill_between(5.*np.arange(len(indx[0])),.125*(176-np.squeeze(NS['binClutterFreeBottom'][indx,Kuray])),
                         color=[.5,.5,.5],alpha=.5)
        ax1.fill_between(5.*np.arange(len(indx[0])),.125*(176-np.squeeze(NS['binRealSurface'][indx,Kuray])),color='k')
        ax1.set_ylim([0,10])
        ax1.set_xlim,([0,max(5.*np.arange(len(indx[0])))])
        ax1.set_ylabel('Height, $[km]$',fontsize=fontsize)
        ax1.set_title('KuNS',fontsize=fontsize)
        ax1.set_xlim([xmin,xmax])
        ax1.set_ylim([ymin,ymax])
        ax1.tick_params(axis='both',direction='in',labelsize=fontsize2,width=2,length=5)

        cbar = plt.colorbar(pm,aspect=10,ax=ax1)
        cbar.set_label('Reflectivity, $[dBZ]$')
        cax = cbar.ax
        cax.tick_params(labelsize=fontsize2)
        
        ##Ka
        height = MS['Height']
        ax2 = axes[1]
        rayMS = 10
        Z =np.transpose(np.squeeze(MS['zFactorCorrected'][indx,Karay,:]))
        ind = np.where(Z <= vmin )
        Z[ind] = np.inf
        Z = np.ma.masked_invalid(Z, copy=True)

        pm = ax2.pcolormesh(5.*np.arange(len(indxMS[0])),height[:,Kuray],Z,cmap=cmap,
                            vmin=vmin,vmax=vmax)
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
    if savefig:
        print('Save file is: '+'Profile_'+band+camp+'.png')
        plt.savefig('Profile_'+band+camp+'.png',dpi=300)    
    plt.show()
        
    return 
