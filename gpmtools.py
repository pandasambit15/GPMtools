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
from pylab import rc


##plot attributes 
rc('axes', linewidth=2.5)

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
    
    fields2=['SLV/zFactorCorrectedNearSurface','SLV/zFactorCorrected','PRE/zFactorMeasured',\
        'SRT/pathAtten','SLV/piaFinal','SLV/epsilon','PRE/binClutterFreeBottom','PRE/binRealSurface',\
        'CSF/flagBB','CSF/binBBPeak','CSF/binBBTop',\
        'CSF/binBBBottom','CSF/qualityBB','CSF/typePrecip','CSF/qualityTypePrecip',\
        'Longitude','Latitude','SLV/paramDSD','SLV/precipRate']
 
    NS={}
    for i in fields2:
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

def GPMDPRPrecip(filename):
    data ={}
    f = h5py.File(filename,"r")
    
    precip = f['Swath']['nearSurfacePrecipRate'][:]
    lat = f['Swath']['Latitude'][:]
    lon = f['Swath']['Longitude'][:]
    
    data['precip'] = precip
    data['lat'] = lat
    data['lon'] = lon
    
    return data


def GPMDPR_planview(filename,camp=' ',savefig=False,Kuray = 23, Karay=11,Crossray=0,xtrack=False,
                    fontsize=14,fontsize2=12,vmin=-20,vmax=75,alpha= 0.8,
                    figsize=(9,4.75),lw = 3,cmap = 'NWSRef',zoom=1,lat_0 = '0',
                    lon_0 = '90',kaswath=True,karayon=True,kurayon=True,precip=False,nearsurf=False):
    
    """

    =============
    
    Creates a plan view map of the GPM DPR Ku swath reflectiviy and near surface precip
    
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
        
    

    if precip == True:
        
        titlestr = 'Near Surface Precip'
        f = h5py.File(filename,"r")
        P = f['NS']['SLV']['precipRateNearSurface'][:]
        lat = f['NS']['Latitude'][:]
        lon = f['NS']['Longitude'][:]
        for i in np.arange(0,P.shape[1]):
            ind = np.where(P[:,i] <= 0)
            P[ind,i] = np.inf
        P = np.ma.masked_invalid(P, copy=True)
        swath_def = pr.geometry.SwathDefinition(lons=lon, lats=lat)
        result=pr.kd_tree.resample_nearest(swath_def,P, area_def,radius_of_influence=5000,
                                   fill_value=np.nan)
        bmap = pr.plot.area_def2basemap(area_def,resolution='l')
        col = bmap.imshow(result,origin='upper',cmap=cmap,vmin=vmin,vmax=vmax,zorder=10,alpha=alpha)

        bmap.drawcoastlines(linewidth=2)
        bmap.drawstates(linewidth=2)
        bmap.drawcountries(linewidth=2)
        parallels = np.arange(-90.,90,zoom*2)
        bmap.drawparallels(parallels,labels=[1,0,0,0],fontsize=fontsize)
        meridians = np.arange(180.,360.,zoom*2)
        bmap.drawmeridians(meridians,labels=[0,0,0,1],fontsize=fontsize)
        bmap.fillcontinents(lake_color='aqua')
        #
        cbar = plt.colorbar()
        cbar.set_label('Precipitation Rate, $[mm hr^{-1}]$',fontsize=fontsize,labelpad = 5)
        cax = cbar.ax
        cax.tick_params(labelsize=fontsize2)
        
        if kurayon:  
            #Ku ray chosen
            x,y=bmap(lon[:,Kuray],lat[:,Kuray])
            plt.plot(x,y,'k--',lw=lw,zorder=12)
            
        #Edge bins of Ku
        x2,y2=bmap(lon[:,0],lat[:,0])
        plt.plot(x2,y2,'k-',lw=lw-.5,zorder=12)
        x3,y3=bmap(lon[:,48],lat[:,48])
        l_ku, = plt.plot(x3,y3,'k-',lw=lw-.5,zorder=13)
    
    else:
        
        data = GPMDPRread(filename)
        NS = data['NS_Ku']
        MS = data['MS_Ka']
        grid_lons, grid_lats = area_def.get_lonlats()

        maxlon=grid_lons.max()
        maxlat=grid_lats.max()
        minlon=grid_lons.min()
        minlat=grid_lats.min()
        swath_def = pr.geometry.SwathDefinition(lons=NS['Longitude'], lats=NS['Latitude'])
        
        if nearsurf:
            titlestr = 'Near Surf. Ku Z'
            Z = data['NS_Ku']['zFactorCorrectedNearSurface']
        else:
            titlestr = 'Ku 3km Reflectivity'
            Z = data['NS_Ku']['zFactorCorrected_3km']
            
        Z[Z < vmin] = np.nan
        result=pr.kd_tree.resample_nearest(swath_def,Z, area_def,radius_of_influence=5000,
                                           fill_value=np.NAN)
    
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
        #bmap.drawmapboundary(fill_color='aqua')
        bmap.fillcontinents(lake_color='aqua')
        #

        cbar = plt.colorbar()
        cbar.set_label('Reflectivity, $[dBZ]$',fontsize=fontsize,labelpad = 5)
        cax = cbar.ax
        cax.tick_params(labelsize=fontsize2)

        if xtrack:
            indx=np.where((NS['Latitude'][:,Kuray] > minlat) & (NS['Latitude'][:,Kuray] < maxlat) \
              & (NS['Longitude'][:,Kuray] > minlon) & (NS['Longitude'][:,Kuray] < maxlon))

            if Crossray == 0:
                Crossray = indx[0][int(indx[0].shape[0]/2)]
                temp = indx[0]
                print('Please choose a Crossray between '+ str(temp.min())+' - '+str(temp.max()))

            indind = np.where(Crossray == indx)
            indind = np.where(indx[0] == Crossray) 

            if indind[0].shape[0] == 0:
                temp = indx[0]
                print('Please choose a cross section number between '+ str(temp.min())+' - '+ str(temp.max()))
                return

            x,y=bmap(NS['Longitude'][Crossray,:],NS['Latitude'][Crossray,:])
            plt.plot(x,y,'k--',lw=lw,zorder=20)
        if kurayon:  
            #Ku ray chosen
            x,y=bmap(NS['Longitude'][:,Kuray],NS['Latitude'][:,Kuray])
            plt.plot(x,y,'k--',lw=lw,zorder=12)

        if karayon:
            #Ka ray chosen
            x,y=bmap(MS['Longitude'][:,Karay],MS['Latitude'][:,Karay])
            plt.plot(x,y,'b:',lw=lw,zorder=13)

        #Edge bins of Ku
        x2,y2=bmap(NS['Longitude'][:,0],NS['Latitude'][:,0])
        plt.plot(x2,y2,'k-',lw=lw-.5,zorder=12)
        x3,y3=bmap(NS['Longitude'][:,48],NS['Latitude'][:,48])
        l_ku, = plt.plot(x3,y3,'k-',lw=lw-.5,zorder=13)
    
        if kaswath:
            #Edge bins of Ka 
            lon = MS['Longitude'][:,0]
            lat = MS['Latitude'][:,0]
            x4,y4=bmap(lon,lat)
            plt.plot(x4,y4,'k:',lw=lw-.5,zorder=11)
            x5,y5=bmap(MS['Longitude'][:,24],MS['Latitude'][:,24])
            l_ka, = plt.plot(x5,y5,'k:',lw=lw-.5,zorder=11)
            plt.legend([l_ku,l_ka],['Ku Swath','Ka Swath'],loc=1,fontsize=fontsize)
        else:
            plt.legend([l_ku],['Ku Swath'],loc=1,fontsize=fontsize)
    
    #plot Cordoba, Rosario
    c1,c2 = bmap(-64.1888,-31.4201)
    bmap.plot(c1,c2,'b*',ms=10,lw=2)
    c3,c4 = bmap(-60.6505,-32.9442)
    bmap.plot(c3,c4,'ko',ms=5)
    c5,c6 = bmap(-63.2667,-32.5833)
    bmap.plot(c5,c6,'ko',ms=5)
    
    plt.title(titlestr,fontsize=fontsize+8)
    plt.xlabel('Longitude', fontsize=fontsize+8,labelpad=30)
    plt.ylabel('Latitude', fontsize=fontsize+8,labelpad=60)
    
    if savefig:
        print('Save file is: '+'3km_Ze'+camp+'.png')
        plt.savefig('3km_Ze'+camp+'.png',dpi=300)
        
    plt.show()
    
    return

def GPMDPR_profile(filename,camp =' ',band='Ku',savefig=False, Kuray = 23, Karay=11,
                   fontsize=14, fontsize2=12, vmin=-20,vmax=75,lw = 3, xmin=0, xmax=1000, 
                   ymin=0,ymax=12,cmap='NWSRef',zoom=1,contour=False,figsize=(15,4),lat_0 = '0',lon_0 = '90'):
    
    """
    =================
    
    Creates profiles of: Ku, Ka, DFR, D0, N0 for a specific ray within the GPM swath
    
    =================
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
    
    if cmap == 'NWSRef':
        cmap = cm.NWSRef
    

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
        
        if contour:
                    pm = ax1.contourf(5.*np.arange(len(indx[0])),height[:,Kuray],Z,cmap=cmap,
                          vmin=vmin,vmax=vmax)
        else:
                    pm = ax1.pcolormesh(5.*np.arange(len(indx[0])),height[:,Kuray],Z,cmap=cmap,
                          vmin=vmin,vmax=vmax)
        
        inds = NS['binClutterFreeBottom'] - 1
        ax1.fill_between(5.*np.arange(len(indx[0])),height[inds[indx[0],Kuray],Kuray],
                          color=[.5,.5,.5],alpha=.5)
        inds2 = NS['binRealSurface'] - 1
        ax1.fill_between(5.*np.arange(len(indx[0])),height[inds2[indx[0],Kuray],Kuray],color='k')
        
        ax1.set_ylim([0,10])
        ax1.set_xlim,([0,max(5.*np.arange(len(indx[0])))])
        ax1.set_ylabel('Height, $[km]$',fontsize=fontsize)
        ax1.set_title('KuNS',fontsize=fontsize)
        ax1.set_xlim([xmin,xmax])
        ax1.set_ylim([ymin,ymax])
        ax1.tick_params(axis='both',direction='in',labelsize=fontsize2,width=2,length=5)

        cbar = plt.colorbar(pm,aspect=10,ax=ax1)
        cbar.set_label('Reflectivity, $[dBZ]$',fontsize=fontsize)
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

        
        if contour:
                    pm = ax1.contourf(5.*np.arange(len(indx[0])),height[:,Kuray],Z,cmap=cmap,
                          vmin=vmin,vmax=vmax)
        else:
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
        cbar.set_label('Reflectivity, $[dBZ]$',fontsize=fontsize)
        cax = cbar.ax
        cax.tick_params(labelsize=fontsize2)
        ax1.set_xlabel('Distance, $[km]$',fontsize=fontsize)
        
        
    elif band=='Ka':
        
        fig,axes = plt.subplots(1,1,figsize=figsize)
        ax2 = axes
        ##Ka
        height = MS['Height']
        Z =np.transpose(np.squeeze(MS['zFactorCorrected'][indxMS,Karay,:]))
        ind = np.where(Z <= vmin )
        Z[ind] = np.inf
        Z = np.ma.masked_invalid(Z, copy=True)

        pm= ax2.pcolormesh(5.*np.arange(len(indxMS[0])),height[:,Karay],Z,cmap=cmap,
                           vmin=vmin,vmax=vmax)
        
        if contour:
                    pm = ax2.contourf(5.*np.arange(len(indxMS[0])),height[:,Karay],Z,cmap=cmap,
                           vmin=vmin,vmax=vmax)
        else:
                    pm = ax2.pcolormesh(5.*np.arange(len(indxMS[0])),height[:,Karay],Z,cmap=cmap,
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
        cbar.set_label('Reflectivity, $[dBZ]$',fontsize=fontsize)
        cax = cbar.ax
        cax.tick_params(labelsize=fontsize2)
        ax2.set_xlabel('Distance, $[km]$',fontsize=fontsize)
        
    elif band == 'N0':
        
        height = NS['Height']
        fig,axes = plt.subplots(1,1,figsize=figsize)
        ax1 = axes
        ##Ku
        Z =np.transpose(np.squeeze(NS['paramDSD'][indx,Kuray,:,0]))
        ind = np.where(Z <= vmin)
        Z[ind] = np.inf
        Z = np.ma.masked_invalid(Z, copy=True)
        
        if contour:
                    pm = ax1.contourf(5.*np.arange(len(indx[0])),height[:,Kuray],Z,cmap=cmap,
                          vmin=vmin,vmax=vmax)
        else:
                    pm = ax1.pcolormesh(5.*np.arange(len(indx[0])),height[:,Kuray],Z,cmap=cmap,
                          vmin=vmin,vmax=vmax)

        ax1.fill_between(5.*np.arange(len(indx[0])),.125*(176-np.squeeze(NS['binClutterFreeBottom'][indx,Kuray])),
                         color=[.5,.5,.5],alpha=.5)
        ax1.fill_between(5.*np.arange(len(indx[0])),.125*(176-np.squeeze(NS['binRealSurface'][indx,Kuray])),color='k')
        ax1.set_ylim([0,10])
        ax1.set_xlim,([0,max(5.*np.arange(len(indx[0])))])
        ax1.set_ylabel('Height, $[km]$',fontsize=fontsize)
        ax1.set_title('N0',fontsize=fontsize)
        ax1.set_xlim([xmin,xmax])
        ax1.set_ylim([ymin,ymax])
        ax1.tick_params(axis='both',direction='in',labelsize=fontsize2,width=2,length=5)

        cbar = plt.colorbar(pm,aspect=10,ax=ax1)
        cbar.set_label('N0, $[?]$',fontsize=fontsize)
        cax = cbar.ax
        cax.tick_params(labelsize=fontsize2)
        ax1.set_xlabel('Distance, $[km]$',fontsize=fontsize)
        
    elif band == 'D0':
        
        height = NS['Height']
        fig,axes = plt.subplots(1,1,figsize=figsize)
        ax1 = axes
        ##Ku
        Z =np.transpose(np.squeeze(NS['paramDSD'][indx,Kuray,:,1]))
        ind = np.where(Z <= vmin)
        Z[ind] = np.inf
        Z = np.ma.masked_invalid(Z, copy=True)
        
        if contour:
                    pm = ax1.contourf(5.*np.arange(len(indx[0])),height[:,Kuray],Z,cmap=cmap,
                          vmin=vmin,vmax=vmax)
        else:
                    pm = ax1.pcolormesh(5.*np.arange(len(indx[0])),height[:,Kuray],Z,cmap=cmap,vmin=vmin,vmax=vmax)

        ax1.fill_between(5.*np.arange(len(indx[0])),.125*(176-np.squeeze(NS['binClutterFreeBottom'][indx,Kuray])),
                         color=[.5,.5,.5],alpha=.5)
        ax1.fill_between(5.*np.arange(len(indx[0])),.125*(176-np.squeeze(NS['binRealSurface'][indx,Kuray])),color='k')
        ax1.set_ylim([0,10])
        ax1.set_xlim,([0,max(5.*np.arange(len(indx[0])))])
        ax1.set_ylabel('Height, $[km]$',fontsize=fontsize)
        ax1.set_title('D0',fontsize=fontsize)
        ax1.set_xlim([xmin,xmax])
        ax1.set_ylim([ymin,ymax])
        ax1.tick_params(axis='both',direction='in',labelsize=fontsize2,width=2,length=5)

        cbar = plt.colorbar(pm,aspect=10,ax=ax1)
        cbar.set_label('D0, $[cm?]$',fontsize=fontsize)
        cax = cbar.ax
        cax.tick_params(labelsize=fontsize2)
        ax1.set_xlabel('Distance, $[km]$',fontsize=fontsize)
        
    elif band == 'precipRate':
        
        height = NS['Height']
        fig,axes = plt.subplots(1,1,figsize=figsize)
        ax1 = axes
        ##Ku
        Z =np.transpose(np.squeeze(NS['precipRate'][indx,Kuray,:]))
        ind = np.where(Z <= vmin)
        Z[ind] = np.inf
        Z = np.ma.masked_invalid(Z, copy=True)
        
        if contour:
                    pm = ax1.contourf(5.*np.arange(len(indx[0])),height[:,Kuray],Z,cmap=cmap,
                          vmin=vmin,vmax=vmax)
        else:
                    pm = ax1.pcolormesh(5.*np.arange(len(indx[0])),height[:,Kuray],Z,cmap=cmap,vmin=vmin,vmax=vmax)

        ax1.fill_between(5.*np.arange(len(indx[0])),.125*(176-np.squeeze(NS['binClutterFreeBottom'][indx,Kuray])),
                         color=[.5,.5,.5],alpha=.5)
        ax1.fill_between(5.*np.arange(len(indx[0])),.125*(176-np.squeeze(NS['binRealSurface'][indx,Kuray])),color='k')
        ax1.set_ylim([0,10])
        ax1.set_xlim,([0,max(5.*np.arange(len(indx[0])))])
        ax1.set_ylabel('Height, $[km]$',fontsize=fontsize)
        ax1.set_title('R',fontsize=fontsize)
        ax1.set_xlim([xmin,xmax])
        ax1.set_ylim([ymin,ymax])
        ax1.tick_params(axis='both',direction='in',labelsize=fontsize2,width=2,length=5)

        cbar = plt.colorbar(pm,aspect=10,ax=ax1)
        cbar.set_label('R, $[mm hr^{-1}]$',fontsize=fontsize)
        cax = cbar.ax
        cax.tick_params(labelsize=fontsize2)
        ax1.set_xlabel('Distance, $[km]$',fontsize=fontsize)
        
    elif band=='DFR':
        
        if Kuray - Karay != 12:
            print('Default matching to designated Ku ray')
            Karay = Kuray - 12
            indxMS=np.where((MS['Latitude'][:,Karay] > minlat) & (MS['Latitude'][:,Karay] < maxlat) \
          & (MS['Longitude'][:,Karay] > minlon) & (MS['Longitude'][:,Karay] < maxlon))
            
        fig,axes = plt.subplots(1,1,figsize=figsize)
        ax2 = axes
        ##Ku 
        Z =np.transpose(np.squeeze(NS['zFactorCorrected'][indx,Kuray,:]))
        ind = np.where(Z <= vmin )
        Z[ind] = np.inf
        Z2 = np.ma.masked_invalid(Z, copy=True)
        ##Ka
        height = MS['Height']
        Z =np.transpose(np.squeeze(MS['zFactorCorrected'][indxMS,Karay,:]))
        ind = np.where(Z <= vmin )
        Z[ind] = np.inf
        Z = np.ma.masked_invalid(Z, copy=True)
        
        DFR = Z2 - Z
        
        pm= ax2.pcolormesh(5.*np.arange(len(indxMS[0])),height[:,Karay],DFR,cmap=cmap,
                           vmin=vmin,vmax=vmax)
        ax2.fill_between(5.*np.arange(len(indxMS[0])),.125*(176-np.squeeze(MS['binClutterFreeBottom'][indxMS,Karay])),
                         color= [.5,.5,.5],alpha=0.5)
        ax2.fill_between(5.*np.arange(len(indxMS[0])),.125*(176-np.squeeze(MS['binRealSurface'][indxMS,Karay])),color='k')

        ax2.set_ylim([0,10])
        ax2.set_xlim,([0,max(5.*np.arange(len(indx[0])))])
        ax2.set_ylabel('Height, $[km]$',fontsize=fontsize)
        ax2.set_title('DFR',fontsize=fontsize)
        ax2.set_xlim([xmin,xmax])
        ax2.set_ylim([ymin,ymax])
        ax2.tick_params(axis='both',direction='in',labelsize=fontsize2,width=2,length=5)

        cbar = plt.colorbar(pm,aspect=10,ax=ax2)
        cbar.set_label('DFR_Ku-Ka, $[dB]$',fontsize=fontsize)
        cax = cbar.ax
        cax.tick_params(labelsize=fontsize2)
        ax2.set_xlabel('Distance, $[km]$',fontsize=fontsize)
        
    elif band=='DFR_raw':
        
        if Kuray - Karay != 12:
            print('Default matching to designated Ku ray')
            Karay = Kuray - 12
            indxMS=np.where((MS['Latitude'][:,Karay] > minlat) & (MS['Latitude'][:,Karay] < maxlat) \
          & (MS['Longitude'][:,Karay] > minlon) & (MS['Longitude'][:,Karay] < maxlon))
            
        fig,axes = plt.subplots(1,1,figsize=figsize)
        ax2 = axes
        ##Ku 
        Z =np.transpose(np.squeeze(NS['zFactorMeasured'][indx,Kuray,:]))
        ind = np.where(Z <= vmin )
        Z[ind] = np.inf
        Z2 = np.ma.masked_invalid(Z, copy=True)
        ##Ka
        height = MS['Height']
        Z =np.transpose(np.squeeze(MS['zFactorMeasured'][indxMS,Karay,:]))
        ind = np.where(Z <= vmin )
        Z[ind] = np.inf
        Z = np.ma.masked_invalid(Z, copy=True)
        
        DFR = Z2 - Z
        
        pm= ax2.pcolormesh(5.*np.arange(len(indxMS[0])),height[:,Karay],DFR,cmap=cmap,
                           vmin=vmin,vmax=vmax)
        ax2.fill_between(5.*np.arange(len(indxMS[0])),.125*(176-np.squeeze(MS['binClutterFreeBottom'][indxMS,Karay])),
                         color= [.5,.5,.5],alpha=0.5)
        ax2.fill_between(5.*np.arange(len(indxMS[0])),.125*(176-np.squeeze(MS['binRealSurface'][indxMS,Karay])),color='k')

        ax2.set_ylim([0,10])
        ax2.set_xlim,([0,max(5.*np.arange(len(indx[0])))])
        ax2.set_ylabel('Height, $[km]$',fontsize=fontsize)
        ax2.set_title('DFR',fontsize=fontsize)
        ax2.set_xlim([xmin,xmax])
        ax2.set_ylim([ymin,ymax])
        ax2.tick_params(axis='both',direction='in',labelsize=fontsize2,width=2,length=5)

        cbar = plt.colorbar(pm,aspect=10,ax=ax2)
        cbar.set_label('DFR_Ku-Ka, $[dB]$',fontsize=fontsize)
        cax = cbar.ax
        cax.tick_params(labelsize=fontsize2)
        ax2.set_xlabel('Distance, $[km]$',fontsize=fontsize)

    elif band=='KuKa':
        fig,axes = plt.subplots(2,1,figsize=figsize)
        ax1 = axes[0]
        ax2 = axes[1]
        
        ##Ku
        height = NS['Height']
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
        cbar.set_label('Reflectivity, $[dBZ]$',fontsize=fontsize)
        cax = cbar.ax
        cax.tick_params(labelsize=fontsize2)
      
        ##Ka
        height = MS['Height']
        Z =np.transpose(np.squeeze(MS['zFactorCorrected'][indxMS,Karay,:]))
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
        cbar.set_label('Reflectivity, $[dBZ]$',fontsize=fontsize)
        cax = cbar.ax
        cax.tick_params(labelsize=fontsize2)
        ax2.set_xlabel('Distance, $[km]$',fontsize=fontsize)
        plt.tight_layout()
    if savefig:
        print('Save file is: '+'Profile_'+band+camp+'.png')
        plt.savefig('Profile_'+band+camp+'.png',dpi=300)    
    plt.show()

    return  

def GPMDPR_xtrack(filename,camp =' ',band='Ku',savefig=False,Crossray=0, Kuray = 23, Karay=11,
                   fontsize=14, fontsize2=12, vmin=-20,vmax=75,lw = 3, xmin=0, xmax=240, 
                   ymin=0,ymax=12,cmap='NWSRef',zoom=1,figsize=(15,4),lat_0 = '0',lon_0 = '90'):
    
    """
    ========
    
    NOTE: May have side lobe issues!!
    
    ========
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
    
    if cmap == 'NWSRef':
        cmap = cm.NWSRef
    

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
    
    if Crossray == 0:
        Crossray = indx[0][int(indx[0].shape[0]/2)]
        
    indind = np.where(Crossray == indx)
    indind = np.where(indx[0] == Crossray)
    
    if indind[0].shape[0] == 0:
        temp = indx[0]
        print('Please choose a Crossray between '+ str(temp.min())+' - '+ str(temp.max()))
        return
    
    fig,axes = plt.subplots(1,1,figsize=figsize)
    ax1 = axes
    Z =np.transpose(np.squeeze(NS['zFactorCorrected'][Crossray,:,:]))
    ind = np.where(Z <= vmin )
    Z[ind] = np.inf
    Z = np.ma.masked_invalid(Z, copy=True)
    temp = np.ones(Z.shape)
    xray = 5.*np.arange(0,49)
    for i in np.arange(0,Z.shape[0]):
        temp[i,:] = xray
    xx = temp
    height = NS['Height']
    
    pm = ax1.pcolormesh(xx,height,Z,cmap=cmap,
                    vmin=vmin,vmax=vmax)
    ax1.fill_between(5.*np.arange(0,49),.125*(176-np.squeeze(NS['binClutterFreeBottom'][Crossray,:])),
                     color=[.5,.5,.5],alpha=.5)
    ax1.fill_between(5.*np.arange(0,49),.125*(176-np.squeeze(NS['binRealSurface'][Crossray,:])),color='k')
    ax1.set_ylim([0,10])
    ax1.set_xlim,([0,max(5.*np.arange(len(indx[0])))])
    ax1.set_ylabel('Height, $[km]$',fontsize=fontsize)
    ax1.set_title('KuNS',fontsize=fontsize)
    ax1.set_xlim([xmin,xmax])
    ax1.set_ylim([ymin,ymax])
    ax1.tick_params(axis='both',direction='in',labelsize=fontsize2,width=2,length=5)

    cbar = plt.colorbar(pm,aspect=10,ax=ax1)
    cbar.set_label('Reflectivity, $[dBZ]$',fontsize=fontsize)
    cax = cbar.ax
    cax.tick_params(labelsize=fontsize2)
    ax1.set_xlabel('Distance, $[km]$',fontsize=fontsize)
    
    if savefig:
        print('Save file is: '+'xtrack_Profile_'+band+camp+'.png')
        plt.savefig('xtrack_Profile_'+band+camp+'.png',dpi=300)    
    plt.show()
    
    return
   
