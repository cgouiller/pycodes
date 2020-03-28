# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

from mat4py import loadmat
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
import numpy as np
from scipy import interpolate
from pathlib import Path


MyCoreDir="e:/Clément/MyCore/"

os.chdir(Path(MyCoreDir + "Analyse/Mixing"))
manips=loadmat('manips.mat')

manips=pd.DataFrame.from_dict(manips,orient='columns')


manips.insert(12,'Chemin',MyCoreDir+ 'Analyse/' + manips.Projet + '/' + manips.Date + '/' + manips.Sets + '/' + manips.NomDoss)

# %% Fig 1b
#%matplotlib inline  

os.chdir(Path(manips.loc[2].Chemin)) #On se place dans le dossier où récupérer l'image
im = np.array(Image.open('0133555.tif')) #On la charge
#On restreint au carré dans le cercle est inscrit:
im=im[manips.loc[2].CenterX:1+manips.loc[2].CenterX+2*manips.loc[2].RadiusPx,manips.loc[2].CenterY:1+manips.loc[2].CenterY+2*manips.loc[2].RadiusPx]
#On crée les axes x et y en cm, et où zéro est les centre de la cuve
x,y=np.arange(len(im))*9/manips.loc[2].RadiusPx,np.arange(len(im))*9/manips.loc[2].RadiusPx
x,y=x-np.mean(x),y-np.mean(y)
# On crée d, distance au centre de n'importe quel point de l'image
xmg,ymg=np.meshgrid(x,y)
d=np.sqrt(xmg**2+ymg**2)
#On met à 0 tout ce qui est à plus de 9cm du centre, c'est-à-dire qui n'est pas dans la cuve
im[d>9]=0
#On trace !

fig, ax = plt.subplots(1,1)
fig.suptitle('Image classique obtenue', fontsize=14, fontweight='bold')
ax.set_xlabel('x [cm]')
ax.set_ylabel('y [cm]')
ax.set_xticks(range(-8,9,2))
ax.set_yticks(range(-8,9,2))


plt.imshow(im,cmap="gray",extent=[x[0],x[-1],x[0],x[-1]])


# %% Fig 2b
im0=loadmat('im0.mat')
im0=np.array(im0['im0'])
x,y=np.arange(len(im0))*9/manips.loc[2].RadiusPx,np.arange(len(im0))*9/manips.loc[2].RadiusPx
x,y=x-np.mean(x),y-np.mean(y)
if np.where(im0==0)[0].size>0:
    xx,yy=range(len(im0)),range(len(im0))
    f = interpolate.interp2d(xx, yy, im0, kind='cubic')
    while np.where(im0==0)[0].size>0:
        im0[np.where(im0==0)[0][0],np.where(im0==0)[1][0]]=f(np.where(im0==0)[0][0],np.where(im0==0)[1][0])

Cfield=(im0-im)/im0
Cfield=Cfield/0.0140 # Pour passer en g/cm2
Cfield[d>9]=0
fig, ax = plt.subplots(1,1)
fig.suptitle('Champ de concentration classique obtenu', fontsize=14, fontweight='bold')
ax.set_xlabel('x [cm]')
ax.set_ylabel('y [cm]')
plt.imshow(Cfield,cmap="viridis",vmin=0,vmax=10,extent=[x[0],x[-1],x[0],x[-1]])

ax.set_xticks(range(-8,9,2))
ax.set_yticks(range(-8,9,2))
plt.colorbar()


# %% Fig 4a
shortlist=manips.loc[manips.loc[:,'Param']=='N']
for i in shortlist.itertuples():
    os.chdir(i.Chemin)
    tmp=loadmat('Conc.mat')
    time=np.array(tmp['time'])
    Cstd=np.array(tmp['Cstd'])
    plt.plot(i.Nombre,np.mean(Cstd[(time>55*60) & (time<65*60)]),'+k')
plt.axis([0,50,0,0.06])
plt.xlabel('N')
plt.ylabel('Cstd')
plt.show()


# %% Fig 4b
shortlist=manips.loc[manips.loc[:,'Param']=='N']
for i in shortlist.itertuples():
    os.chdir(i.Chemin)
    tmp=loadmat('spectre.mat')
    k=np.array(tmp['k'])
    Sp=np.array(tmp['Sp'])
    plt.plot(k,Sp,label='N='+str(i.Nombre))
#plt.axis([0,50,0,0.06])
plt.xlabel('k [mm$^{-1}$]')
plt.ylabel('Sp')
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid()
plt.show()

# %% Fig 4c
shortlist=manips.loc[manips.loc[:,'Param']=='R']
for i in shortlist.itertuples():
    os.chdir(i.Chemin)
    tmp=loadmat('Conc.mat')
    time=np.array(tmp['time'])
    Cstd=np.array(tmp['Cstd'])
    plt.plot(i.Rayon,np.mean(Cstd[(time>55*60) & (time<65*60)]),'+k')
plt.axis([0,5,0,0.06])
plt.xlabel('Rayon [mm]')
plt.ylabel('Cstd')
plt.show()


# %% Fig 4d
shortlist=manips.loc[manips.loc[:,'Param']=='R']
for i in shortlist.itertuples():
    os.chdir(i.Chemin)
    tmp=loadmat('spectre.mat')
    k=np.array(tmp['k'])
    Sp=np.array(tmp['Sp'])
    plt.plot(k,Sp,label='R='+str(i.Rayon)+'mm')
#plt.axis([0,50,0,0.06])
plt.xlabel('k [mm$^{-1}$]')
plt.ylabel('Sp')
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid()
plt.show()
