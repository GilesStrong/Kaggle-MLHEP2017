import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
import time

doTrain = True
if doTrain:
    ds_tracks = pd.read_csv('../data/DS_1_train.csv')
    ds_electrons = pd.read_csv('../data/DS_1_electron_train.csv')
else:
    ds_tracks = ds_test = pd.read_csv('../data/DS_1_test.csv')
    ds_electrons = pd.read_csv('../data/DS_1_electron_test.csv')

print len(ds_tracks)
#ds_tracks = ds_tracks[:10]
#ds_electrons = ds_electrons[:5]

##################################################
# This bad boy block adds the dR information to the dataset
##################################################

class electron:
    def __init__(self, x1, y1, z1, tx, ty):
        self.x1 = x1
        self.y1 = y1
        self.z1 = z1
        self.tx = tx
        self.ty = ty
        
    def get_x_pos(self, z):
        x0 = self.x1-self.tx*self.z1
        suggestions = x0 + z*np.sin(self.tx)
        # replace position before the track starts with a large number
        np.put(suggestions, np.where(z < self.z1), 999000)
        return suggestions
    
    def get_y_pos(self, z):
        y0 = self.y1-self.ty*self.z1
        suggestions = y0 + z*np.sin(self.ty)
        # replace position before the track starts with a large number
        np.put(suggestions, np.where(z < self.z1), 999000)
        return suggestions
    
    def get_distance_from(self,x,y,z):
        dx = x - self.get_x_pos(z)
        dy = y - self.get_y_pos(z)
        return (dx**2 + dy**2)**0.5
    
    def get_dTX(self, tx):
        return tx - self.tx
    
    def get_dTY(self, ty):
        return ty - self.ty

t1 = time.time()

###########################
# create the electrons from the dataset
###########################
electrons = []
for i in range(len(ds_electrons)):
    e = electron(ds_electrons.iloc[i]['X'],
                 ds_electrons.iloc[i]['Y'],
                 ds_electrons.iloc[i]['Z'],
                 ds_electrons.iloc[i]['TX'],
                 ds_electrons.iloc[i]['TY'])
    electrons.append(e)

###########################
# Now take all the tracks in the set and compute the distance to the electron path
###########################
tracks_xs = ds_tracks['X']
tracks_ys = ds_tracks['Y']
tracks_zs = ds_tracks['Z']

distances = np.zeros(shape=(len(tracks_xs), len(electrons)))
for i, electron in enumerate(electrons):
    distances[:,i] = electron.get_distance_from(tracks_xs, tracks_ys, tracks_zs)

n_angles = 3
el_indices = distances.argsort()[:,:n_angles]

dTX = np.zeros(shape=(len(ds_tracks), n_angles))
dTY = np.zeros(shape=(len(ds_tracks), n_angles))

for i in range(len(ds_tracks)):
    el_ind = el_indices[i]
    
    for i_el in range(n_angles):
        dTX[i, i_el] = electrons[el_ind[i_el]].get_dTX(ds_tracks.iloc[i]['TX'])
        dTY[i, i_el] = electrons[el_ind[i_el]].get_dTY(ds_tracks.iloc[i]['TY'])

# sort the distances for each track
distances.sort(axis=1)

###########################
# and finally add to data:
# - minimum distances to electrons
# - differences in angles to nearest electrons
###########################

final = ds_tracks.copy()

# distances
for i in range(5):
    final['dR'+str(i)] = distances[:,i]

# angles
for i in range(n_angles):
    final['dTX'+str(i)] = dTX[:,i]
    final['dTY'+str(i)] = dTY[:,i]

t2 = time.time()
print 'n tracks:', len(ds_tracks)
print 'seconds:', t2-t1

final

if doTrain:
    final.to_csv('../data/DS_5_train.csv', header=True)
else:
    final.to_csv('../data/DS_5_test.csv', header=True)

os.system('say "features added"')

