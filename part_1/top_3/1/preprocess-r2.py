import numpy as np
import pandas as pd

def angle(A,B):
    return np.dot(A,B)/ (np.linalg.norm(A) * np.linalg.norm(B))

def finde(ev, dfe, g):
    d = np.array(ev[["X", "Y", "Z"]].values - dfe[["X", "Y", "Z"]].values, dtype=float)
    dee = d/np.linalg.norm(d, axis=1).reshape(-1,1)
    angs = np.einsum('ij,ij->i', dee, g)
    ee = dfe.iloc[angs.argmax()]
    ev['ang'] = angs.max()
    ev['angP'] = np.arccos(angle(np.array((ev.TX, ev.TY, 1)), np.array((ee.TX, ee.TY, 1))))
    return ev

df_test = pd.read_hdf('data.h5', "test_r1")
dfe_test = pd.read_csv('DS_1_electron_test.csv', index_col=0).reset_index()
dfe_test["TZ"] = 1
gt = dfe_test[["TX", "TY", "TZ"]].values
gt/=np.linalg.norm(gt, axis=1).reshape(-1,1)

from joblib import Parallel, delayed
import multiprocessing

def applyParallel(dfGrouped, func):
    retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name, group in dfGrouped)
    return pd.concat(retLst)
def finde_test(ev):
    return finde(ev, dfe_test, gt)

xdf_test = applyParallel(df_test.groupby(df_test.index), finde_test)
# xdf_test = df.apply(finde, axis=1, args=(dfe_test,gt))
xdf_test.to_hdf('testdata-r2.h5', "test_r2")
