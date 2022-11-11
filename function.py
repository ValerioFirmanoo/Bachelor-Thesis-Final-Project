import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import os.path
from scipy import stats
import networkx as nx
#from math import pow
import math
import cmath
from numpy.polynomial.polynomial import polyfit


#gestione dei dati/strain/repliche...----------------------------------------------------------------------------------------------------------------------------------------------

def open_flat (strain, date):
    adir=os.getcwd()
    path_f=adir+"/"+"data"+"/"+strain+"/"+str(date)+"/"+str(date)+"_flat.csv"
    data=pd.read_csv(path_f, sep=',')
    return data


def carica_indici(strain, name):
    tabella_indici=pd.read_csv('~/tesi/data/indice.csv')
    strain=np.array(tabella_indici[tabella_indici.strain==name].date, dtype = int)
    if strain.size==0:
        print('error in loading index '+ name)
        return
    return strain


def read_replica_flat (name, date):
    adir=os.getcwd()
    path_f=adir+"/"+"data"+"/"+name+"/"+str(date)+"/"+str(date)+"_flat.csv"
    data=pd.read_csv(path_f, sep=',')
    if data.size==0:
        print('error in adding replicas of '+ name)
        return
    return data


#da finire read replica tot
def read_replica_tot (name, date):
    adir=os.getcwd()
    path_f=adir+"/"+"data"+"/"+name+"/"+str(date)+"/"+"line_data/"+str(date)+"_total.csv"
    data=pd.read_csv(path_f, sep=',')
    if data.size==0:
        print('error in adding replicas of '+ name)
        return
    return  data#[~data.line.str.contains('line')]


def unify_replicas_flat( name , index ):
    df=read_replica_flat( name , index[0] )
    for i in index[1:]:
        df=df.append(read_replica_flat( name , i),ignore_index=True)
    return df


def unify_replicas_tot( name , index ):
    df=read_replica_tot( name , index[0] )
    for i in index[1:]:
        df=df.append(read_replica_tot( name , i),ignore_index=True)
    return df


#statistics funcions-------------------------------------------------------------------------------------------------------------------------------------------------------------------


def dev_std_filter( x ,mean,  dev_std , n_dev_std):
    mask_filter= (x < mean + n_dev_std*dev_std) & (x > mean - n_dev_std*dev_std)
    return x.loc[mask_filter]


def bin_mean( t , values , bin_widht , window ,  t_custom_min , t_custom_max , interval='normal'):
    if interval=='normal':
        t_min=t.min()
        t_max=t.max()
        n_bin=int((t_max-t_min)/bin_widht)
        result=[]
        for i in range(n_bin):
            T= (t > t_min+i*bin_widht-0.5*window) & ( t < t_min+i*bin_widht+0.5*window)
            values_in_bin=values.loc[T]
            good_points=dev_std_filter(values_in_bin ,values_in_bin.mean(),values_in_bin.std(), 3)
            result.append(good_points.mean())
    if interval=='custom':
        t_min=t_custom_min
        t_max=t_custom_max
        n_bin=int((t_max-t_min)/bin_widht)
        result=[]
        for i in range(n_bin):
            T= (t > t_min+i*bin_widht-0.5*window) & ( t < t_min+i*bin_widht+0.5*window)
            values_in_bin=values.loc[T]
            good_points=dev_std_filter(values_in_bin ,values_in_bin.mean(),values_in_bin.std(), 3)
            result.append(good_points.mean())
    return result

def bin_sum( t , values , bin_widht , window ,  t_custom_min , t_custom_max , interval='normal'):
    if interval=='normal':
        t_min=t.min()
        t_max=t.max()
        n_bin=int((t_max-t_min)/bin_widht)
        result=[]
        for i in range(n_bin):
            T= (t > t_min+i*bin_widht-0.5*window) & ( t < t_min+i*bin_widht+0.5*window)
            values_in_bin=values.loc[T]
            good_points=dev_std_filter(values_in_bin ,values_in_bin.mean(),values_in_bin.std(), 3)
            result.append(good_points.sum())
    if interval=='custom':
        t_min=t_custom_min
        t_max=t_custom_max
        n_bin=int((t_max-t_min)/bin_widht)
        result=[]
        for i in range(n_bin):
            T= (t > t_min+i*bin_widht-0.5*window) & ( t < t_min+i*bin_widht+0.5*window)
            values_in_bin=values.loc[T]
            good_points=dev_std_filter(values_in_bin ,values_in_bin.mean(),values_in_bin.std(), 3)
            result.append(good_points.sum())
    return result

#funzioni gestione serie temporali (funzioni di Giorgio)-----------------------------------------------------------------------------------------------------------------------------------------------


def derivata(x,y):
    if len(x)!=len(y):
        raise LenError('input arrays have different length')
    dim=len(x)
    dy=[0]*dim

    dy[dim-1]=(y[dim-1]-y[dim-2])/(x[dim-1]-x[dim-2])
    dy[0]=(y[1]-y[0])/(x[1]-x[0])
    for i in list(range(dim-2)):
        dy[i+1]=(y[i+2]-y[i])/(x[i+2]-x[i])

    return dy

def derivata3(x,Y):
    if len(x)!=len(Y):
        raise LenError('input arrays have different length')

    dim=len(x)

	#y[0]=0.5*(y[0]+y[1])
	#y[dim-1]=0.5*(y[dim-1]+y[dim-2])
    y=np.copy(Y)

    for i in list(range(dim-2)):
        y[i+1]=(y[i+2]+y[i+1]+y[i])/3

    dim=len(x)
    dy=[0]*dim

    dy[dim-1]=(y[dim-1]-y[dim-2])/(x[dim-1]-x[dim-2])
    dy[0]=(y[1]-y[0])/(x[1]-x[0])

    for i in list(range(dim-2)):
        dy[i+1]=(y[i+2]-y[i])/(x[i+2]-x[i])

    return dy
   #return [dy,Y] 

def derivata_sg(x,y,window=5,poly_order=4): #usa il filtro di savitzky-golay per il calcolo della derivata

    if len(x)!=len(y):
        raise LenError('input arrays have different length')
    dim=len(y)
    if dim<window:
        appo=derivata3(x,y)
        der=np.array(appo) #[0])
        #Y=np.array(appo[1])
    if dim>=window:
        import scipy.signal as sg 
        Y=sg.savgol_filter(y,window,poly_order)
        der=np.array(derivata(x,Y))

    return der
    #return [dy,Y] #il primo vettore è quello che contiene la derivata, il secondo contiene le y filtrate

def derivata_sg_window(x,y,window,poly_order=4): #usa il filtro di savitzky-golay per il calcolo della derivata

    if len(x)!=len(y):
        raise LenError('input arrays have different length')
    dim=len(y)
    if dim<window:
        appo=derivata3(x,y)
        der=np.array(appo) #[0])
        #Y=np.array(appo[1])
    if dim>=window:
        import scipy.signal as sg 
        Y=sg.savgol_filter(y,window,poly_order)
        der=np.array(derivata(x,Y))

    return der
    #return [dy,Y] #il primo vettore è quello che contiene la derivata, il secondo contiene le y filtrate


#funzione per fit fluorescenza-------------------------------------------------------------------------------------------------------------------------------------------------------------------


def fit_func_fluo(x, a1, a2):
    return (0.55/a1)-(a2/a1)*x

def fit_func_fluo2(x, b1):
    return (0.55/b1)-x

#carica le date/indice delle repliche (evito di intasare il notebook)

#integrazione di array di derivate discrete----------------------------------------------------------------------------------------------------------------------------------------------------------
def discrete_int(der_array,delta_t,A0):
	A=np.zeros(len(der_array))
	A[0]=A0
	appo=A[0]
	for i in (range(1,len(A))):
		A[i]=appo+der_array[i]*delta_t
		appo=A[i]
	return A
	
	

p1longori3_index=[]
p1longter3_index=[]
p1ori3_index=[]
p1ter3_index=[]
p5ori3_index=[]
p5ter3_index=[]
p1longori3_index_sbagliato=[]

p1longori3_index=carica_indici(p1longori3_index,'P1longori3')
p1longter3_index=carica_indici(p1longter3_index,'P1longter3')
p1ori3_index=carica_indici(p1ori3_index,'P1ori3')
p1ter3_index=carica_indici(p1ter3_index,'P1ter3')
p5ori3_index=carica_indici(p1longori3_index,'P5ori3')
p5ter3_index=carica_indici(p1longori3_index,'P5ter3')
p1longori3_index_sbagliato=np.append(p1longori3_index,20160509)
growth_path='/home/valerio/tesi/data/growth_rate/'
#df_prova=unify_replicas_tot('P5ori3',p5ori3_index)
#print(df_prova)

#print(df_prova)
#print (df_prova.t1Trans)


