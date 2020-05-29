#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 14:09:52 2020
Autheurs : Yoann Le Saint - Mael Jaouen - Maxime Le - Oualid Ben Mohamed
"""
import math as m
import numpy as np
from matplotlib import pyplot as plt
import random as rd
import copy

#param : point 
#return : valeur f(x) Rosenbrock
def f(x):
    y=np.array(x)
    ssi1=y[1:]
    ssi=y[0:-1]
    X=np.sum(100*(ssi1-ssi**2)**2+(1-ssi)**2)
    return X
    

#param : point 
#return : valeur g(x) sphèrique
def g(x):
    return np.sum(np.array(x)**2)

#param L : np.Array de points 
#return : le barycentre np.Array
def barycentre(L):
    n=len(L)
    s=L[0]
    for i in range(1,n):
        s+=L[i]
    
    return s/n

#param (a,b): deux points/vecteurs np.Array (ICI nvx barycentre et ancien)
#return : ||a-b|| (euclidienne) (entier)

def Norme(bary1,bary2):
    return m.sqrt(np.sum((np.array(bary1)-np.array(bary2))**2))


# Initialisation du début
#param entier n,dim : nombre de points, dimension des points (de départ), (a,b) intevalle de valeurs pour les vecteurs
#return les Points (liste, liste d'aa), Vpoints (barycentre) np.Array
def initialisation(n,dim,a,b):
    #Points = [np.array(),,]
    Points=[]
    Vpoints=[]
    for i in range(n):
        Points.append(np.random.uniform(low=a, high=b, size=(dim)))
        Vpoints.append(np.zeros(dim))
    return Points, Vpoints


def simulation(Tmax,n,eps, phi1, phi2,dt, dim,f,w,pmin,pmax):
    # Initialisation des points sur l'ensemble de travail
    Points1, Vpoints = initialisation(n,dim,pmin,pmax)
    Point = [Points1]
    t=0
    bary1=barycentre(Vpoints) #premier barycentre 
    bary2=bary1+eps+1 #second barycentre placé assez loin du premier pour rentrer dans la boucle
    pb=Points1
    gb=pb[0]
    while t<Tmax and Norme(bary1,bary2)>eps:
        # Boucle temporelle
        u1,u2 = rd.random(), rd.random()
        Vpoints2=copy.deepcopy(Vpoints)
        Vpoints=[]
        Points2=copy.deepcopy(Points1)
        Points1=[]
        for j in range(n):
            # Contruction de pb et gb
            if f(Point[t][j])<f(pb[j]):
                pb[j] = Point[t][j]
            if f(pb[j])<f(gb):
                gb=pb[j]
        for i in range(n):
            #Boucle sur les n points
            Vpoints.append(w*Vpoints2[i] + phi1*u1*(pb[i]-Points2[i]) + phi2*u2*(gb-Points2[i]))
            Points1.append(Points2[i] + Vpoints[i]*dt)
            # max(Xinf,min(Xsup, x)) -> Xinf<x<Xsup
            Points1[i] = limite(Points1[i],pmin,pmax)
        Point.append(Points1)
        bary2=bary1
        bary1=barycentre(Point[t]) #nouveau barycentre
        t+=1
    return Point, gb,t

def limite(Point,pmin,pmax):
    # Tous les points sont compris entre pmin et pmax
    for k in range(len(Point)):
        Point[k] = max(pmin,min(Point[k],pmax))
    return Point

def affichage(Tmax,n,eps, phi1, phi2,dt, f,w,pmin,pmax):
    p, gb , t = simulation(Tmax,n,eps, phi1, phi2,dt, 2,f,w,pmin,pmax)
    print(gb)
    for point in p:
        for i in point:
            plt.scatter(i[0], i[1]) #placer points
        plt.axis([pmin,pmax,pmin,pmax])
        plt.pause(0.1) # pause avec duree en secondes
        plt.clf()
    plt.show()


def phi(Tmax,n,eps,phi2,dt,f,w,pmin,pmax):
    # Variation de phi2 entre 0 et 1
    phi1 = np.linspace(0,100,100)
    l=[]
    for p in phi1:
        print(p)
        #l.append(np.linalg.norm(EtudeStat(Tmax,n,eps,p,phi2,dt,f,30,w,pmin,pmax),2))
        l.append(simulation(Tmax,n,eps,p,phi2,dt,2,f,w,pmin,pmax)[2])
    plt.plot(phi1,l)
    plt.show()


def EtudeN(Tmax,eps,phi1,phi2,dt,f,w,pmin,pmax):
    # Variation de phi2 entre 0 et 1
    n = [i for i in range(1,10000)]
    for p in n:
        s = simulation(Tmax,p,eps,phi1,phi2,dt,2,f,w,pmin,pmax)[1]
        #print("n =", p, " : ", s)
        print(s)
        plt.axis([pmin,pmax,pmin,pmax])
        plt.scatter(s[0],s[1])
        plt.pause(0.01)
    plt.show()


def EtudeStat(Tmax,n,eps, phi1, phi2,dt,f, N,w,pmin,pmax):
    x=simulation(Tmax,n,eps,phi1,phi2,dt,2,f,w,pmin,pmax)[1]
    for i in range(N):
        x+= simulation(Tmax,n,eps,phi1,phi2,dt,2,f,w,pmin,pmax)[1]
    return x/(N+1)

#print(simulation(500,1000,0.001,1,1,0.1,4,f,0.9,-10,10)[1])
#affichage(500,100,0.001,1,1,0.1,f,0.9,-10,10)
#phi(250,500,0.01,0.5,0.1,f,0.9,-10,10)
#EtudeN(100,0.01,1,1,0.1,f,0.5,-10,10)
#print(EtudeStat(100,50,0.01,1,1,1,f,100,0.9,-10,10))