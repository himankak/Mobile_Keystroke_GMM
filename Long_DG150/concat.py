# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 12:06:40 2019

@author: Biomedia4n6
"""

import pandas 

#os.chdir("D:\\Keystroke\\Long_DG150")
fout=open("out.csv","a")
path = "D:\\Keystroke\\mobile_pace-sensor_features.csv"
data = pandas.read_csv(path)
# first file:
for line in open("D:\\Keystroke\\Long_DG150\\USER(1).csv"):
    fout.write(line)
# now the rest:    
for num in range(2,201):
    f = open("USER"+str(num)+".csv")
    f.next() # skip the header
    for line in f:
         fout.write(line)
    f.close() # not really needed
fout.close()