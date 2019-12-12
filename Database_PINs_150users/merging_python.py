# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:15:33 2019

@author: Biomedia4n6
"""

import csv

with open('user_data.txt', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split(",") for line in stripped if line)
    with open('user_data.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(('title', 'intro'))
        writer.writerows(lines)
        
#        while i < 8:
#        print(i)
#        i += 1
#        lines = (line.split(",") for line in stripped if line)