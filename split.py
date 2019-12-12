# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 12:17:03 2019

@author: Biomedia4n6
"""

import csv

with open('mobile_pace-sensor_features.csv') as fin:    
    csvin = csv.DictReader(fin)
    # Category -> open file lookup
    outputs = {}
    for row in csvin:
        cat = row['subject']
        # Open a new file and write the header
        if cat not in outputs:
            fout = open('USER WISE FEATURES/{}.csv'.format(cat), 'w', newline='')
            dw = csv.DictWriter(fout, fieldnames=csvin.fieldnames)
            dw.writeheader()
            outputs[cat] = fout, dw
        # Always write the row
        if any(row):
            outputs[cat][1].writerow(row)
    # Close all the files
    for fout, _ in outputs.values():
        fout.close()