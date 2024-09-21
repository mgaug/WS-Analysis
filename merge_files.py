#import h5py
#import os
import pandas as pd

file_names = [ "WS2003-23_long.h5", "WS2023-24_long.h5" ]
outfile = "WS2003-24_long.h5"

dff0 = pd.read_hdf(file_names[0])
dff1 = pd.read_hdf(file_names[1])

dff = pd.concat([dff0, dff1])

dff.to_hdf(outfile,key="dff",mode="w") #writes data to hdf file



#d_struct = {} #Here we will store the database structure
##for i in d_names:
#  print("Opening: ",i)
#   f = h5py.File(i,'r+')
#   print("Keys: ",f.keys())   
#   d_struct[i] = f.keys()
#   print ("Structure: ",d_struct[i])
#   f.close()

#for i in d_names:
#   print("Treating: ",i)
#   os.system('h5copy -i %s -o %s -s dff -d dff' % (i, outfile))   
#   for j  in d_struct[i]:
#      print("Copying: ",j)      
#      os.system('h5copy -i %s -o %s -s %s -d %s' % (i, outfile, j, j))
