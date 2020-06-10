
import h5py, os, glob
import numpy as np
import sys
from shutil import copy, move
import csv 

sys.path.insert(0, '/localdata/bkhamesra3/LIGO_Waveforms/metadata_correction_scripts')
from metadata import metadata


def read_attr(filepath, outfile):
    fileh5 = h5py.File(filepath)
    key = fileh5.attrs.keys()
#    print key, fileh5.attrs.keys()
    values = [fileh5.attrs[k] for k in key]
    atrb = dict(zip( key, values))

    spin1 = np.array((atrb['spin1x'], atrb['spin1y'], atrb['spin1z']))
    spin2 = np.array((atrb['spin2x'], atrb['spin2y'], atrb['spin2z']))

    print "spin1 = ",spin1
    print "\n spin2 = ", spin2

    if np.count_nonzero(spin1)==0 and np.count_nonzero(spin2)==0:
	atrb['spin-type']='NonSpinning'
    elif np.count_nonzero(spin1[0:2])>0 or np.count_nonzero(spin2[0:2]>0):
	atrb['spin-type']='Precessing'
    else:
	atrb['spin-type'] = 'AlignedSpins'
    print("\n spin type = {} \n \n".format(atrb['spin-type'] 	))
    output_file = open(outfile, 'w')
    output = csv.writer(output_file)
    for key, val in atrb.items():
	output.writerow([key,val])
    output_file.close()
    #return atrb 
    fileh5.close()
   



filepath = sorted(glob.glob("/localdata/bkhamesra3/LIGO_Waveforms/GW_Waveforms/GT/*.h5"))
#h5 = "/localdata/bkhamesra3/LIGO_Waveforms/GW_Waveforms/GT/GT0370.h5"
output_dir = "/localdata/bkhamesra3/LIGO_Waveforms/GW_Waveforms/Metadata"

for h5 in filepath:
	print("{} Starting \n".format(os.path.basename(h5)))	
	gt_tag = os.path.basename(h5).split('.')[0]
	outfile_name = os.path.join(output_dir,"Metadata_"+gt_tag+".csv")

	attrib = read_attr(h5, outfile_name)
	
