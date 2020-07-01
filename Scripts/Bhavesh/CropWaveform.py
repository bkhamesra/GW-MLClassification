import numpy as np
import glob, os
import matplotlib.pyplot as plt

#Set minimum length of waveform
min_duration = 400

def write_data(outfile, strain_data):
    '''Write strain data to output file
    
    Parameters - 
    outfile (str) - output file path
    strain_data - Strain data - (t, real(h), imag(h))
    '''    
    output_file = open(outfile,'w')
    hdr = '# Time \t h_plus \t h_cross \n'
    np.savetxt(output_file, strain_data, header=hdr, delimiter='\t', newline='\n')
    output_file.close()

def crop_wf(filename, len_init, len_final):
    '''Crop waveform to have same number of points before and after merger - len_init points before and len_final points after
    
    Parameters- 
    filename - filepath containing the strain data
    len_init - initial length before the merger
    len_final - final length after the merger
    '''  
    
    #Load strain data and compute amplitude and phase
    time, hp, hx = np.loadtxt(filename, usecols=(0,1,2,), unpack=True)
    amp = abs(hp+1.j*hx)
    phase = -np.unwrap(np.angle(hp + 1.j*hx))
    
    #Find the peak of amplitude
    idx =np.where(amp== np.amax(amp))
    t_maxamp = time[idx]
       
    #Only consider waveforms with a minimum length
    if t_maxamp-time[0]>=min_duration:
        
        #Time-shift to set 0 at the peak
        time = time-t_maxamp
        data = np.array((time, hp,hx))
        
        #Crop and save the new waveform
        idx_init = np.where(time==0.0)[0]-len_init
        idx_fnl = np.where(time==0.0)[0]+len_final
        
        data = data.T[idx_init:idx_fnl+1]
        outfile = os.path.join("../FilteredData", os.path.basename(filename))
        write_data(outfile, data)
        print("{} \t {} \t {} \t {} \t{} \n".format(filename, len(time),np.shape(data)[0], len_init+len_final, idx_init-idx_fnl))
    
    

def find_shortestwf(dirname):
    '''Find the shortest waveform and determine number of datapoints in inspiral and post merger signal
       dirname (str) - path of directory with all waveform data
    '''
    waveform_files = sorted(glob.glob(os.path.join(dirname,'*.txt')))
    vin = []
    vf = []
    filenames = []
    
    for wf_file in waveform_files:
        t, hp, hx = np.genfromtxt(wf_file, unpack=True, usecols=(0,1,2))
        amp = abs(hp+1.j*hx)
        idx =np.where(amp== np.amax(amp))
        if t[idx]-t[0]>=min_duration:
            vin.append(idx)
            vf.append(len(amp)-idx[0]-1)
            filenames.append(wf_file)
    
    len_init = sorted(vin)[0]
    len_final = sorted(vf)[0]
    return len_init, len_final

#Compute lengths of shortest waveform for each group waveform bank
dirpath =[ "../../Data/WaveformData/GT", "../../Data/WaveformData/SXS"]
wfpath = ["../../Data/WaveformData", "../../Data/WaveformData"]
len_init1, len_final1 = find_shortestwf(dirpath[0])
len_init2, len_final2 = find_shortestwf(dirpath[1])
#print len_init1,len_final1, len_init2,len_final2

#Find minimum length of the waveform
len_init = min(len_init2, len_init1)
len_final = min(len_final1, len_final1)
#print len_init, len_final

#print("Filename \t Length Original \t Final \t l_in+l_f \t idx_in + inx_f \n")
filenames = sorted((glob.glob(dirpath[0]+"/*.txt"))+ glob.glob(dirpath[1]+"/*.txt"))

for f in filenames:
	crop_wf(f, len_init, len_final)
#	print f

