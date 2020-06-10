import numpy as np
import glob, os
import matplotlib.pyplot as plt

def write_data(outfile, data):
	
	output_file = open(outfile,'w')
	hdr = '# Time \t h_plus \t h_cross \n'
	np.savetxt(output_file, data, header=hdr, delimiter='\t', newline='\n')
	output_file.close()

def crop_wf(filename, len_init, len_final):

	time, hp, hx = np.loadtxt(filename, usecols=(0,1,2,), unpack=True)
	amp = abs(hp+1.j*hx)
	phase = -np.unwrap(np.angle(hp + 1.j*hx))
	
	idx =np.where(amp== np.amax(amp))
	t_ref = time[idx]
	if t_ref-time[0]>=400:
		time = time-t_ref
		data = np.array((time, hp,hx))
	
		idx_init = np.where(time==0.0)[0]-len_init
		idx_fnl = np.where(time==0.0)[0]+len_final

		data = data.T[idx_init:idx_fnl+1]
		outfile = os.path.join("../FilteredData", os.path.basename(filename))
		write_data(outfile, data)
	#plt.plot(data[:,0], data[:,1])
	#plt.show()
	#plt.close()
		print("{} \t {} \t {} \t {} \t{} \n".format(filename, len(time),np.shape(data)[0], len_init+len_final, idx_init-idx_fnl))
	
	

def find_shortestwf(dirname):
	wf = sorted(glob.glob(os.path.join(dirname,'*.txt')))
	vin = []
	vf = []
	filenames = []
	for f in wf:
		t, hp, hx = np.genfromtxt(f, unpack=True, usecols=(0,1,2))
		amp = abs(hp+1.j*hx)
		idx =np.where(amp== np.amax(amp))
		if t[idx]-t[0]>=400:
			vin.append(idx)
			vf.append(len(amp)-idx[0]-1)
			filenames.append(f)
	#print sorted(vin)[0:10], sorted(vf)[0:10]
	
	#for i in sorted(vin):
	#	if i>100:
	#		len_init = i
	#		break
	len_init = sorted(vin)[0]
	len_final = sorted(vf)[0]
	return len_init, len_final

dirpath =[ "../Waveform_txtFiles/GT", "../Waveform_txtFiles/SXS"]
wfpath = ["../Waveform_txtFiles", "../Waveform_txtFiles"]
len_init1, len_final1 = find_shortestwf(dirpath[0])
len_init2, len_final2 = find_shortestwf(dirpath[1])
print len_init1,len_final1, len_init2,len_final2

len_init = min(len_init2, len_init1)
len_final = min(len_final1, len_final1)
print len_init, len_final
print("Filename \t Length Original \t Final \t l_in+l_f \t idx_in + inx_f \n")
filenames = sorted((glob.glob(dirpath[0]+"/*.txt"))+ glob.glob(dirpath[1]+"/*.txt"))

for f in filenames:
	crop_wf(f, len_init, len_final)
#	print f

