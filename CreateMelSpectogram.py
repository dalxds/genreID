import librosa
import librosa.display
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt


def full_frame(width=None , height=None):
	mpl.rcParams['savefig.pad_inches']=0
	figsize= None if width is None else(width, height)
	fig = plt.figure(figsize=figsize)
	ax= plt.axes([0,0,1,1], frameon=False)
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	plt.autoscale(tight=True)


if __name__ == '__main__':
	#filename = os.path.abspath("/home/kostas/Desktop/songs/001/001681.mp3")
	#y, sr = librosa.load(filename, mono=True)
	#S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=2048//2)
	#S_dB = librosa.power_to_db(S, ref=np.max)
	#full_frame()
	#librosa.display.specshow(S_dB, sr=sr)
	#plt.savefig('/home/kostas/Desktop/songs/melspecs/001681.png')
	print os.getcwd()
	path=os.getcwd()
	path=path+"/songs"
	os.chdir(path)
	for i in os.listdir(path):
		print i
		newpath="/home/kostas/Desktop/melspecs/"+i
		if not os.path.exists(newpath):
			os.makedirs(newpath)
		os.chdir(path+"/"+i)
		for j in os.listdir(path+"/"+i):
			print j
			filename = os.path.abspath(j)
			y, sr = librosa.load(filename, mono=True , duration=30.0)
			S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=2048//2)
			S_dB = librosa.power_to_db(S, ref=np.max)
			full_frame()
			librosa.display.specshow(S_dB, sr=sr)
			name=j[0:6]
			print name
			plt.savefig(('/home/kostas/Desktop/melspecs/'+i)+'/'+name+'.png')
