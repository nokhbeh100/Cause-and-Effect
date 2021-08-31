import pandas as pd
from datetime import datetime
import torch
from torch.utils.data import Dataset
import os
import numpy as np
from matplotlib import pyplot as plt
from random import randint


def mapTime(x):
	return datetime.strptime(x, "%m/%d/%Y").strftime("%Y%m%d")


class OAI(Dataset):
	"""OAI interface."""

	def __init__(self, root_dir, transform=None, output_cols=None):
		self.root_dir = root_dir
		self.images_dir = os.path.join(root_dir,'images')
		kxr = pd.read_csv(os.path.join(root_dir,'kxrsq01.txt'), sep='\t', header=0, skiprows=[1])
		if (output_cols is None):
			output_cols = ['xrkl']
		

		cols = ['src_subject_id', 'barcdbu', 'side', 'xrosfm'] + output_cols

		kxr = kxr[cols]
		kxr.dropna(inplace=True)


		self.labels = kxr
		self.output_cols = output_cols

		self.transform = transform


	def __len__(self):
		return len(self.labels)

	def __getitem__(self, sampleNo):

		record = self.labels.iloc[sampleNo]
		# 0= neither;1=left; 2=right; 3=bilateral (the image is usually flipped)
		side = 2-record['side']

		cacheLoc = os.path.join(self.images_dir, f"{record['barcdbu'][3:]}-{side}.npy")
		image = np.load(cacheLoc)

		#if randint(0, 1):
			#image = np.fliplr(image)
		#image = image[128:512-128, 128:]

		out = record[self.output_cols].to_numpy()

		if self.transform:
			return self.transform(image.astype(np.float32)), out[0]
		else:
			return image.astype(np.float32), out[0]


# cat backup.txt | grep 01702004 -v > kxrsq01.txt

#ds = OAI('.')

#for i, c in ds:
	#plt.subplot(1,2,1)
	#plt.imshow(i)
	#plt.subplot(1,2,2)
	#plt.hist(i.reshape(-1), bins=20)
	#plt.colorbar()
	#plt.show()
