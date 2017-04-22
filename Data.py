import SimpleITK as sitk
import numpy as np
import csv
import os
import re
import random
from PIL import Image
import matplotlib.pyplot as plt
from skimage import measure, morphology
import scipy

from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class DataHandler(object):
	def __init__(self):
		pass


	def load_itk_image(self, filename):
		'''
		Takes in a string to the file location
		Returns:
			numpyImage: a numpy array of the image raw data
			numpyOrigin:
			numpySpacing: the spacing conversion between voxels in the
				x, y and z direction and real world lengths
		'''
		itkimage = sitk.ReadImage(filename)
		numpyImage = sitk.GetArrayFromImage(itkimage)

		numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
		numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

		return numpyImage, numpyOrigin, numpySpacing

	def readCSV(self, filename):
		'''
		Takes in a string to the CSV file location
		Returns a list of lines in the csv file
		'''
		lines = []
		with open(filename, "rb") as f:
			csvreader = csv.reader(f)
			for line in csvreader:
				lines.append(line)
		return lines

	def worldToVoxelCoord(self, worldCoord, origin, spacing):
		'''
		Converts a distance in world coordinates to voxel coordinates
		'''

		stretchedVoxelCoord = np.absolute(worldCoord-origin)
		voxelCoord = stretchedVoxelCoord / spacing
		return voxelCoord

	def normalizePlanes(self, npzarray):
		maxHU = 400.0
		minHU = -1000.0
		npzarray = (npzarray-minHU) / (maxHU - minHU)
		npzarray[npzarray>1] = 1
		npzarray[npzarray<0] = 0
		return npzarray

	def generateSamples(self, imageDir, csv, savedir, targetWorldWidth, targetVoxelWidth):
		#get a list of files and candidates
		mhd = re.compile(r".*\.mhd")
		files = [f for f in os.listdir(imageDir) if mhd.match(f) != None]
		cands = self.readCSV(csv)

		count = {'0':0, '1':0}

		#organize candidates into a dictionary with patient id as the key
		candDict = {}
		for cand in cands[1:]:
			if not candDict.has_key(cand[0]):
				candDict[cand[0]] = []
			candDict[cand[0]].append(cand)

		candDictVoxel = {}
		#extract candidates
		for f in reversed(files):
			print("Extracting from file {}".format(f))
			print("Candidates")
			if candDict.has_key(f[0:-4]): #if the patient has no candidates, skip
				img, origin, spacing = self.load_itk_image("{}{}".format(imageDir, f))  #load image
				voxelWidth = targetWorldWidth / spacing  #calculate the width of the box to extract

				#extract each candidate in patient f
				for cand in candDict[f[0:-4]]:

					worldCoord = np.asarray([float(cand[3]), float(cand[2]), float(cand[1])])
					voxelCoord = self.worldToVoxelCoord(worldCoord, origin, spacing)

					#make a dictionary of candidates and their voxel coordinates for
					#use later in random sample generation
					if not candDictVoxel.has_key(cand[0]):
						candDictVoxel[cand[0]] = []
					candDictVoxel[cand[0]] += [voxelCoord]

					count[cand[4]] += 1 #count the number of candidates that are true nodules and are nt
					patch = self.extractPatch(voxelCoord, voxelWidth, img) #extract the patch

					#for z in range(patch.shape[0]):
					#	plt.imshow(patch[z], cmap='gray')
						#plt.show()

					#resize the patch to targetVoxelWidth
					if(patch.shape[0] != 0) and (patch.shape[1] != 0) and (patch.shape[2] != 0):
						patch = scipy.ndimage.interpolation.zoom(patch, (float(targetVoxelWidth[0])/patch.shape[0],
																	float(targetVoxelWidth[1])/patch.shape[1],
																	float(targetVoxelWidth[2])/patch.shape[2]))

						#save sample
						np.ndarray.tofile(patch, "{}{}/{}-{}".format(savedir, cand[4], cand[0], count[cand[4]]))

				#extract random samples
				print("Random")
				for i in range(300):
					#generate random coordinates
					coord = np.array([int(random.uniform(voxelWidth[0], img.shape[0]-voxelWidth[0])),
									int(random.uniform(voxelWidth[1], img.shape[1]-voxelWidth[1])),
									int(random.uniform(voxelWidth[2], img.shape[2]-voxelWidth[2]))
						])
					bad = False
					low = coord - voxelWidth
					high = coord + voxelWidth

					#check if the coordinates conflict with any known candidates
					for cand in candDictVoxel[f[0:-4]]:
						if (low[0]<cand[0]<high[0]) and (low[1]<cand[1]<high[1]) and (low[2]<cand[2]<high[2]):
							bad = True
							break

					#if not, then extract the sample, resize, and save
					if not bad:
						patch = self.extractPatch(coord, voxelWidth, img)
						patch = scipy.ndimage.interpolation.zoom(patch, (float(targetVoxelWidth[0])/patch.shape[0],
																	float(targetVoxelWidth[1])/patch.shape[1],
																	float(targetVoxelWidth[2])/patch.shape[2]))
						np.ndarray.tofile(patch, "{}{}/{}-{}".format(savedir, 2, f[0:-4], i))

	def generateRandomSamples():
		pass

	def extractPatch(self, voxelCoord, voxelWidth, numpyImage):
		patch = numpyImage[int(voxelCoord[0]-voxelWidth[0]/2):int(voxelCoord[0]+voxelWidth[0]/2),
							int(voxelCoord[1]-voxelWidth[1]/2):int(voxelCoord[1]+voxelWidth[1]/2),
							int(voxelCoord[2]-voxelWidth[2]/2):int(voxelCoord[2]+voxelWidth[2]/2)]
		patch = self.normalizePlanes(patch)
		return patch

	def plot_3d(self, image, threshold=-300):
    	# Position the scan upright,
    	# so the head of the patient would be at the top facing the camera
		p = image.transpose(2,1,0)

		verts, faces = measure.marching_cubes_classic(p, threshold)

		fig = plt.figure(figsize=(10, 10))
		ax = fig.add_subplot(111, projection='3d')

		# Fancy indexing: `verts[faces]` to generate a collection of triangles
		mesh = Poly3DCollection(verts[faces], alpha=0.70)
		face_color = [0.45, 0.45, 0.75]
		mesh.set_facecolor(face_color)
		ax.add_collection3d(mesh)

		ax.set_xlim(0, p.shape[0])
		ax.set_ylim(0, p.shape[1])
		ax.set_zlim(0, p.shape[2])

		plt.show()


def main():
	handler = DataHandler()
	'''
	shape = {}
	spacingg = {}
	mhd = re.compile(r".*\.mhd")
	files = [f for f in os.listdir("/media/amos/My Passport/Data/luna/subset0/subset0/") if mhd.match(f) != None]
	for f in files:
		img, origin, spacing = handler.load_itk_image("/media/amos/My Passport/Data/luna/subset0/subset0/{}".format(f))
		if not shape.has_key(str(img.shape)):
			shape[str(img.shape)] = 0;
		if not spacingg.has_key(str(spacing)):
			spacingg[str(spacing)] = 0;
		shape[str(img.shape)] += 1;
		spacingg[str(spacing)] += 1;

	print(shape)
	print(spacingg)
	'''

	handler.generateSamples("/media/amos/My Passport/Data/luna/subset0/", "data/csv/candidates.csv", "/media/amos/My Passport/Data/luna/samples/", (15,15,15), (15,20,20))
	fileName = "1.3.6.1.4.1.14519.5.2.1.6279.6001.148447286464082095534651426689"
	cands = handler.readCSV("data/csv/candidates.csv")
	numpyImage, numpyOrigin, numpySpacing = handler.load_itk_image("data/images/{}.mhd".format(fileName))
	np.save("numpyimg", numpyImage)
	a = np.fromfile("samples/1/1.3.6.1.4.1.14519.5.2.1.6279.6001.148447286464082095534651426689-1").reshape(15,20,20)
	print(a.shape)
	for z in range(a.shape[0]):
		plt.imshow(a[z], cmap='gray')
		#plt.show()

	#handler.plot_3d(numpyImage)
	print(numpyImage)
	print(numpyOrigin)
	print(numpyImage.shape)
	print(numpySpacing)
	for cand in cands[1:]:

		if cand[0] == fileName and cand[4] == '1':
			worldCoord = np.asarray([float(cand[3]), float(cand[2]), float(cand[1])])
			voxelCoord = handler.worldToVoxelCoord(worldCoord, numpyOrigin, numpySpacing)
			voxelWidth = 65
			print(voxelCoord)
			patch = numpyImage[100:150, int(voxelCoord[1]-voxelWidth/2):int(voxelCoord[1]+voxelWidth/2),int(voxelCoord[2]-voxelWidth/2):int(voxelCoord[2]+voxelWidth/2)]
			patch = handler.normalizePlanes(patch)

			outputDir = "patches/"
			#plt.imshow(patch, cmap='gray')
			#plt.show()
			#plt.imshow(scipy.ndimage.interpolation.rotate(patch, 30), cmap='gray')
			#plt.show()
			#plt.imshow(scipy.misc.imresize(patch, (40,50,50), interp='bilinear'), cmap='gray')
			#plt.show()
			print(patch.shape)
			patch = scipy.ndimage.interpolation.zoom(patch, (1.5,1.5,1.6))
			print(patch)
			print(patch.shape)
			#plt.imshow(scipy.misc.imresize(patch, (85,85,85), interp='bilinear'), cmap='gray')
			#plt.show()
			#Image.fromarray(patch*255).convert('L').save("bla")
main()
