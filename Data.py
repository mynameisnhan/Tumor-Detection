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

	def load_samples(self, sampleDir, shape):
		print("Loading false nodules")
		falseNodules = [(np.fromfile("{}0/{}".format(sampleDir, f)).reshape(shape), 1) for f in os.listdir("{}0/".format(sampleDir))]
		print("Loading true nodules")
		trueNodules = [(np.fromfile("{}1/{}".format(sampleDir, f)).reshape(shape), 1) for f in os.listdir("{}1/".format(sampleDir))]
		print("Loading random samples")
		randomSamples = [(np.fromfile("{}2/{}".format(sampleDir, f)).reshape(shape), 0) for f in os.listdir("{}2/".format(sampleDir))]
		samples = falseNodules + trueNodules + randomSamples
		random.shuffle(samples)
		xs = [x[0] for x in samples]
		ys = [y[1] for y in samples]

		return np.asarray(xs), np.asarray(ys)
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
	files = [f for f in os.listdir("/media/amos/My Passport/Data/luna/subset0/") if mhd.match(f) != None]

	z = []
	x = []
	y = []

	for f in files:
		print(f)
		img, origin, spacing = handler.load_itk_image("/media/amos/My Passport/Data/luna/subset0/{}".format(f))
		if not shape.has_key(str(img.shape)):
			shape[str(img.shape)] = 0;
		if not spacingg.has_key(str(spacing)):
			spacingg[str(spacing)] = 0;
		shape[str(img.shape)] += 1;
		spacingg[str(spacing)] += 1;
		z += [spacing[0]]
		x += [spacing[1]]
		y += [spacing[2]]
	'''


	z = [1.7999999523162842, 2.5, 2.5, 1.25, 0.625, 1.7999999523162842, 2.5, 0.5, 2.5, 0.69999998807907104, 1.25, 1.25, 1.0, 0.5, 2.5, 1.0, 2.0, 0.5, 1.0, 2.5, 2.5, 2.5, 1.0, 1.0, 1.25, 1.7999999523162842, 2.5, 2.5, 1.7999999523162842, 1.0, 1.25, 1.0, 2.5, 1.25, 1.7999999523162842, 1.0, 0.625, 1.25, 1.25, 2.5, 1.0, 2.5, 2.5, 0.625, 0.625, 1.25, 1.25, 1.2499998807907104, 0.625, 1.0, 2.5, 2.5, 2.5, 2.5, 2.5, 0.625, 0.625, 1.25, 1.7999999523162842, 2.5, 1.0000001192092896, 2.5, 1.0, 1.7999999523162842, 1.0, 2.5, 2.5, 0.625, 0.625, 1.25, 1.25, 2.5, 2.5, 2.5, 0.69999998807907104, 0.75, 2.5, 2.5, 0.625, 2.5, 1.25, 1.0, 2.5, 2.5, 2.0, 1.0, 2.0, 0.5, 0.69999998807907104]
	x = [0.7421875, 0.7617189884185791, 0.7421879768371582, 0.54882800579071045, 0.7421879768371582, 0.72265625, 0.55664098262786865, 0.607421875, 0.703125, 0.6171875, 0.61523401737213135, 0.859375, 0.7421875, 0.513671875, 0.7226560115814209, 0.572265625, 0.6640620231628418, 0.73046875, 0.78125, 0.703125, 0.7421879768371582, 0.61523401737213135, 0.681640625, 0.78125, 0.69335901737213135, 0.7421875, 0.69726598262786865, 0.9765620231628418, 0.68359375, 0.6171875, 0.65429699420928955, 0.61328125, 0.625, 0.65429699420928955, 0.6640625, 0.611328125, 0.6054689884185791, 0.6210939884185791, 0.9492189884185791, 0.6171879768371582, 0.7109375, 0.6640620231628418, 0.78125, 0.5859379768371582, 0.78125, 0.5859379768371582, 0.703125, 0.52539098262786865, 0.703125, 0.681640625, 0.703125, 0.5859379768371582, 0.859375, 0.703125, 0.78125, 0.703125, 0.625, 0.59570300579071045, 0.64453125, 0.78125, 0.67578125, 0.78125, 0.78125, 0.5859375, 0.578125, 0.6445310115814209, 0.703125, 0.8203120231628418, 0.7226560115814209, 0.65429699420928955, 0.94335901737213135, 0.7421879768371582, 0.859375, 0.6445310115814209, 0.537109375, 0.638671875, 0.703125, 0.8203120231628418, 0.703125, 0.8203120231628418, 0.859375, 0.5234375, 0.78125, 0.59960901737213135, 0.6640620231628418, 0.744140625, 0.703125, 0.63671875, 0.65234375]
	y  = [0.7421875, 0.7617189884185791, 0.7421879768371582, 0.54882800579071045, 0.7421879768371582, 0.72265625, 0.55664098262786865, 0.607421875, 0.703125, 0.6171875, 0.61523401737213135, 0.859375, 0.7421875, 0.513671875, 0.7226560115814209, 0.572265625, 0.6640620231628418, 0.73046875, 0.78125, 0.703125, 0.7421879768371582, 0.61523401737213135, 0.681640625, 0.78125, 0.69335901737213135, 0.7421875, 0.69726598262786865, 0.9765620231628418, 0.68359375, 0.6171875, 0.65429699420928955, 0.61328125, 0.625, 0.65429699420928955, 0.6640625, 0.611328125, 0.6054689884185791, 0.6210939884185791, 0.9492189884185791, 0.6171879768371582, 0.7109375, 0.6640620231628418, 0.78125, 0.5859379768371582, 0.78125, 0.5859379768371582, 0.703125, 0.52539098262786865, 0.703125, 0.681640625, 0.703125, 0.5859379768371582, 0.859375, 0.703125, 0.78125, 0.703125, 0.625, 0.59570300579071045, 0.64453125, 0.78125, 0.67578125, 0.78125, 0.78125, 0.5859375, 0.578125, 0.6445310115814209, 0.703125, 0.8203120231628418, 0.7226560115814209, 0.65429699420928955, 0.94335901737213135, 0.7421879768371582, 0.859375, 0.6445310115814209, 0.537109375, 0.638671875, 0.703125, 0.8203120231628418, 0.703125, 0.8203120231628418, 0.859375, 0.5234375, 0.78125, 0.59960901737213135, 0.6640620231628418, 0.744140625, 0.703125, 0.63671875, 0.65234375]


	print(np.percentile(z, 50))
	print(np.percentile(x, 50))
	print(np.percentile(y, 50))



	#handler.generateSamples("/media/amos/My Passport/Data/luna/subset0/", "data/csv/candidates.csv", "/media/amos/My Passport/Data/luna/samples/", (15,15,15), (15,20,20))
	#xs, ys = handler.load_samples("samples/", (1, 16,16,10))


	#for z in range(xs.shape[2]):
		#plt.imshow(xs[0][0][z], cmap='gray')
		#plt.show()



if __name__ == "__main__":
	main()
