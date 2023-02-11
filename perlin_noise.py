import random
import math
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

class Perlin_Noise():
	def __init__(self, sizes = (21,21), mesh_size = (4,4)):
		self.sizes = sizes #Width / Height
		self.dim = len(sizes)
		self.mesh_size = mesh_size
		
		self.grid = [[0 for col in range(sizes[1])] for row in range(sizes[0])]
		self.grad_grid = [[0 for col in range(sizes[1])] for row in range(sizes[0])]
		
		for i in range(0,self.sizes[1],self.mesh_size[1]):
			for j in range(0,self.sizes[0],self.mesh_size[0]):
				self.grad_grid[i][j] = get_random_vector(self.dim)
		

		for i in range(sizes[1]):
			for j in range(sizes[0]):
				self.grid[i][j] = self.point_perlin(j,i)
				

	def save_as_image(self,file_name,file_path = ''):
		"""
		Saves the perlin_noise as a black and white png image
		
		Args:
			file_name: Name of the saved image (do not include the .png extension) 
			file_path: Path to the file where the image should be saved (empty string for current directory)
			
		Returns:
			nothing
		"""
		
		local_grid = self.scale_grid(0,256)
		
		im = Image.new(mode="L", size=self.sizes)
		pixels = im.load()
		
		for i in range(self.sizes[1]):
			for j in range(self.sizes[0]):
				pixels[i,j] = int(local_grid[i][j])
				
		im.save(f'{file_path}/{file_name}.png')
	
	def point_perlin(self,x,y):
		"""
		Generate the perlin noise of a point based on the grad_grid and points coordinates
		
		Args:
			x: X coordinate
			x: Y coordinate
			
		Returns:
			value: Value of the noise at this point
		"""
		
		mesh_x0 = x  - (x % self.mesh_size[0])
		mesh_x1 = min(mesh_x0 + self.mesh_size[0],self.sizes[1]-1)
		mesh_y0 = y  - (y % self.mesh_size[1])
		mesh_y1 = min(mesh_y0 + self.mesh_size[1],self.sizes[0]-1)

		dx = (x - mesh_x0)/self.mesh_size[0]
		dy = (y - mesh_y0)/self.mesh_size[1]

		n0 = dot_product(self.grad_grid[mesh_y0][mesh_x0], [(-mesh_x0+x)/self.mesh_size[0], (-mesh_y0+y)/self.mesh_size[1]]);
		n1 = dot_product(self.grad_grid[mesh_y0][mesh_x1], [(-mesh_x1+x)/self.mesh_size[0], (-mesh_y0+y)/self.mesh_size[1]]);
		ix0 = interpolate(n0, n1, dx);

		n0 = dot_product(self.grad_grid[mesh_y1][mesh_x0], [(-mesh_x0+x)/self.mesh_size[0], (-mesh_y1+y)/self.mesh_size[1]]);
		n1 = dot_product(self.grad_grid[mesh_y1][mesh_x1], [(-mesh_x1+x)/self.mesh_size[0], (-mesh_y1+y)/self.mesh_size[1]]);
		ix1 = interpolate(n0, n1, dx);
		
		value = interpolate(ix0, ix1, dy);
		return value
	
	def scale_grid(self,low = 0, high = 1):
		"""
		Scales the self.grid attribute so that the data is now fitting in a certain range.
		Default range of self.grid is [-1:1]
		
		Args:
			low: Lower limit
			high: Upper limit
			
		Returns:
			scaled_grid: 2D array of the scaled self.grid
		"""
		
		scaled_grid = [[0 for col in range(self.sizes[1])] for row in range(self.sizes[0])]
		
		minimum = min([min(x) for x in self.grid])
		maximum = max([max(x) for x in self.grid])
		
		for i in range(self.sizes[1]):
			for j in range(self.sizes[0]):
				scaled_grid[i][j] = (((self.grid[i][j]-minimum)*(high-low))/(maximum-minimum))+low
				
		return scaled_grid
		
	def show_voxel_map(self, flatten_level = 1):
		"""
		Displays a voxel map of the generated perlin_noise
		Should not be used with sizes bigger than 50*50, it can become really laggy (prefer show_map in this case)
		
		Args:
			flatten_level: By how much should the data be flattened (default 1 = untouched).
			
		Returns:
			nothing
		"""
		
		# define the coordinates system
		x, y, z = np.indices((self.sizes[1],self.sizes[0], self.sizes[0]))
		local_grid = self.scale_grid(0,1/flatten_level)
		
		# generate all the cubes from the local_grid
		cubelist = []
		for i in range(self.sizes[1]):
			for j in range(self.sizes[0]):
				cube = (x == i) & (y == j) & (z <= (math.floor((self.sizes[0]-3)*local_grid[i][j])))
				cubelist.append(cube)
					
		# combine the objects into a single boolean array
		voxelarray = np.bitwise_or.reduce(cubelist)
		
		# set the colors of each object (unused here)
		colors = np.empty(voxelarray.shape, dtype=object)
		
		# plot everything
		ax = plt.figure().add_subplot(projection='3d')
		ax.voxels(voxelarray, facecolors='green', edgecolor='k')

		plt.show()
	
	def show_map(self, flatten_level = 1):
		"""
		Displays a 3d map of the generated perlin_noise.
		
		Args:
			flatten_level: By how much should the data be flattened (default 1 = untouched).
			
		Returns:
			nothing
		"""
		
		local_grid = self.scale_grid(0,1/flatten_level)
		
		fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

		X = np.arange(self.sizes[0])
		Y = np.arange(self.sizes[1])
		X, Y = np.meshgrid(X, Y)
		Z = np.array(local_grid)
		
		surf = ax.plot_surface(X, Y, Z, cmap=cm.gist_earth,
                       linewidth=0, antialiased=False)
		ax.set_zlim(0,1.5)
		plt.show()

def dot_product(u,v):
	r = 0
	for n in range(len(u)):
		r += u[n]*v[n]
	return r

def get_random_vector(n):
	"""
	Returns a unitary vector of the specified size (Current max size : 2)
	"""
	
	if n == 1:
		return random.choice([[1],[-1]])
	
	if n == 2:
		angle_1 = random.random()*2*math.pi
		s = math.sqrt(2)
		return [1*math.cos(angle_1),1*math.sin(angle_1)]
		#return random.choice([[s,s],[s,-s],[-s,-s],[-s,s]])
		
def smoothstep(w):
	if (w <= 0.0):
		return 0.0
	if (w >= 1.0):
		return 1.0
	return 6*w**5 - 15*w**4 + 10*w**3

def interpolate(a, b, weight):
	return a + (b - a) * smoothstep(weight)

 


if __name__ == '__main__':
	p = Perlin_Noise([301,301],[50,50])
	#p.save_as_image("image","mon_perlin1")
	p.show_map(4)
