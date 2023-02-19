import random
import math
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

class Perlin_Noise():
	def __init__(self, sizes, mesh_size):
		self.sizes = sizes #Width / Height
		self.dim = len(sizes)
		self.mesh_size = mesh_size
		
		self.grid = np.zeros(sizes)
		
		#Create a grad_grid with a size that allows each point from the grid to have four neighbour vectors
		self.grad_grid = np.zeros(np.append(np.array(sizes) + np.array(mesh_size) + 1 - (np.array(sizes)%np.array(mesh_size)), self.dim))

		for y in range(0,self.grad_grid.shape[1],self.mesh_size[1]):
			for x in range(0,self.grad_grid.shape[0],self.mesh_size[0]):
				self.grad_grid[x,y] = get_random_vector(self.dim)
		
		
		for y in range(sizes[1]):
			for x in range(sizes[0]):
				self.grid[x,y] = self.point_perlin(x,y)
				

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
				pixels[i,j] = int(local_grid[i,j])
				
		im.save(f'{file_path}/{file_name}.png')
	
	def point_perlin(self,x,y):
		"""
		Generate the perlin noise of a point based on the grad_grid and points coordinates
		
		Args:
			x: X coordinate
			y: Y coordinate
			
		Returns:
			value: Value of the noise at this point
		"""
		
		mesh_x0 = x  - (x % self.mesh_size[0])
		mesh_x1 = min(mesh_x0 + self.mesh_size[0],self.grad_grid.shape[0]-1)
		mesh_y0 = y  - (y % self.mesh_size[1])
		mesh_y1 = min(mesh_y0 + self.mesh_size[1],self.grad_grid.shape[1]-1)

		dx = (x - mesh_x0)/self.mesh_size[0]
		dy = (y - mesh_y0)/self.mesh_size[1]

		n0 = dot_product(self.grad_grid[mesh_x0,mesh_y0], [(-mesh_x0+x)/self.mesh_size[0], (-mesh_y0+y)/self.mesh_size[1]])
		n1 = dot_product(self.grad_grid[mesh_x1,mesh_y0], [(-mesh_x1+x)/self.mesh_size[0], (-mesh_y0+y)/self.mesh_size[1]])
		ix0 = interpolate(n0, n1, dx)

		n0 = dot_product(self.grad_grid[mesh_x0,mesh_y1], [(-mesh_x0+x)/self.mesh_size[0], (-mesh_y1+y)/self.mesh_size[1]])
		n1 = dot_product(self.grad_grid[mesh_x1,mesh_y1], [(-mesh_x1+x)/self.mesh_size[0], (-mesh_y1+y)/self.mesh_size[1]])
		ix1 = interpolate(n0, n1, dx)
		
		value = interpolate(ix0, ix1, dy)
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
		
		scaled_grid = np.zeros(self.sizes)
		
		minimum = np.amin(self.grid)
		maximum = np.amax(self.grid)
		
		for y in range(self.sizes[1]):
			for x in range(self.sizes[0]):
				scaled_grid[x,y] = (((self.grid[x,y]-minimum)*(high-low))/(maximum-minimum))+low
				
		return scaled_grid
		
	def show_voxel_map(self, flatten_level = 1):
		"""
		Displays a voxel map of the generated perlin_noise
		Should not be used with sizes bigger than 50*50, it can become really laggy (prefer show_map in this case)
		
		Args:
			flatten_level: By how much the data should be flattened (default 1 = untouched).
			
		Returns:
			nothing
		"""
		
		parameters = {
			'stone_height': 1,
			'water': True,
			'water_level': 2,
		}
		
		# define the coordinates system
		x, y, z = np.indices((self.sizes[1],self.sizes[0], self.sizes[0]))
		local_grid = self.scale_grid(0,1/flatten_level)
		
		# generate all the cubes from the local_grid
		cubelist = []
		for i in range(self.sizes[1]):
			for j in range(self.sizes[0]):
				cube = (x == i) & (y == j) & (z <= (math.floor((self.sizes[0])*local_grid[j,i])))
				cubelist.append(cube)
					
		# combine the objects into a single boolean array
		voxelarray = np.bitwise_or.reduce(cubelist)
		
		# set the colors of each object (unused here)
		colors = np.zeros(voxelarray.shape + (3,))
		
		#set all colors to green
		colors[:,:,:, 0] = .06
		colors[:,:,:, 1] = .35
		colors[:,:,:, 2] = .10
		
		
		if parameters['water']:
			#set all blocks with z<=water_level to blue		
			for x in range(colors.shape[0]):
				for y in range(colors.shape[1]):
					for z in range(parameters['water_level']):
						voxelarray[x,y,z] = True
						voxelarray[x,y,z] = True
						voxelarray[x,y,z] = True
						if voxelarray[x,y,min(z+1,voxelarray.shape[2]-1)] == False: #if there is no block over
							colors[x,y,:z+1, 0] = 0
							colors[x,y,:z+1, 1] = 0
							colors[x,y,:z+1, 2] = .5
		
		# plot everything
		ax = plt.figure().add_subplot(projection='3d')
		ax.voxels(voxelarray, facecolors = colors,edgecolor='k')

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
		
def smoothstep(w):
	if (w <= 0.0):
		return 0.0
	if (w >= 1.0):
		return 1.0
	return 6*w**5 - 15*w**4 + 10*w**3

def interpolate(a, b, weight):
	return a + (b - a) * smoothstep(weight)

 


if __name__ == '__main__':
	p = Perlin_Noise([30,30],[20,20])
	#p.save_as_image("image","mon_perlin1")
	p.show_voxel_map(3)
	#p.show_map(3)
