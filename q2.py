from sklearn.cluster import KMeans
import scipy as sp
import math
import numpy.linalg as la
from scipy.stats import multivariate_normal as mvn
from PIL import Image

img = Image.open("balloons.jpg")
print(sp.array(img).size)
print(img.size[1]*img.size[0]*3)
simple_x = sp.array(img).reshape(img.size[1]*img.size[0], 3)
print(simple_x)
simple_y = sp.genfromtxt('simple.data',delimiter=',',skip_header=1,usecols=0,dtype=int)
# simple_x = sp.genfromtxt('simple.data',delimiter=',',skip_header=1,usecols=(1,2),dtype=float)#.reshape(-1,1)

K = 5
D = simple_x.shape[1]
N = simple_x.shape[0]


def seedVars():
	global model
	global pred_y
	global mus
	global priors
	global posts
	global covars

	print("Kmeans")
	model = KMeans(n_clusters=K).fit(simple_x)
	pred_y = model.predict(simple_x)

	# Seed Params
	print("Seeding mus")
	mus = model.cluster_centers_
	print("mus shape")
	print(mus.shape)
	print("xs shape")
	print(simple_x.shape)
	print(N,K,D)

	print("Seeding priors")
	priors = sp.vectorize(lambda y: len(simple_x[sp.where(pred_y == y)])/float(len(simple_x)))(range(K))

	print("Seeding posteriors")
	posts = sp.zeros((K,N))
	for c in range(K):
		for i in range(N):
			posts[c,i] = 1 if pred_y[i] == c else 0

	print("Seeding covar mats")
	covars = sp.zeros((K,D,D))
	# print((simple_x[0] - mus[0]).shape)
	for j in range(K):
		for i in range(N):
			ys = sp.reshape(simple_x[i] - mus[j], (D,1))
			covars[j] += posts[j,i] * sp.dot(ys,ys.T)
		covars[j] /= posts[j,:].sum()

def mStep():
	global model
	global pred_y
	global mus
	global priors
	global posts
	global covars
	
	print("priors")
	priors = sp.zeros(K)
	for c in range(K):
		priors[c] = sp.sum(posts[c])/N

	print("mus")
	mus = sp.zeros((K,D))
	for c in range(K):
		for j in range(D):
			for i in range(N):
				mus[c,j] += posts[c,i]/N/priors[c]*simple_x[i,j]


	print("Covars")
	for j in range(K):
		for i in range(N):
			ys = sp.reshape(simple_x[i] - mus[j], (D,1))
			covars[j] += posts[j,i] * sp.dot(ys,ys.T)
		covars[j] /= posts[j,:].sum()

def pfn(i,c,k,covmats,data,clusters):
	# print(math.exp(-0.01*sp.dot((data[i]-clusters[c]).T,data[i]-clusters[c])))
	return 1.0/math.sqrt((2*math.pi)**D)\
		*math.exp(-0.01*sp.dot((data[i]-clusters[c]).T,(data[i]-clusters[c])))

def eStep():
	global posts
	global covars
	global mus
	global priors

	print("posts")
	posts = sp.zeros((K,N))
	for j in range(K):
			posts[j,:] = priors[j] * mvn(mus[j],covars[j]).pdf(simple_x)
	posts /= posts.sum(0)

print("seeding")
seedVars()
# mStep()
for i in range(1):
	print("------ STEP {} -------".format(i))
	print("e step")
	eStep()
	print("m step")
	mStep()

newImg1 = Image.new('RGB', (img.size[0],img.size[1]))
pix = newImg1.load()
print(img.size[0]*img.size[1])
for i in range (img.size[0]):
	for j in range (img.size[1]):
		mean_index = posts[:,j*img.size[0]+i].argmax(axis=0)
		v = mus[mean_index,:]
		pix[i, j]=tuple(sp.vectorize(lambda x: int(x))(v))
newImg1.save("balloons-clustered-em-1.png")
print(priors)
print(posts)

'''
for i in range(len(mus)):
	covars.append(calc_covar_matrix(
		mus[i],
		simple_x[sp.where(pred_y == i)]))
'''