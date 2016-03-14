from sklearn.cluster import KMeans
import scipy as sp
import math
import numpy.linalg as la
from scipy.stats import multivariate_normal as mvn
from PIL import Image
from numpy.core.umath_tests import matrix_multiply as mm

img_file_name = "nature"
img = Image.open(img_file_name+".jpg")
print(sp.array(img).size)
print(img.size[1]*img.size[0]*3)
simple_x = sp.array(img).reshape(img.size[1]*img.size[0], 3)
print(simple_x)
simple_y = sp.genfromtxt('simple.data',delimiter=',',skip_header=1,usecols=0,dtype=int)
# simple_x = sp.genfromtxt('simple.data',delimiter=',',skip_header=1,usecols=(1,2),dtype=float)#.reshape(-1,1)

K = 10
D = simple_x.shape[1]
N = simple_x.shape[0]
ll_old = 0
diff_thresh = 300

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
		ys = simple_x - mus[j,:]
		covars[j] = (posts[j,:,None,None] \
		* mm(ys[:,:,None], \
		ys[:,None,:])).sum(axis=0)
	covars /= posts.sum(axis=1)[:,None,None]

def mStep():
	global model
	global pred_y
	global mus
	global priors
	global posts
	global covars
	
	print("priors")
	priors = sp.sum(posts,axis=1)/N

	print("mus")
	mus = sp.dot(posts, simple_x)
	mus /= posts.sum(1)[:, None]


	print("Covars")
	for j in range(K):
		ys = simple_x - mus[j,:]
		covars[j] = (posts[j,:,None,None] \
		* mm(ys[:,:,None], \
		ys[:,None,:])).sum(axis=0)
	covars /= posts.sum(axis=1)[:,None,None]
	
def pfn(data,cluster):
	# print(math.exp(-0.01*sp.dot((data[i]-clusters[c]).T,data[i]-clusters[c])))
	print("-----")
	print(data)
	return 1.0/math.sqrt((2*math.pi)**D)\
		*math.exp(-0.01*sp.dot((data-cluster).T,(data-cluster)))

def eStep():
	global posts
	global covars
	global mus
	global priors

	print("posts")
	posts = sp.zeros((K,N))
	for j in range(K):
			posts[j,:] = priors[j] * mvn(mus[j],covars[j], allow_singular=True).pdf(simple_x)
	posts /= posts.sum(0)
	
def ll():
	global posts
	global covars
	global mus
	global priors
	global ll_old
	ll_new = 0
	for prior, mu, covar in zip(priors, mus, covars):
		ll_new += prior*mvn(mu, covar, allow_singular=True).pdf(simple_x)
	ll_new = sp.log(ll_new).sum()
	print("Diff is {}".format(ll_new - ll_old))
	if abs(ll_new - ll_old) < diff_thresh:
		return False
	ll_old = ll_new
	
	return True
				
print("seeding")
seedVars()
# mStep()
iters = 0
for i in range(100):
	print("------ STEP {} -------".format(i))
	print("e step")
	eStep()
	print("m step")
	mStep()
	iters += 1
	if not ll():
		break
	
	
newImg1 = Image.new('RGB', (img.size[0],img.size[1]))
pix = newImg1.load()
print(img.size[0]*img.size[1])
for i in range (img.size[0]):
	for j in range (img.size[1]):
		mean_index = posts[:,j*img.size[0]+i].argmax(axis=0)
		v = mus[mean_index,:]
		pix[i, j]=tuple(sp.vectorize(lambda x: int(x))(v))
newImg1.save(img_file_name + ("-em-k{}-{}.png".format(K,iters)))
print(priors)
print(posts)
