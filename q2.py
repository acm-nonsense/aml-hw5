from sklearn.cluster import KMeans
import scipy as sp
import math
import numpy.linalg as la

simple_y = sp.genfromtxt('simple.data',delimiter=',',skip_header=1,usecols=0,dtype=int)
simple_x = sp.genfromtxt('simple.data',delimiter=',',skip_header=1,usecols=(1,2),dtype=float)#.reshape(-1,1)

K = 2
D = simple_x.shape[1]
N = simple_x.shape[0]

print(K,D,N)

def calc_covar_matrix(means,data):
	pass


# Do KMeans
model = KMeans(n_clusters=K).fit(simple_x)
pred_y = model.predict(simple_x)

# Seed Params
mus = model.cluster_centers_
priors = sp.vectorize(lambda y: len(simple_x[sp.where(pred_y == y)])/float(len(simple_x)))(range(K))
posts = sp.zeros((K,N))
for c in range(K):
	for i in range(N):
		posts[c,i] = 1 if pred_y[i] == c else 0

covars = sp.zeros((K,D,D))

# print(priors)
# print(mus)

# print(simple_x.shape)

# covars = sp.vectorize(lambda c: sp.sum(sp.vectorize(lambda d: sp.dot(simple_x[d]-mus[c],(simple_x[d]-mus[c]).T) / (N * priors[c]))(range(N))))(range(K))
for c in range(K):
	for j in range(D):
		for k in range(D):
			for i in range(N):
				covars[c,j,k] += posts[c,i]/N/priors[c]	\
				*(simple_x[i,j]-mus[c,j])	\
				*(simple_x[i,k]-mus[c,k])

def pfn(i,c,k,covmats,data,clusters):
	return 1.0/math.sqrt(((2*math.pi)**k)*la.det(covmats[c]))\
		*math.exp(\
			-0.5*sp.dot(\
				sp.dot((data[i]-clusters[c]).T,la.inv(covmats[c]))\
				,(data[i]-clusters[c])\
				)\
			)\

posts = sp.zeros((K,N))
for c in range(K):
	for i in range(N):
		norm = 0
		for j in range(K):
			norm += pfn(i,j,K,covars,simple_x,mus)*priors[c]
		posts[c,i] = pfn(i,c,K,covars,simple_x,mus)*priors[c]/norm

print(covars)
print(posts)

'''
for i in range(len(mus)):
	covars.append(calc_covar_matrix(
		mus[i],
		simple_x[sp.where(pred_y == i)]))
'''