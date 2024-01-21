import numpy as np
import kmeans
import common
import naive_em
import em

#########Calculate Minimum Cost for One EM Iteration###################

X = np.loadtxt("toy_data.txt")

seeds = {0:0 , 1:0 , 2:0 , 3:0 , 4:0}
means = {1:seeds , 2:seeds , 3:seeds , 4:seeds}

"""
for k in means:
    for seed in means[k]:
        GM = common.init(X,k,seed)
        iter = kmeans.run(X,GM[0],GM[1])
        means[k][seed] = iter[2]
        #print("Means: " + str(k) + '; Seed: ' + str(seed) + '; Cost: ' + str(iter[2]))
    means[k] = ( min(means[k].values()) , iter )
"""

#print(means)

##################Plot One Iteration###################
'''
#print("Data Dimensions = " + str(X.shape))
GM = common.init(X,3,0)
#print(GM[1][0:5,:])
#common.plot(X,GM[0],GM[1],"Initializtion: (Means = 3 ; Seed = 0)")
post = kmeans.estep(X,GM[0])
#print(post[0:5,:])
iter = kmeans.run(X,GM[0],GM[1])
#common.plot(X , iter[0] , iter[1], "One Iteration: (Means = 3 , Seed = 0)")
'''

###################Test Naive EM########################
'''
Y = np.loadtxt("toy_data.txt")
init = common.init(Y,3,0)
post = naive_em.estep(Y,init[0])
#print(init[0])
#print(post[0][:5,:])
#print(post[1])
mix = naive_em.mstep(X,post[0])
#print(mix)
'''

###################Calculate maximum likelihood for Naieve EM###############

'''
for k in means:
    for seed in means[k]:
        GM = common.init(X,k,seed)
        loss = naive_em.run(X,GM[0],GM[1])
        means[k][seed] = loss[2]
    means[k] = max(means[k].values())

print(means)
'''

'''
for seed in means[4]:
    GM = common.init(X,4,seed)
    loss = naive_em.run(X,GM[0],GM[1])
    sds = means[4]
    sds[seed] = loss[2]

print(max(sds.values()))
'''


############Calculate BiC##############################################
'''
for k in means:
    GM = common.init(X,k,0)
    loss = naive_em.run(X,GM[0],GM[1])
    BiC = common.bic(X,loss[0],loss[2])
    means[k] = (loss[2],BiC)

print(means)
'''

###############Test Unobserved EM e-step#############
'''
#X = np.loadtxt('test_incomplete.txt')
X = np.loadtxt('netflix_incomplete.txt')
init = common.init(X,3,0)
est = em.estep(X,init[0])
print(X[-5:])
print(est[0][-5:])
print(est[1])
'''


##############Test Unobserved EM m-step##############
'''
X = np.loadtxt('test_incomplete.txt')
init = common.init(X,3,0)
est = em.estep(X,init[0])
#print(est[0])
mix = em.mstep(X,est[0],init[0])
#print(mix[0])
#print(mix[1])
#print(mix[2])
print(mix[3][:5])
print(X[:5])
'''

##############Compute Max LL for incomplete data#################
X = np.loadtxt('netflix_incomplete.txt')
'''
means = {1:seeds,12:seeds}
for k in means:
    for seed in seeds:
        GM = common.init(X,k,seed)
        trial = em.run(X,GM[0],GM[1])
        means[k][seed] = trial[2]
    means[k] = np.max(means[k].values)
print(means)
'''
print([X == 0])