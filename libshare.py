import numpy as np
from scipy.special import logsumexp
import ot
def error(mu1,mu2,pi,pi2):
    difs =create_K(mu1,mu2)
    return ot.emd2(pi,pi2,difs)  
def sinkhorn_np_logspace(K, eps,u=None, v=None, mu_x=None, mu_y=None, n_iter=200,tolu=1e-10):
    if (mu_x is None):
        mu_x = np.ones(K.shape[0]) / K.shape[0]
    if (mu_y is None):
        mu_y = np.ones(K.shape[1]) / K.shape[1]

    ufinal = np.zeros(len(mu_x))
    vfinal = np.zeros(len(mu_y))

    ind1 = mu_x>0
    ind2 = mu_y>0
    mu_x = mu_x[ind1]
    mu_y = mu_y[ind2]

    K = K[np.ix_(ind1,ind2)]

    if u is None:
        u = np.zeros(len(mu_x))
    if v is None:
        v = np.zeros(len(mu_y))




    listu = []
    listv = []
    for i in range(n_iter):
        M = (-K + u[:, np.newaxis] + v[:, np.newaxis].T) / eps
        u = eps * (np.log(mu_x) - logsumexp(M, axis=1)) + u
        #u =u- u[-1]
        M = (-K + u[:, np.newaxis] + v[:, np.newaxis].T) / eps
        v = eps * (np.log(mu_y) - logsumexp(M.T, axis=1)) + v


        listu.append(u)
        listv.append(v)
        if i>0:
            m1 = np.max(u-listu[-2])
            m2 = np.max(v-listv[-2])
            if(np.maximum(m1,m2)<tolu):
                #print(i)
                break
    uu = np.exp(u / eps)
    vv = np.exp(v / eps)

    val1 = np.matmul(u/eps, mu_x)- np.matmul(np.log(mu_x), mu_x)
    val2 = np.matmul(v/eps, mu_y) - np.matmul(np.log(mu_y), mu_y)

    val = val1 + val2
    pi = np.dot(np.diag(uu), np.dot(np.exp(-K / eps), np.diag(vv)))
    ufinal[ind1] = u
    vfinal[ind2] = v
    return val, pi,ufinal,vfinal,listu[-1],listv[-1]


   


from numpy.linalg import *
from ot.bregman import sinkhorn_knopp
def sinkhorn_em_new(X,M,itr_num=30,seed=1,var0 =None,update = None,tol=1e-8,update_em_pi=False,do_sinkhorn=True,n_iter_sink=100,
                tol_sigma=1e-8,sigma_0=1.0, batch_size = None,sigma_diag=False,tols=1e-5, init_mu_type='subsample', H0 = 0, update_sigma_ind =None):
   
    if(update_sigma_ind is None):
        update_sigma_ind = np.arange(M)
    if(var0 is None):
        pi = None
        sigma = None
        mu = None
    else:
        pi =var0['pi']
        sigma = var0['sigma']
        mu = var0['mu']
   
    #itr_num = 30

    X = np.array(X)
    D, N = X.shape
    if(batch_size is not None):
        N =batch_size
    sink_losses = []
    sigma_all = []
    pi_all = []
    mu_all = []
    props = []
    liks=[]
    # Initialization for the parameters
   
    # We should have a good initialize of mu.
    # By doing so, we can avoid dividing zero when probs of all samples are approximately close to 0 for a bad initialized mixutre
   
    if(pi is None):
        pi = np.array([1]*M)/M  # (M,) array
    np.random.seed(seed)  
    if(sigma is None):
        sigma = np.tile(np.diag(np.array([sigma_0]*D))[np.newaxis,:,:], [M, 1,1])  # 1xD matrix, interpreted as uniqueness, this parameter is SHARED for all mixtures
    if(mu is None):
        if(init_mu_type == 'subsample'):
            #mu = X[:,np.random.randint(X.shape[1], size=M)].T[:,:,np.newaxis]
            N = X.shape[1]
            ind0 = np.random.randint(N, size=1)[0]
            x0=X[:, ind0]
            listind =[ind0]
            xlist=[x0]

            for i in range(M-1):
                rem=[i for i in range(N) if i not in listind]
                d = X[:,rem]-x0[:,np.newaxis]
                d1=np.sum(d**2, axis=0)
                indnew = np.where(np.random.multinomial(1,pvals=d1/sum(d1)))[0][0]
                listind.append(indnew)
                xlist.append(X[:,rem[indnew]])
            mu = np.array(xlist)[:,:,np.newaxis]
            #print(listind)
        if(init_mu_type == 'subsample2'):
            #mu = X[:,np.random.randint(X.shape[1], size=M)].T[:,:,np.newaxis]
            N = X.shape[1]
            ind0 = np.random.choice(N, D+1, replace=False)
            ind1 = [n for n in range(N) if n not in ind0]
            mu = np.zeros((M, D, 1))
            mu[0,:, 0]= np.mean(X[:, ind0], axis=1)
            mu[1,:, 0]= np.mean(X[:, ind1], axis=1)
           
        if(init_mu_type == 'gaussian'):
            mu = np.array(np.random.normal(loc=0, scale=0.1, size=M*D)).reshape([M,D,1])  # MxDx1 matrix, each row represents an mean vector for a mixture
   
        if(init_mu_type == 'H0'):
            mu = np.zeros((M, D, 1))
            sigma_new = np.zeros((M, D, D))
            H = np.zeros((M, len(H0)))
            H[0,:] = H0
            H[1,:] = 1-H0
           
            for j in range(M):
                Hj = H[j,:].reshape((1,N))  # (1,N) matrix

                mu[j,...] = (np.sum(X*Hj,axis=1)/np.sum(Hj)).reshape((-1,1))

                X_mu=X - mu[j,...]

                sigma[j,:,:] = np.matmul(X_mu*Hj,X_mu.T)/np.sum(Hj)+tol_sigma*np.eye(D)

           
    for itr in range(itr_num):
        if(batch_size is None):
            X2 = X
            indsample = range(N)
        else:
            indsample = sort(np.random.choice(X.shape[1], size=batch_size, replace=False))
            X2 = X[:,indsample]
            #X2 = X
        mu_all.append(mu)
        sigma_all.append(sigma)
        pi_all.append(pi)
        # ===== Expectation Stage (Calculating the posterior statistics, based on old parameters)
        # H is of size MxN, where hij is the prob of xj been in mixture j
        Difs = np.zeros((M,N))
        #print(X2.shape)
       
        H0 = np.zeros((M,N))
       
        for j in range(M):
            cov =  sigma[j,:,:]
            muj = np.array(mu[j,...]).reshape((-1,1))  # Dx1

            X_mu = X2 - muj  # DxN
            Difs[j,:]= (np.linalg.slogdet(cov)[1] + np.diag(np.matmul(np.matmul(X_mu.T,inv(cov)),X_mu)) + D*np.log(2*np.pi)) / 2
           
            D = len(muj)
            cov = np.array(cov).reshape(D,D)
            aux= Difs[j,:]-np.log(pi[j])
            H0[j,:] = -aux
       
        lik = np.mean(logsumexp(H0,0))
        H0 = np.exp(H0-logsumexp(H0,0,keepdims=True))
           
        liks.append(lik)
       
        suma=0
       
       
        if(do_sinkhorn):
            #print(pi)
            current_L, pip,u,v,_,_ = sinkhorn_np_logspace(Difs, 1,  mu_x=pi, mu_y=np.ones((N))/N, n_iter = n_iter_sink,tolu=tols)
            #print(np.sum(pip,0))
            #print(np.sum(pip,1))
            H= -Difs+v+u[:,np.newaxis]
             
            tilted = np.exp(u.reshape(-1,1) - logsumexp(u.reshape(-1,1)))
            sink_losses.append(current_L)
            s= logsumexp(-Difs+v+u[:,np.newaxis],axis=0)
            H = np.exp(H-s)
        else:
            H = H0
        Hlik = H0
        pi_new = np.sum(H0,axis=1)/np.sum(H0)  # (M,) array
        props.append(H)
        X_mu = []  # MxDxN matrix
        for j in range(M):
            X_mu.append(X2 - mu[j,...])
        X_mu = np.array(X_mu)

       
       
        mu_new = np.zeros_like(mu)
        sigma_new = np.zeros_like(sigma)
        for j in range(M):
            Hj = H[j,:].reshape((1,N))  # (1,N) matrix
            # NON SYMMETRIC
            #mu_new[j,...] = (np.sum(X2*Hj,axis=1)/np.sum(Hj)).reshape((-1,1))
            # SYMMETRIC
            mu_new[j,...] = (np.sum(X2*(2*Hj-1),axis=1)/N).reshape((-1,1))
            #if(sigma_diag is False):
            #    sigma_new[j,:,:] = np.matmul(X_mu[j,...]*Hj,X_mu[j,...].T)/np.sum(Hj)+tol_sigma*np.eye(D)
           
            #else:
            #    A= np.matmul(X_mu[j,...]*Hj,X_mu[j,...].T)/np.sum(Hj)+tol_sigma*np.eye(D)
            #    var = np.sum(np.diag(A))/D
            #    sigma_new[j,:,:]  =np.eye(D)*var
           
        if(update['pi']):
            if(update_em_pi):
                pi = pi_new  # (M,) array
            else:
                #print([pi.shape, pi_new.shape])
                pi = pi_new#update_pi_sinkhorn(pi, Difs, current_L, u).flatten()
                #print(pi.shape)
        if(update['mu']):
            #print('hola')
            mu = mu_new  # MxDx1 matrix
        if(update['sigma']):
           # if(update_sigma_ind is None)
            sigma[update_sigma_ind,:,:] = sigma_new[update_sigma_ind,:,:]  # 1xD matrix
        dif = mu-mu_all[-1]
        dif2 = sigma-sigma_all[-1]
        #print([np.max(abs(dif)),np.max(abs(dif2))])
        if(np.max(np.abs(dif))+np.max(np.abs(dif2))<tol):
            #print(itr)
            break
            #H = np.exp(H0-logsumexp(H0,0,keepdims=True))
    #H = np.exp(H0-logsumexp(H0,0,keepdims=True))
           
    return pi, mu.reshape(M,D), sigma,mu_all,H,liks,sink_losses, sigma_all,pi_all,Hlik,props

def sinkhorn_em_old(X,M,itr_num=30,seed=1,var0 =None,update = None,tol=1e-8,update_em_pi=False,do_sinkhorn=True,n_iter_sink=100,
                tol_sigma=1e-8,sigma_0=1.0, batch_size = None):
   
   
    if(var0 is None):
        pi = None
        sigma = None
        mu = None
    else:
        pi =var0['pi']
        sigma = var0['sigma']
        mu = var0['mu']
   
    #itr_num = 30
   
    X = np.array(X)
    D, N = X.shape
    if(batch_size is not None):
        N =batch_size
    sink_losses = []
    sigma_all = []
    pi_all = []
    mu_all = []
    liks=[]
    # Initialization for the parameters
   
    # We should have a good initialize of mu.
    # By doing so, we can avoid dividing zero when probs of all samples are approximately close to 0 for a bad initialized mixutre
   
    if(pi is None):
        pi = np.array([1]*M)/M  # (M,) array
    np.random.seed(seed)  
    if(sigma is None):
        sigma = np.tile(np.diag(np.array([sigma_0]*D))[np.newaxis,:,:], [M, 1,1])  # 1xD matrix, interpreted as uniqueness, this parameter is SHARED for all mixtures
    if(mu is None):
        mu = X[:,np.random.randint(X.shape[1], size=M)].T[:,:,np.newaxis]
        #mu = np.array(np.random.normal(loc=0, scale=0.1, size=M*D)).reshape([M,D,1])  # MxDx1 matrix, each row represents an mean vector for a mixture
   
    for itr in range(itr_num):
        if(batch_size is None):
            X2 = X
            indsample = range(N)
        else:
            indsample = sort(np.random.choice(X.shape[1], size=batch_size, replace=False))
            X2 = X[:,indsample]
            #X2 = X
        mu_all.append(mu)
        sigma_all.append(sigma)
        pi_all.append(pi)
        # ===== Expectation Stage (Calculating the posterior statistics, based on old parameters)
        # H is of size MxN, where hij is the prob of xj been in mixture j
        Difs = np.zeros((M,N))
        #print(X2.shape)
       
        H0 = np.zeros((M,N))
       
        for j in range(M):
            cov =  sigma[j,:,:]
            muj = np.array(mu[j,...]).reshape((-1,1))  # Dx1
           
            X_mu = X2 - muj  # DxN
            Difs[j,:]= (np.log(np.linalg.det(cov)) + np.diag(np.matmul(np.matmul(X_mu.T,inv(cov)),X_mu)) + D*np.log(2*np.pi)) / 2
           
            D = len(muj)
            cov = np.array(cov).reshape(D,D)
            aux= Difs[j,:]+np.log(pi[j])
            H0[j,:] = -aux
       
        lik = np.mean(logsumexp(H0,0))
        H0 = np.exp(H0-logsumexp(H0,0,keepdims=True))
           
        liks.append(lik)
       
        suma=0
       
       
        if(do_sinkhorn):
            current_L, _,u,v,_,_ = sinkhorn_np_logspace(Difs, 1,  mu_x=pi, mu_y=np.ones((N))/N, n_iter = n_iter_sink)
       
            H= np.exp(-Difs+v+u[:,np.newaxis])
            #H, _ = sinkhorn_knopp(pi,np.ones((N))/N,  Difs, 1, log=True)
            sink_losses.append(current_L)
            s= np.exp(logsumexp(-Difs+v+u[:,np.newaxis],axis=0))
            H = H/s
        else:
            H = H0
       
        pi_new = np.sum(H0,axis=1)/np.sum(H0)  # (M,) array
        #print(pi_new)
        X_mu = []  # MxDxN matrix
        for j in range(M):
            X_mu.append(X2 - mu[j,...])
        X_mu = np.array(X_mu)

       
       
        mu_new = np.zeros_like(mu)
        sigma_new = np.zeros_like(sigma)
        for j in range(M):
            Hj = H[j,:].reshape((1,N))  # (1,N) matrix
           
            mu_new[j,...] = (np.sum(X2*Hj,axis=1)/np.sum(Hj)).reshape((-1,1))
            sigma_new[j,:,:] = np.matmul(X_mu[j,...]*Hj,X_mu[j,...].T)/np.sum(Hj)+tol_sigma*np.eye(D)
           
        if(update['pi']):
            if(update_em_pi):
                pi = pi_new  # (M,) array
            else:
                #print([pi.shape, pi_new.shape])
                pi = pi_new#update_pi_sinkhorn(pi, Difs, current_L, u).flatten()
                #print(pi.shape)
        if(update['mu']):
            #print('hola')
            mu = mu_new  # MxDx1 matrix
        if(update['sigma']):
            sigma = sigma_new  # 1xD matrix
        dif = mu-mu_all[-1]
        if(np.max(np.abs(dif))<tol):
            #print(itr)
            break
   
    return pi, mu.reshape(M,D), sigma,mu_all,H,liks,sink_losses, sigma_all,pi_all,indsample




def create_K(x_real, y_real):
    N_x = x_real.shape[0]
    N_y = y_real.shape[0]

    normx = np.tile(np.sum(x_real ** 2, 1, keepdims=True),  [1, N_y])
    normy = np.tile(np.sum(y_real **2 , 1, keepdims=True).T, [N_x, 1])
    #print(x_real.shape)
    #print(y_real.T.shape)
    z = np.matmul(x_real, y_real.T)
    return (normx - 2 * z + normy)

