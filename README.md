# An Alternative to the Log-Likelihood with Entropic Optimal Transport
Entropic Optimal Transport Estimator vs Maximum Likelihood Estimator (EOTE vs MLE)


A study of a novel estimator inspired by tools from optimal transport. The latter theory essentially offers a geometry on the space of measures, 
and the estimator satisfies a minimum distance type equation.

The properties of this novel estimator are discussed, and especially its robustness is analyzed against the reference maximum likelihood estimator. It is believed it 
should perform better in the context of model misspecification, and it is proved in the simple setting of a simple symmetric Gaussian Mixture Model.

It seems however that the algorithms for the simulations (EM and Sinkhorn EM counterpart) behave too similarly, in this context, 
to empirically observe consequential differences.
