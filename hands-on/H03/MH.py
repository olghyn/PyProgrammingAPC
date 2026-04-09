import random

def symmetric_probability (mu, std) :
    if hasattr(mu, '__len__') :
        return [m + std * random.gauss(0.0,1.0) for m in mu]
    try :
        return mu + std * random.gauss(0.0,1.0)
    except : 
        raise

class MHsampler () :
    """Basic Metropolis–Hastings (MH) sampler with independent walkers.

    This class implements a simple Metropolis–Hastings Markov Chain Monte Carlo
    (MCMC) sampler using a symmetric proposal distribution. Each walker evolves
    independently according to the same proposal scale (`step`) and target
    log-probability function.

    Parameters
    ----------
    nwalkers : int
        Number of independent Markov chains (walkers).
    logprob : callable
        Function computing the log-probability of a parameter vector.
        Must have signature:
            logprob(x, *lp_args, **lp_kwargs)
        where `x` is an array-like object of length `ndim`.
    ndim : int
        Dimensionality of the parameter space.
    lp_args : tuple, optional
        Positional arguments passed to `logprob`.
    lp_kwargs : dict, optional
        Keyword arguments passed to `logprob`.

    Attributes
    ----------
    step : float
        Proposal scale for the symmetric proposal distribution.
    samples : list of list
        Stored samples for each walker.
    loglike : list of list
        Log-probability values corresponding to stored samples.
    accepted : list of list
        Boolean flags indicating whether each proposal was accepted.
    """
    
    step = 0.05

    def __init__ ( 
        self, nwalkers, logprob, ndim,
        lp_args = tuple(), lp_kwargs = dict()  
    ) :
        """Initialize the Metropolis–Hastings sampler.
        """
        if isinstance( nwalkers, int ) :
            self.nwalkers = nwalkers
        else :
            raise AttributeError( 'nwalkers should be an integer' )
        if isinstance( ndim, int ) :
            self.ndim = ndim
        else :
            raise AttributeError( 'ndim should be an integer' )
        if hasattr( logprob, '__call__' ) :
            self.lprob = logprob
        else :
            raise AttributeError( 'logprob should be a function' )
            
        self.lp_args = lp_args
        self.lp_kwargs = lp_kwargs
        self.samples = [[] for _ in range(nwalkers)]
        self.loglike = [[] for _ in range(nwalkers)]
        self.accepted = [[] for _ in range(nwalkers)]
        
    def _newsample ( self, xi, pi ) :
        """Generate a new Metropolis–Hastings proposal.

        Parameters
        ----------
        xi : array-like
            Current position of the walker.
        pi : float
            Log-probability evaluated at `xi`.

        Returns
        -------
        xnew : array-like
            New state (either the accepted proposal or the previous state).
        pnew : float
            Log-probability at the returned state.
        accepted : bool
            Whether the proposal was accepted.

        Notes
        -----
        The proposal distribution is assumed to be symmetric. The acceptance
        probability is computed as:

            min(1, exp(lnprob_new - lnprob_current))
        """
        from math import exp
        
        # draw new proposal
        xtrial = symmetric_probability( xi, self.step )
        
        # compute log probability
        lnprob = self.lprob( xtrial, *self.lp_args, **self.lp_kwargs )
        
        # compute running probability
        k = exp( min( [ 0.0, lnprob-pi ] ) )
        
        # compute acceptance
        accept = random.choices( [ False, True ], weights = ( 1-k, k ) )[0]
        return (
            xtrial if accept else xi,
            lnprob if accept else pi,
            accept
        )

    def run ( self, nsteps, pstart=None ) :
        """Run the sampler for a fixed number of steps.

        Parameters
        ----------
        nsteps : int
            Number of Metropolis–Hastings iterations to perform per walker.
        pstart : sequence of array-like, optional
            Initial positions for each walker. Must have length `nwalkers`,
            and each element must have length `ndim`.

        Raises
        ------
        AttributeError
            If `nsteps` is not a positive integer or if `pstart` has
            incompatible dimensions.
        RuntimeError
            If no starting point is provided and no previous samples exist.

        Notes
        -----
        Each walker evolves independently. Samples, log-probabilities,
        and acceptance flags are appended to the internal storage.
        """
        
        # here only checking the inputs
        if not isinstance( nsteps, int ) :
            raise AttributeError( 'nsteps should be an integer' )
        if nsteps < 1 :
            raise AttributeError( 'nsteps cannot be smaller than 1')
        if pstart is None :
            for s in self.samples :
                if len(s) < 1 : raise RuntimeError( 
                    'no previous samples to start with, select a value for pstart attribute'
                )
        else :
            # checking pstart has the right dimensions
            if len(pstart) != self.nwalkers : raise AttributeError(
                'please select a starting point for each walker'
            )
            else :
                for i, p in enumerate(pstart) :
                    if len(p) != self.ndim : raise AttributeError(
                        f'walker {i:d} got the wrong number of coordinates'
                    )
            # setting pstart as first point in the chain:
            for i, p in enumerate(pstart) : 
                self.samples[i] += [p]
                self.loglike[i] += [self.lprob(p, *self.lp_args, **self.lp_kwargs)]
                self.accepted[i] += [True]
            
        # from here the function should work whether you pass pstart or not
        for i in range( self.nwalkers ) :
            for _ in range( nsteps ) :
                
                # new proposal
                pnew, lnew, anew = self._newsample( self.samples[i][-1], self.loglike[i][-1] )
                
                # update instance attributes
                self.samples[i] += [pnew]
                self.loglike[i] += [lnew]
                self.accepted[i] += [anew]
                
            print(f'done for chain {i:d}')
        print( 'done for all chains' )

    def acceptance_fraction ( self ) :
        """Compute the overall acceptance fraction.

        Returns
        -------
        float
            Ratio of accepted proposals to total proposals across
            all walkers and steps.
        """
        # inline implementation, just for fun
        return sum([ sum(accept) for accept in self.accepted ]) / sum([ len(accept) for accept in self.accepted ])

    def get_flat_chain ( self, burnin=0, step=1 ) :
        """Return a flattened view of the chains.

        Parameters
        ----------
        burnin : int, optional
            Number of initial samples to discard from each chain.
        step : int, optional
            Thinning factor. Only every `step`-th sample is retained.

        Returns
        -------
        list
            Flattened list of samples from all walkers after burn-in
            and thinning.
        """
        flat_chain = []
        for chain in self.samples :
            flat_chain += chain[burnin::step]
        return flat_chain
