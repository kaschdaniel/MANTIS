import numpy as np


def detection(E):
    '''
    

    Parameters
    ----------
    E : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    return np.abs(E)**2


def determine_winner(E, det1=0, det2=-1):
    '''
    

    Parameters
    ----------
    E : TYPE
        DESCRIPTION.
    det1 : TYPE, optional
        DESCRIPTION. The default is 0.
    det2 : TYPE, optional
        DESCRIPTION. The default is -1.

    Returns
    -------
    int
        DESCRIPTION.

    '''
    if E[det1] > E[det2]:
        return 0
    else:
        return 1
    
def loss_function(E, y, det1, det7):
    """
    Accepts batch of evaluations at once
    """
    dim = E.ndim
    if dim==3:#Full history E-given
        Efinal = E[-1,:,:] #form: [channel, batch]
    elif dim==2:
        Efinal = E
    else:
        raise ValueError("Dimension of E faulty, ndim=2 or ndim=3 expected!")
    I = np.abs(Efinal)**2 #shape (channel, batch)
    scores = I[[det1, det7], :]
    target = np.zeros_like(scores)   # (2, batch)
    target[0, y == 1] = 1
    target[1, y == 7] = 1
    print(target[0])
    print("ululi")
    #calc difference to target1
    print(I[det1,:])
    diff1=np.diff(target[0], I[det1,:])
    print(diff1)
    #Stimmt noch nicht, mach jetzt aber ne Pause
    return 