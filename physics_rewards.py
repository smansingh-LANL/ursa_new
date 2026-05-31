import torch 
def reward_massconservation(x,y):
    """ 
    Compute = integrate \rho for both samples. Volume is constant, so we don't 
    treate it explicitly.
    """
    # input fields
    
    rhoV_input = x[0,::]
    
    # output fields
    rhoV_output = y[0,::]
    #normalizer
    normalizer = torch.abs(torch.sum(rhoV_input))
    #mass discrepancy
    flux = torch.abs(torch.sum(rhoV_output - rhoV_input))
    reward = torch.mean(flux / normalizer)
    return -reward
def reward_momentumconservation(x,y):
    """ Compute = \rho * A * v for both samples."""
    # input fields
    rhoV_input = x[0,::]*x[1:2,::] #density * velocity
    # output fields
    rhoV_output = y[0,::]*y[1:2,::]
    #normalizer
    normalizer = torch.abs(torch.sum(rhoV_input))
    #mass discrepancy
    flux = torch.abs(torch.sum(rhoV_output - rhoV_input))
    reward = torch.mean(flux / normalizer)
    return -reward
def reward_energyconservation(y,x):
    """ 
    Compute = integrate \rho for both samples. Volume is constant, so we don't 
    treate it explicitly.
    """
    # input fields
    rhoV_input = x[4,::]
    # output fields
    rhoV_output = y[4,::]
    #normalizer
    normalizer = torch.abs(torch.sum(rhoV_input))
    #mass discrepancy
    flux = torch.abs(torch.sum(rhoV_output - rhoV_input))
    reward = torch.mean(flux / normalizer)
    return -reward
