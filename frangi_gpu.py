#https://github.com/isyiming/Frangi-filter-based-Hessian
# Implement the above function in torch-gpu

# It will instead process a video instead of a frame such that resource can be reused 
# The purpose of this is to accelerate
import torch
import math
def Frangi_gpu(I):
    pass

def create_layer(Sigma, device):
    S_round = torch.round(3*Sigma)
    grid_range = torch.arange(-S_round, S_round+1, 1)
    [X,Y] = torch.meshgrid(grid_range,grid_range)

    DGaussxx = 1/(2*math.pi*pow(Sigma,4)) * (X**2/pow(Sigma,2) - 1) * torch.exp(-(X**2 + Y**2)/(2*pow(Sigma,2)))
    DGaussxy = 1/(2*math.pi*pow(Sigma,6)) * (X*Y) * torch.exp(-(X**2 + Y**2)/(2*pow(Sigma,2)))   
    DGaussyy = 1/(2*math.pi*pow(Sigma,4)) * (Y**2/pow(Sigma,2) - 1) * torch.exp(-(X**2 + Y**2)/(2*pow(Sigma,2)))
    Dlayer = torch.nn.Conv2d(1, 3, grid_range.shape[0], padding='same', bias = False, device = device)
    Dlayer.weight.data[0,0,:,:] = DGaussxx
    Dlayer.weight.data[1,0,:,:] = DGaussxy
    Dlayer.weight.data[2,0,:,:] = DGaussyy

    return Dlayer

def Hessian2D(Dlayer, frames, device):
    # Input : frames [B,1,H,W] grey scale image, dlayer conv layers built with sigma
    # Output: The 2d Hessian of the image
    with torch.no_grad():
        frames = frames.to(device)
        out = Dlayer(frames)

    return out[:,None,0,:,:], out[:,None,1,:,:], out[:,None,2,:,:]


def eig2image(Dxx,Dxy,Dyy):
    # This function eig2image calculates the eigen values from the
    # hessian matrix, sorted by abs value. And gives the direction
    # of the ridge (eigenvector smallest eigenvalue) .
    # input:Dxx,Dxy,Dyy图像的二阶导数
    # output:Lambda1,Lambda2,Ix,Iy
    #Compute the eigenvectors of J, v1 and v2


    tmp = torch.sqrt( (Dxx - Dyy)**2 + 4*Dxy**2)

    v2x = 2*Dxy
    v2y = Dyy - Dxx + tmp

    mag = torch.sqrt(v2x**2 + v2y**2)
    i = torch.Tensor(mag!=0)

    v2x[i==True] = v2x[i==True]/mag[i==True]
    v2y[i==True] = v2y[i==True]/mag[i==True]

    v1x = -v2y 
    v1y = v2x

    mu1 = 0.5*(Dxx + Dyy + tmp)
    mu2 = 0.5*(Dxx + Dyy - tmp)

    check=abs(mu1)>abs(mu2)
            
    Lambda1=mu1.clone().detach()
    Lambda1[check==True] = mu2[check==True]
    Lambda2=mu2.clone().detach()
    Lambda2[check==True] = mu1[check==True]
    
    Ix=v1x
    Ix[check==True] = v2x[check==True]
    Iy=v1y
    Iy[check==True] = v2y[check==True]
    
    return Lambda1,Lambda2,Ix,Iy

def FrangiFilter2D(I):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    I = I.type(torch.float32)

    defaultoptions = {'FrangiScaleRange':(1,10), 'FrangiScaleRatio':2, 'FrangiBetaOne':0.5, 'FrangiBetaTwo':15,'BlackWhite':True};  
    options=defaultoptions

    sigmas=torch.arange(options['FrangiScaleRange'][0],options['FrangiScaleRange'][1],options['FrangiScaleRatio'])
    sigmas.sort()#升序

    beta  = torch.tensor(2*pow(options['FrangiBetaOne'],2),device=device)
    c     = torch.tensor(2*pow(options['FrangiBetaTwo'],2),device=device)

    #存储滤波后的图像
    shape=(I.shape[0],len(sigmas),I.shape[2],I.shape[3])
    ALLfiltered=torch.zeros(shape) 

    #Frangi filter for all sigmas 
    Rb=0
    S2=0

    
    Dlayers = [create_layer(sigma, device) for sigma in sigmas]

    for i in range(len(sigmas)):

        #Make 2D hessian
        [Dxx,Dxy,Dyy] = Hessian2D(Dlayers[i],I,device)

        #Correct for scale 
        Dxx = pow(sigmas[i],2)*Dxx  
        Dxy = pow(sigmas[i],2)*Dxy  
        Dyy = pow(sigmas[i],2)*Dyy
         
        #Calculate (abs sorted) eigenvalues and vectors  
        [Lambda2,Lambda1,Ix,Iy]=eig2image(Dxx,Dxy,Dyy)  


        #Compute some similarity measures  
        Lambda1[Lambda1==0] = torch.finfo(torch.float64).eps

        Rb = (Lambda2/Lambda1)**2  
        S2 = Lambda1**2 + Lambda2**2
        
        #Compute the output image
        Ifiltered = torch.exp(-Rb/ beta) * (torch.ones(I.shape, device=device)-torch.exp(-S2/c))

        #see pp. 45  
        if(options['BlackWhite']): 
            Ifiltered[Lambda1<0]=0
        else:
            Ifiltered[Lambda1>0]=0
        
        #store the results in 3D matrices  
        ALLfiltered[:,i,:,:] = Ifiltered[:,0,:,:]
        #ALLangles[:,:,i] = angles

        # Return for every pixel the value of the scale(sigma) with the maximum   
        # output pixel value  
    if len(sigmas) > 1:
        maxout , _ = torch.max(ALLfiltered, dim = 1)
    maxout = maxout[:,None,:,:]
    return maxout