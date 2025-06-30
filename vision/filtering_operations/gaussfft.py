import numpy as np

from numpy.fft import fft2, ifft2, fftshift

def gaussfft(pic, t):
    # creating a grid of coordinates
    N,M = pic.shape
    
    x = np.linspace(-N//2, N//2 - 1, N)
    y = np.linspace(-M//2, M//2 - 1, M)
    X, Y = np.meshgrid(x, y)
    
    
    # generating a filter based on a sampled version of the Gaussian function
    gauss = (np.exp(-(X**2 + Y**2) / (2. * t)))/2 * np.pi * t
    
    # Fourier transform the original image and the Gaussian filter
    pic_hat = fft2(pic)
    gauss_hat = fft2(gauss)
    
    # multiplying the Fourier transforms
    result_hat = pic_hat * gauss_hat
    
    # inverting the resulting Fourier transform
    result = np.real(fftshift(ifft2(result_hat)))
    
    return result
