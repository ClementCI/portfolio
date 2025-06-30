import numpy as np
from scipy.signal import convolve2d, correlate2d
import matplotlib.pyplot as plt

from Functions import *
from gaussfft import gaussfft

        
def pad_to_5x5(mask):
    padded_mask = np.zeros((5, 5))
    padded_mask[1:4, 1:4] = mask  # place the 3x3 mask in the center of the 5x5 array
    return padded_mask

def deltax():
    dxmask = np.array([[0, 0, 0],
                   [0.5, 0, -0.5],
                   [0, 0, 0]])
    return pad_to_5x5(dxmask)

def deltay():
    dymask = np.array([[0, 0.5, 0],
                   [0, 0, 0],
                   [0, -0.5, 0]])
    return pad_to_5x5(dymask)

def deltaxx():
    dxxmask = np.array([[0, 0, 0],
                   [1, -2, 1],
                   [0, 0, 0]])
    return pad_to_5x5(dxxmask)

def deltayy():
    dyymask = np.array([[0, 1, 0],
                   [0, -2, 0],
                   [0, 1, 0]])
    return pad_to_5x5(dyymask)

def deltaxy():
    dxymask = convolve2d(deltax(), deltay(), 'same')
    return dxymask

def deltaxxx():
    dxxxmask =convolve2d(deltax(), deltaxx(), 'same')
    return dxxxmask

def deltayyy():
    dyyymask = convolve2d(deltay(), deltayy(), 'same')
    return dyyymask

def deltaxxy():
    dxxymask = convolve2d(deltaxx(), deltay(), 'same')
    return dxxymask

def deltaxyy():
    dxyymask = convolve2d(deltax(), deltayy(), 'same')
    return dxyymask

def Lv(inpic, shape = 'same'):
        Lx = convolve2d(inpic, deltax(), shape)
        Ly = convolve2d(inpic, deltay(), shape)
        return np.sqrt(Lx**2 + Ly**2)

def Lvvtilde(inpic, shape = 'same'):
        Lx = convolve2d(inpic, deltax(), shape)
        Ly = convolve2d(inpic, deltay(), shape)
        Lxx = convolve2d(inpic, deltaxx(), shape)
        Lyy = convolve2d(inpic, deltayy(), shape)
        Lxy = convolve2d(inpic, deltaxy(), shape)
        
        result = Lx**2*Lxx + 2*Lx*Ly*Lxy + Ly**2*Lyy
        
        return result

def Lvvvtilde(inpic, shape = 'same'):
        Lx = convolve2d(inpic, deltax(), shape)
        Ly = convolve2d(inpic, deltay(), shape)
        Lxxx = convolve2d(inpic, deltaxxx(), shape)
        Lyyy = convolve2d(inpic, deltayyy(), shape)
        Lxxy = convolve2d(inpic, deltaxxy(), shape)
        Lxyy = convolve2d(inpic, deltaxyy(), shape)
        
        result = Lx**2 * Lxxx + 3* Lx**2 *Ly *Lxxy + 3*Lx* Ly**2 *Lxyy + Ly**3 *Lyyy
      
        return result

def extractedge(inpic, scale, shape, threshold=None):
        
        pic = discgaussfft(inpic, scale)
        
        zeropic = Lvvtilde(pic, shape)
        
        maskpic1 = Lvvvtilde(pic, shape)<0
        
        curves = zerocrosscurves(zeropic, maskpic1)
        
        if threshold is not None:
            maskpic2 = Lv(pic, shape) > threshold
            curves = thresholdcurves(curves, maskpic2)
        
        return curves
        
def houghline(pic, curves, magnitude, nrho, ntheta, nlines=20, verbose=False):
        # Allocate space for the accumulator
        acc = np.zeros((nrho, ntheta))
        
        # Define a coordinate system in the accumulator space
        H, W = magnitude.shape
        
        theta_max = np.pi / 2
        theta_range = np.linspace(-theta_max, theta_max, ntheta)
        
        rho_max = np.sqrt((W)**2 + (H)**2)
        rho_range = np.linspace(-rho_max, rho_max, nrho)
        
        dtheta = theta_range[1] - theta_range[0]
        drho = rho_range[1] - rho_range[0]
        
        # Image center
        x_c = W / 2
        y_c = H / 2
        
        # For each edge point
        x_coords = curves[1]
        y_coords = curves[0]
        for i in range(len(x_coords)):
            x_centered, y_centered = x_coords[i] - x_c, y_coords[i] - y_c
            # For each theta
            for theta in theta_range:
                
                # Compute rho
                rho = x_centered * np.cos(theta) + y_centered * np.sin(theta)
                
                # Compute indexes
                rho_idx = int((rho + rho_max) / drho)
                theta_idx = int((theta + theta_max) / dtheta)
            
                # Update accumulator
                if 0 <= rho_idx < nrho and 0 <= theta_idx < ntheta:
                    acc[rho_idx, theta_idx] += 1
       
        # Find optima
        acc = discgaussfft(acc, 1)
        pos, value, _ = locmax8(acc)
    
        # Sort and keep the nlines strongest lines
        indexvector = np.argsort(value)[-nlines:]
        pos = pos[indexvector]
        
        # Extract params for the strongest lines
        linepar = []
        for idx in range(nlines):
            thetaidxacc = pos[idx, 0]
            rhoidxacc = pos[idx, 1]
            
            # Convert back
            theta_val = theta_range[thetaidxacc]
            rho_val = rho_range[rhoidxacc]
            linepar.append((rho_val, theta_val))
        
        if verbose == 1:  # Plot accumulator space
            plt.imshow(acc, extent=[theta_range[0], theta_range[-1], rho_range[-1], rho_range[0]], aspect='auto', cmap='hot')
            plt.title("Espace d'accumulateur de la transformation de Hough")
            plt.xlabel("Theta (radians)")
            plt.ylabel("Rho (pixels)")
            plt.colorbar(label="Votes")
            plt.show()
            
        elif verbose == 2:  # Overlay lines on the original image
            showgrey(pic, display=False)
            for rho, theta in linepar:
                x0 = rho * np.cos(theta)
                y0 = rho * np.sin(theta)
                dx = -1000 * np.sin(theta)
                dy = 1000 * np.cos(theta)
                
                # Come back to the right center
                x0 += x_c
                y0 += y_c
                
                plt.plot([x0 - dx, x0 + dx], [y0 - dy, y0 + dy], 'r-', linewidth=2)
         
            plt.xlim(0, W)
            plt.ylim(H, 0)
           
        # Return the output data [linepar, acc]
        return linepar, acc
    
    
    
    
    
def houghline2(pic, curves, magnitude, nrho, ntheta, Lx, Ly, nlines = 20, verbose = False):
        # Allocate space for the accumulator
        acc = np.zeros((nrho, ntheta))
        
        # Define a coordinate system in the accumulator space
        H, W = magnitude.shape
        
        theta_max = np.pi / 2
        theta_range = np.linspace(-theta_max, theta_max, ntheta)
        
        rho_max = np.sqrt((W)**2 + (H)**2)
        rho_range = np.linspace(-rho_max, rho_max, nrho)
        
        dtheta = theta_range[1] - theta_range[0]
        drho = rho_range[1] - rho_range[0]
        
        # Image center
        x_c = W / 2
        y_c = H / 2
        
        # For each edge point
        x_coords = curves[1]
        y_coords = curves[0]
        for i in range(len(x_coords)):
            x, y = x_coords[i], y_coords[i]
            x_centered, y_centered = x - x_c, y - y_c
            # For each theta
            for theta in theta_range:
                
                # Compute rho
                rho = x_centered * np.cos(theta) + y_centered * np.sin(theta)
                
                # Compute indexes
                rho_idx = int((rho + rho_max) / drho)
                theta_idx = int((theta + theta_max) / dtheta)
            
                # Compute gradient direction
                phi = np.arctan2(Ly[y,x], Lx[y,x])
            
                # Update accumulator
                if 0 <= rho_idx < nrho and 0 <= theta_idx < ntheta:
                    alignment = np.abs(phi - theta)
                    alignment = min(alignment, np.pi - alignment)
                    vote = np.log(1+magnitude[y, x]) * np.cos(alignment)
                    acc[rho_idx, theta_idx] += vote
       
        # Find optima
        acc = discgaussfft(acc, 1)
        pos, value, _ = locmax8(acc)
    
        # Sort and keep the nlines strongest lines
        indexvector = np.argsort(value)[-nlines:]
        pos = pos[indexvector]
        
        # Extract params for the strongest lines
        linepar = []
        for idx in range(nlines):
            thetaidxacc = pos[idx, 0]
            rhoidxacc = pos[idx, 1]
            
            # Convert back
            theta_val = theta_range[thetaidxacc]
            rho_val = rho_range[rhoidxacc]
            linepar.append((rho_val, theta_val))
        
        if verbose == 1:  # Plot accumulator space
            plt.imshow(acc, extent=[theta_range[0], theta_range[-1], rho_range[-1], rho_range[0]], aspect='auto', cmap='hot')
            plt.title("Espace d'accumulateur de la transformation de Hough")
            plt.xlabel("Theta (radians)")
            plt.ylabel("Rho (pixels)")
            plt.colorbar(label="Votes")
            plt.show()
            
        elif verbose == 2:  # Overlay lines on the original image
            showgrey(pic, display=False)
            for rho, theta in linepar:
                x0 = rho * np.cos(theta)
                y0 = rho * np.sin(theta)
                dx = -1000 * np.sin(theta)
                dy = 1000 * np.cos(theta)
                
                # Come back to the right center
                x0 += x_c
                y0 += y_c
                
                plt.plot([x0 - dx, x0 + dx], [y0 - dy, y0 + dy], 'r-', linewidth=2)
         
            plt.xlim(0, W)
            plt.ylim(H, 0)
           
        # Return the output data [linepar, acc]
        return linepar, acc
    

def houghedgeline(pic, scale, gradmagnthreshold, nrho, ntheta, nlines = 20, verbose = False):
        # Extract edge points    
        curves = extractedge(pic, scale, 'same', gradmagnthreshold)
        
        # Extract magnitude
        magnitude = Lv(pic, 'same')
        
        # Plot lines
        linepar, acc =  houghline(pic, curves, magnitude, nrho, ntheta, nlines, verbose)
        
        return None

def houghedgeline2(pic, scale, gradmagnthreshold, nrho, ntheta, nlines = 20, verbose = False):
        # Extract edge points
        curves = extractedge(pic, scale, 'same', gradmagnthreshold)
        
        # Extract magnitude
        magnitude = Lv(pic, 'same')
        
        # For the gradient direction
        Lx = convolve2d(pic, deltax(), 'same')
        Ly = convolve2d(pic, deltay(), 'same')
        
        # Plot lines
        linepar, acc =  houghline2(pic, curves, magnitude, nrho, ntheta, Lx, Ly, nlines, verbose)
        
        return None
         
