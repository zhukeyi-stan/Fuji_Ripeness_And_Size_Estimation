import numpy as np
import scipy
import random
import cv2

import scipy.spatial
from .pcloud_operation import get_mean_depth,outlier_removal,gen_pc

r_min=20
r_max=50
# r_min=0
# r_max=100

def largest_segment(points, r_min=r_min, r_max=r_max):
    # n=points.shape[0]
    # dist_mat=1e8*np.ones((n,n))
    # for i in range(n):
    #     for j in range(i,n):
    #         if i==j:
    #             pass
    #         dist=np.linalg.norm(points[i,:]-points[j,:])
    #         dist_mat[i,j]=dist
    #         dist_mat[j,i]=dist
    # return np.max(dist_mat)
    try:
        points=points[scipy.spatial.ConvexHull(points).vertices]
        dist_mat=scipy.spatial.distance_matrix(points,points)
    except:
        dist_mat=scipy.spatial.distance_matrix(points,points)
    # return dist_mat.max()
    return np.clip(dist_mat.max()*1000,2*r_min,2*r_max)

def bounded_sphere_ransac_fit(pts, thresh=0.01, maxIteration=1000, r_min=r_min, r_max=r_max):
    """
    Find the parameters (center and radius) to define a Sphere.

    :param pts: 3D point cloud as a numpy array (N,3).
    :param thresh: Threshold distance from the Sphere hull which is considered inlier.
    :param maxIteration: Number of maximum iteration which RANSAC will loop over.

    :returns:
    - `center`: Center of the cylinder np.array(1,3) which the cylinder axis is passing through.
    - `radius`: Radius of cylinder.
    - `inliers`: Inlier's index from the original point cloud.
    ---
    """

    n_points = pts.shape[0]
    best_inliers = []
    best_center=[]
    best_radius=-1
    for it in range(maxIteration):

        # Samples 4 random points
        id_samples = random.sample(range(0, n_points), 4)
        pt_samples = pts[id_samples]

        # We calculate the 4x4 determinant by dividing the problem in determinants of 3x3 matrix

        # Multiplied by (x²+y²+z²)
        d_matrix = np.ones((4, 4))
        for i in range(4):
            d_matrix[i, 0] = pt_samples[i, 0]
            d_matrix[i, 1] = pt_samples[i, 1]
            d_matrix[i, 2] = pt_samples[i, 2]
        M11 = np.linalg.det(d_matrix)
        if M11==0:
            continue
        # Multiplied by x
        for i in range(4):
            d_matrix[i, 0] = np.dot(pt_samples[i], pt_samples[i])
            d_matrix[i, 1] = pt_samples[i, 1]
            d_matrix[i, 2] = pt_samples[i, 2]
        M12 = np.linalg.det(d_matrix)

        # Multiplied by y
        for i in range(4):
            d_matrix[i, 0] = np.dot(pt_samples[i], pt_samples[i])
            d_matrix[i, 1] = pt_samples[i, 0]
            d_matrix[i, 2] = pt_samples[i, 2]
        M13 = np.linalg.det(d_matrix)

        # Multiplied by z
        for i in range(4):
            d_matrix[i, 0] = np.dot(pt_samples[i], pt_samples[i])
            d_matrix[i, 1] = pt_samples[i, 0]
            d_matrix[i, 2] = pt_samples[i, 1]
        M14 = np.linalg.det(d_matrix)

        # Multiplied by 1
        for i in range(4):
            d_matrix[i, 0] = np.dot(pt_samples[i], pt_samples[i])
            d_matrix[i, 1] = pt_samples[i, 0]
            d_matrix[i, 2] = pt_samples[i, 1]
            d_matrix[i, 3] = pt_samples[i, 2]
        M15 = np.linalg.det(d_matrix)

        # Now we calculate the center and radius
        center = [0.5 * (M12 / M11), -0.5 * (M13 / M11), 0.5 * (M14 / M11)]
        radius = np.sqrt(np.dot(center, center) - (M15 / M11))
        radius = np.clip(radius,r_min/1000,r_max/1000)

        # Distance from a point
        pt_id_inliers = []  # list of inliers ids
        dist_pt = center - pts
        dist_pt = np.linalg.norm(dist_pt, axis=1)

        # Select indexes where distance is biggers than the threshold
        pt_id_inliers = np.where(np.abs(dist_pt - radius) <= thresh)[0]

        if len(pt_id_inliers) > len(best_inliers):
            best_inliers = pt_id_inliers
            best_center = center
            best_radius = radius
    return best_center, best_radius, best_inliers

def least_square_fit(spX,spY,spZ, 
                     loss='arctan', 
                     init_values=None, 
                     ftol=10e-7,
                     xtol=10e-7,
                     bounds = None,
                     jac='3-point',
                     r_min=r_min, r_max=r_max
                     ):

    if init_values is None:
        # raise ValueError("Initial values must be provided")
        init_values=[0.04,np.mean(spX),np.mean(spY),np.mean(spZ)]
    if not bounds:
        # raise ValueError("Bounds must be provided")
        bounds=([r_min/1000,init_values[1]-0.4,init_values[2]-0.4,init_values[3]-0.4],
                [r_max/1000,init_values[1]+0.4,init_values[2]+0.4,init_values[3]+0.4])
    
    def sublinear_exp_loss(z):
        '''
        must take a 1-D ndarray z=f**2 and return an array_like with shape (3, m) 
        where row 0 contains function values, row 1 contains first derivatives and row 2 
        contains second derivatives. '''
        return np.array([1-np.exp(-z), np.exp(-z), -np.exp(-z)])

    def tanh_loss(z):
        '''
        must take a 1-D ndarray z=f**2 and return an array_like with shape (3, m) 
        where row 0 contains function values, row 1 contains first derivatives and row 2 
        contains second derivatives. '''
        z = np.clip(z, -50, 50)
        return np.array([np.tanh(z), 1/np.cosh(z)**2, -2*np.tanh(z)/np.cosh(z)**2])

    def sublinear_loss(z):
        '''
        must take a 1-D ndarray z=f**2 and return an array_like with shape (3, m) 
        where row 0 contains function values, row 1 contains first derivatives and row 2 
        contains second derivatives. '''
        return np.array([1-1/(1+z), 1/(1+z)**2, -2/(1+z)**3])

    def sphere_function(x, *args, **kwargs):
        '''
        callable for scipy.optimize.leastsq
        x is the parameter vector, x = [r, x0, y0, z0]
        args are the data to be fit
        returns 1d array of differences between model and data
        '''
        r, x0, y0, z0 = x
        X, Y, Z = args
        return np.sqrt((X-x0)**2 + (Y-y0)**2 + (Z-z0)**2) - r
        

    result = scipy.optimize.least_squares(sphere_function, init_values,
                                            loss=loss,
                                            ftol=ftol,
                                            xtol=xtol,
                                            bounds=bounds,
                                            max_nfev=250,
                                            args=(spX, spY, spZ), 
                                            jac=jac)
    
    result.x[0]=np.clip(result.x[0],r_min/1000,r_max/1000)
    return result

def max_dist_mask(depth,mask,focal_length,removal_rate,r_min=r_min,r_max=r_max):
    points=[]
    depths=[]
    y,x=np.where(mask)

    for i in range(x.shape[0]):
        points.append([y[i],x[i]])
        if depth[y[i],x[i]]<0.5:
            continue
        depths.append(depth[y[i],x[i]])
    if len(points)<50 or len(depths)<50:
        return -1
    else:
        max_dist=0
        average_depth=get_mean_depth(depths,removal_rate)
        if average_depth<0.5:
            return -1
        points=np.array(points)
        try:
            points=points[scipy.spatial.ConvexHull(points).vertices]
            dist_mat=scipy.spatial.distance_matrix(points,points)
        except:
            dist_mat=scipy.spatial.distance_matrix(points,points)
        max_dist=dist_mat.max()
        # for i in range(len(points)):
        #     for j in range(i+1,len(points)):
        #         x1,y1=points[i]
        #         x2,y2=points[j]
        #         dist=np.sqrt((x1-x2)**2+(y1-y2)**2)
        #         if dist>max_dist:
        #             max_dist=dist
        d=max_dist/focal_length*average_depth*1000
        d=np.clip(d,2*r_min,2*r_max)
        return d
    
def hough_mask(depth,mask,focal_length,removal_rate,r_min=r_min,r_max=r_max):
    points=[]
    depths=[]
    y,x=np.where(mask)

    # img=mask.reshape(mask.shape[0],mask.shape[1],1)
    img=mask.astype(np.uint8)*255
    # img=cv2.cvtColor(img.astype(int)*255, cv2.COLOR_BGR2GRAY)

    for i in range(x.shape[0]):
        points.append([y[i],x[i]])
        if depth[y[i],x[i]]<0.5:
            continue
        depths.append(depth[y[i],x[i]])
    if len(points)<50 or len(depths)<50:
        return -1
    else:
        circles=cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=8, minDist=50, param1=100, param2=80, minRadius=10, maxRadius=200)
        if circles is not None:
            average_depth=get_mean_depth(depths,removal_rate)
            d=circles[0,0,2]*2/focal_length*average_depth*1000
            d=np.clip(d,2*r_min,2*r_max)
            return d
        else:
            return -1

def bbox_estimation(depth,mask,bbox,focal_length,removal_rate,r_min=r_min,r_max=r_max):
    points=[]
    depths=[]
    x1,y1,x2,y2=bbox
    d_=np.max([x2-x1,y2-y1])

    y,x=np.where(mask)
    for i in range(x.shape[0]):
        points.append([y[i],x[i]])
        if depth[y[i],x[i]]<0.5:
            continue
        depths.append(depth[y[i],x[i]])
    if len(points)<50 or len(depths)<50:
        return -1
    average_depth=get_mean_depth(depths,removal_rate)
    d=d_/focal_length*average_depth*1000
    d=np.clip(d,2*r_min,2*r_max)
    return d

def size_estimation(algorithm,args):
    if algorithm in ['largest_segment','least_square','bounded_ransac']:
        ## 3D
        img,depth,bbox,mask,focal_length,removal_rate=args
        pc=gen_pc(img,depth,bbox,mask,focal_length)
        pc=pc[:,:3]
        pc_filtered=outlier_removal(pc,removal_rate[0],removal_rate[1])
        if algorithm=='largest_segment':
            d=largest_segment(pc_filtered)
        elif algorithm=='least_square':
            res=least_square_fit(pc_filtered[:,0],pc_filtered[:,1],pc_filtered[:,2])
            d=res.x[0]*1000*2
        elif algorithm=='bounded_ransac':
            _,r,_=bounded_sphere_ransac_fit(pc_filtered)
            d=r*1000*2
    elif algorithm in ['bbox','max_dist_mask','hough_mask']:
        ## 2D
        depth,mask,bbox,focal_length,removal_rate=args
        if algorithm=='bbox':
            d=bbox_estimation(depth,mask,bbox,focal_length,removal_rate)
        elif algorithm=='max_dist_mask':
            d=max_dist_mask(depth,mask,focal_length,removal_rate)
        elif algorithm=='hough_mask':
            d=hough_mask(depth,mask,focal_length,removal_rate)
        pass
    else:
        raise Exception("Sizing Algorithm not valid.")
    return d
