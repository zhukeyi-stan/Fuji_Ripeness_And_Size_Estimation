import numpy as np

def gen_pc(img,depth,bbox,mask,focal_length):
    def transform(px, py, z, f):
        x = px / f * z
        y = py / f * z
        return x, y
    
    pc=[]
    h,w,_=img.shape
    x1,y1,x2,y2=bbox
    x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
    for x_ in range(x1,x2):
        for y_ in range(y1,y2):
            if mask[y_,x_]:
                z=depth[y_,x_]
                if z==0:
                    continue
                x,y=transform(x_-w/2,y_-h/2,z,focal_length)
                pc.append([x,y,z,img[y_,x_,2],img[y_,x_,1],img[y_,x_,0]])
    # if len(pc)==0:
    #     # No point is in the point cloud
    #     pass
    pc=np.array(pc)
    
    return pc

def outlier_removal(pc,low_bound,high_bound):
    z_low=0.2
    z_high=6
    if low_bound==0 and high_bound==100:
        return pc
    pc=pc[pc[:,2]<z_high,:]
    pc=pc[z_low<pc[:,2],:]
    ind=np.argsort(pc[:,2])
    sorted_pc=pc[ind,:]
    n=sorted_pc.shape[0]
    ind_begin=int(np.floor(n*low_bound/100.))
    ind_end=int(np.ceil(n*high_bound/100.))
    cut_pc=sorted_pc[ind_begin:ind_end,:]
    # return filter(cut_pc)
    return cut_pc

def get_mean_depth(depths,removal_rate):
    depths=np.array(depths)
    depths=depths[depths>0]
    residues=np.logical_and(depths>=np.percentile(depths,removal_rate[0]),depths<=np.percentile(depths,removal_rate[1]))
    depths_modified=depths[residues]
    if np.isnan(np.mean(depths_modified)):
        pass
    return np.mean(depths_modified)

# def filter(pc):
#     import open3d as o3d
#     pcd=o3d.geometry.PointCloud()
#     pcd.points=o3d.utility.Vector3dVector(pc)
#     # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
#     cl, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.05)
#     pcd_new = pcd.select_by_index(ind)
#     return np.asarray(pcd_new.points)