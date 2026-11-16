"""Dedicated straight-sided DFG cylinder grid with wake and circular-ring points.

Delaunay triangulation is accepted only when its exterior exactly matches the
specified rectangle and polygonal hole. This is not a general CAD mesher.
"""
import numpy as np
from scipy.spatial import Delaunay
from physgraph.core.types import Mesh2D
from physgraph.mesh.quality import mesh_quality


def cylinder_flow_mesh(case, level):
    if type(level) is not int or level < 2:
        raise ValueError('cylinder mesh level must be an integer >=2')
    h = .16/level
    # Dense wake to x=1; retain the exact outer rectangle and probe axes.
    split = min(1., case.length)
    xs = np.unique(np.r_[np.linspace(0,split,int(np.ceil(split/h))+1),
                         np.linspace(split,case.length,max(1,int(np.ceil((case.length-split)/(2*h))))+1)])
    ys = np.linspace(0,case.height,int(np.ceil(case.height/h))+1)
    x,y = np.meshgrid(xs,ys)
    grid = np.column_stack((x.ravel(),y.ravel()))
    center = np.asarray(case.center)
    grid = grid[np.linalg.norm(grid-center,axis=1) > case.radius+2.4*h]
    # Keep wall nodes even when coarse refinement rings approach a wall.
    perimeter = np.vstack((np.column_stack((xs,np.zeros_like(xs))),
                           np.column_stack((xs,np.full_like(xs,case.height))),
                           np.column_stack((np.zeros_like(ys),ys)),
                           np.column_stack((np.full_like(ys,case.length),ys))))
    theta = 2*np.pi*np.arange(16*level)/(16*level)
    directions = np.column_stack((np.cos(theta),np.sin(theta)))
    rings = [center+case.radius*directions]
    clearance = min(*center,case.length-center[0],case.height-center[1])
    for offset in (.5*h,h,2*h):
        if case.radius+offset < .95*clearance:
            rings.append(center+(case.radius+offset)*directions)
    points = np.unique(np.vstack((grid,perimeter,*rings)),axis=0)
    triangles = Delaunay(points).simplices.copy()
    centroids = points[triangles].mean(axis=1)
    triangles = triangles[np.linalg.norm(centroids-center,axis=1) > case.radius*(1-1e-12)]
    v = points[triangles]
    a,b=v[:,1]-v[:,0],v[:,2]-v[:,0]
    flip=a[:,0]*b[:,1]-a[:,1]*b[:,0]<0
    triangles[flip]=triangles[flip][:,[0,2,1]]
    edges,counts=np.unique(np.sort(triangles[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0,return_counts=True)
    exterior=edges[counts==1]
    tags=[]
    for edge in exterior:
        xy=points[edge]
        if np.all(np.abs(np.linalg.norm(xy-center,axis=1)-case.radius)<1e-10): tag='cylinder'
        elif np.all(np.abs(xy[:,0])<1e-12): tag='left'
        elif np.all(np.abs(xy[:,0]-case.length)<1e-12): tag='right'
        elif np.all(np.abs(xy[:,1])<1e-12): tag='bottom'
        elif np.all(np.abs(xy[:,1]-case.height)<1e-12): tag='top'
        else: raise ValueError('Delaunay grid did not preserve the prescribed domain boundary')
        tags.append(tag)
    if tags.count('cylinder') != 16*level:
        raise ValueError('Delaunay grid did not preserve all obstacle facets')
    boundary=np.zeros(len(points),bool)
    boundary[np.unique(exterior)]=True
    mesh=Mesh2D(points,triangles.astype(np.int64),boundary,exterior.astype(np.int64),np.array(tags))
    quality=mesh_quality(mesh)
    if quality['holes']!=1 or quality['min_scaled_quality']<1e-4:
        raise ValueError('cylinder mesh quality/topology failed')
    return mesh
