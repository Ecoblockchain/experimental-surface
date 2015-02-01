#!/usr/bin/env python3

import bpy
import bmesh

def timeit(method):
  from time import time
  def timed(*args, **kw):
    ts = time()
    result = method(*args, **kw)
    te = time()
    print('\t{:s} {:2.2f}'.format(method.__name__,te-ts))
    return result
  return timed

class Surface(object):

  def __init__(self,
               noise,
               stp_attract,
               stp_reject,
               nearl,
               farl,
               remesh_mode,
               remesh_scale,
               remesh_depth,
               remesh_itt,
               obj_name='geom',
               nmax=10**6):

    self.__delete_default()

    self.noise = noise
    self.stp_attract = stp_attract
    self.stp_reject = stp_reject
    self.nearl = nearl
    self.farl = farl
    self.remesh_mode = remesh_mode
    self.remesh_scale = remesh_scale
    self.remesh_depth = remesh_depth
    self.remesh_itt = remesh_itt

    self.obj_name = obj_name
    self.nmax = nmax

    obj = bpy.data.objects[self.obj_name]
    bpy.ops.object.select_pattern(pattern=self.obj_name)
    bpy.context.scene.objects.active = obj
    bpy.data.objects[self.obj_name].select = True
    self.obj = obj

    self.itt = 0

    return

  def __delete_default(self):
    '''
    Removes default objects from the scene.
    '''

    names = ['Cube', 'Icosphere', 'Sphere']
    for name in names:
      try:
        bpy.data.objects[name].select = True
        bpy.ops.object.delete()
      except Exception:
        pass

    return

  def save(self,fn):

    bpy.ops.wm.save_as_mainfile(filepath=fn)

    return

  def __get_bmesh(self):
    '''
    Convert the object to a bmesh.
    '''

    bpy.data.objects[self.obj_name].select = True
    self.obj = bpy.context.active_object
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(self.obj.data)

    return bm

  def __to_mesh(self):
    '''
    Must be called to write the changes performed on bm from __get_bmesh to
    persist.
    '''

    bpy.ops.object.mode_set(mode='OBJECT')

    return

  def __triangulate(self,v):
    '''
    Performs triangulation on vertices in v ((n,3) array)
    '''

    from scipy.spatial import Delaunay as delaunay
    flags = 'QJ Qc Pp'

    tri = delaunay(v,
                   incremental=False,
                   qhull_options=flags)

    return tri

  #@timeit
  def __get_tri_edges(self,tri):
    '''
    Get all unique internal edges (vertex-vertex pairs) of delaunay
    triangulation tri.
    '''

    from numpy import row_stack, all, transpose, tile, sort,\
      column_stack, arange

    indices,indptr = tri.vertex_neighbor_vertices
    edges = []
    for i,a,b in column_stack((arange(len(indices)-1),
                               indices[:-1],indices[1:])):
      edges.append(sort(transpose((tile(i,b-a),indptr[a:b])),axis=1))

    stacked = row_stack(edges)
    mask = all(stacked>-1,axis=1)
    stacked = stacked[mask]
    final = row_stack(set((tuple(sorted(e)) for e in stacked)))

    return final

  #@timeit
  def update_data_structure(self):
    '''
    Constructs the necessary data structures
    '''

    from numpy import row_stack, array, diff
    from numpy.random import randint


    bm = self.__get_bmesh()
    self.bm = bm

    vertices = row_stack([v.co for v in bm.verts])
    self.vertices = vertices
    self.vnum = len(vertices)

    faces = list(f for f in bm.faces)
    faces_indices = array([f.index for f in faces],'int')
    self.faces_indices = faces_indices
    if any(diff(faces_indices)!=1):
      '''
      in the rest of the code assume that the indices of faces in
      faces_indics are ordered and have no gaps. this is not a very good
      assumption in general, but it has worked here so far.
      '''
      #TODO: Should rewrite the data structures/code to avoid this
      raise AssertionError('there is a gap in the indices of bm.faces.')

    face_vertex_indices = [[v.index for v in f.verts] for f in faces]
    self.face_vertex_indices = face_vertex_indices

    edges = list(bm.edges)
    self.edges = edges

    face_normals = row_stack([f.normal for f in faces])
    self.face_normals = face_normals

    face_centroids = row_stack([f.calc_center_bounds() for f in faces])
    self.face_centroids = face_centroids

    tri = self.__triangulate(face_centroids)

    tri_edges = self.__get_tri_edges(tri) # slow
    self.tri_edges = tri_edges

    return

  #@timeit
  def balance(self):
    '''
    Performs growth step. Neighboring faces are pulled towards each other,
    unless they are closer than `self.nearl`, and faces that are connected via
    a Delaunay triangulation of the face centroids are pushed apart if they are
    within `self.farl` of each other.
    '''

    from numpy import zeros, reshape, diff, logical_and
    from numpy.linalg import norm

    nearl = self.nearl
    farl = self.farl

    vertices = self.vertices
    face_vertex_indices = self.face_vertex_indices
    face_centroids = self.face_centroids
    tri_edges = self.tri_edges
    edges = self.edges
    nmax = self.nmax

    fnum = len(face_centroids)

    dx_attract = zeros((fnum,3),'float')
    dx_reject = zeros((fnum,3),'float')

    ## attract
    for edge in edges:
      try:
        f1,f2 = edge.link_faces

        vdx = face_centroids[f1.index]-face_centroids[f2.index]
        nrm = norm(vdx)

        if nrm<nearl:
          continue

        force = -vdx/nrm*0.5

        dx_attract[f1.index,:] += force
        dx_attract[f2.index,:] -= force
      except Exception:
        pass

    vvdx = diff(self.face_centroids[tri_edges,:],axis=1).squeeze()
    nrm = norm(vvdx,axis=1)

    mask = logical_and(nrm<farl,nrm>0.)
    force = (farl/reshape(nrm[mask],(-1,1))-1.)*vvdx[mask,:]

    for (a,b),f in zip(tri_edges[mask,:],force):
      dx_reject[a,:] -= f
      dx_reject[b,:] += f

    dx_attract *= self.stp_attract
    dx_reject *= self.stp_reject

    dx = dx_attract + dx_reject

    self.print_force_info(dx_reject,dx_attract, dx)

    return dx

  def print_force_info(self,dx_reject,dx_attract,dx):

    from numpy.linalg import norm
    from numpy import abs

    rsum = norm(abs(dx_reject),axis=0)
    asum = norm(abs(dx_attract),axis=0)
    dsum = norm(abs(dx),axis=0)
    rstr = 'reject:  max: {:4.4f} * {:4.4f} {:4.4f} {:4.4f}'\
        .format(abs(dx_reject).max(),*rsum)
    astr = 'attract: max: {:4.4f} * {:4.4f} {:4.4f} {:4.4f}'\
        .format(abs(dx_attract).max(),*asum)
    dstr = 'dx: max: {:4.4f} * {:4.4f} {:4.4f} {:4.4f}'\
        .format(abs(dx).max(),*dsum)
    print(rstr)
    print(astr)
    print(dstr)
    print()

    return

  #@timeit
  def vertex_noise(self):
    '''
    Randomly move the vertices around a little.
    '''

    from numpy.random import multivariate_normal
    from numpy.linalg import norm
    from numpy import reshape

    vnum = self.vnum
    x = multivariate_normal(mean=[0]*3,cov=[[1,0,0],[0,1,0],[0,0,1]],size=vnum)
    l = reshape(norm(x,axis=1),(vnum,1))
    x /= l

    self.vertices += x*self.noise

    return

  def light_react(self,dx):

    from numpy import zeros, logical_not, ones, dot, array, abs, reshape
    from numpy.linalg import norm

    light_source = array([0,0,200],'float')
    vertices = self.vertices
    face_vertex_indices = self.face_vertex_indices
    face_normals = self.face_normals

    facemap = zeros(len(face_vertex_indices),'bool')
    facescale = zeros(len(face_vertex_indices),'float')

    for i,c in enumerate(self.face_centroids):
      result,obj,mat,loc,normal = bpy.context.scene.ray_cast(light_source,c)
      if result:
        facemap[i] = True

      else:
        diff = (light_source-c)
        diff /= norm(diff)
        scale = abs(dot(diff,face_normals[i,:]))
        facescale[i] = scale

    facescale /= facescale.max()
    facescale[facescale<0.1] = 0.1

    dx *= reshape(facescale,(-1,1))

    return

  def vertex_update(self):
    '''
    Write everything back to the mesh
    '''

    vertices = self.vertices
    for i,v in enumerate(self.bm.verts):
      v.co = vertices[i,:]

    return

  def remesh(self):
    '''
    Uses blender Remesh modifier to reconstruct the mesh.
    '''

    bpy.ops.object.modifier_add(type='SMOOTH')
    self.obj.modifiers['Smooth'].factor = 0.2
    self.obj.modifiers['Smooth'].iterations = 2
    bpy.ops.object.modifier_apply(modifier='Smooth',apply_as='DATA')

    bpy.ops.object.modifier_add(type='REMESH')
    self.obj.modifiers['Remesh'].mode = self.remesh_mode
    self.obj.modifiers['Remesh'].scale = self.remesh_scale
    self.obj.modifiers['Remesh'].octree_depth = self.remesh_depth
    bpy.ops.object.modifier_apply(modifier='Remesh',apply_as='DATA')

    return

  #@timeit
  def step(self):
    '''
    Do all the things.
    '''

    self.itt += 1

    self.update_data_structure()
    self.vertex_noise()
    dx = self.balance()
    #self.light_react(dx)

    for i,jj in enumerate(self.face_vertex_indices):
      self.vertices[jj,:] += dx[i,:]

    self.vertex_update()
    self.__to_mesh()

    if not self.itt%self.remesh_itt:
      self.remesh()

    return

def main():

  from time import time

  steps = 1000

  noise = 0.08 # 0.0008
  stp_attract = 0.08 #0.05
  stp_reject = 0.01 #0.01
  #stp_reject = array([1,1,0.3],'float')*0.1
  nearl = 0.1
  farl = 5.0

  remesh_mode = 'SMOOTH'
  remesh_scale = 0.6
  remesh_depth = 6
  remesh_itt = 15

  obj_name = 'geom'
  out_fn = 'c'

  t1 = time()

  S = Surface(noise=noise,
             stp_attract=stp_attract,
             stp_reject=stp_reject,
             nearl=nearl,
             farl=farl,
             remesh_mode=remesh_mode,
             remesh_scale=remesh_scale,
             remesh_depth=remesh_depth,
             remesh_itt=remesh_itt,
             obj_name=obj_name)

  for i in range(steps):
    try:
      S.step()
      itt = S.itt

      if not itt%5:
        print(itt,S.vertices.shape)

      if not itt%15:
        fnitt = './res/{:s}_{:05d}.blend'.format(out_fn,itt)
        S.save(fnitt)
        print(fnitt)

    except KeyboardInterrupt:
      print('KeyboardInterrupt')
      break

  S.save('./res/{:s}.blend'.format(out_fn))

  print('\ntime:',time()-t1,'\n\n')

  return


if __name__ == '__main__':

  if False:
    import pstats
    import cProfile
    pfilename = 'profile.profile'
    cProfile.run('main()',pfilename)
    p = pstats.Stats(pfilename)
    p.strip_dirs().sort_stats('cumulative').print_stats()
  else:
    main()

