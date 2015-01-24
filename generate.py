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

    bpy.data.objects[self.obj_name].select = True
    self.obj = bpy.context.active_object
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(self.obj.data)

    return bm

  def __to_mesh(self):

    bpy.ops.object.mode_set(mode='OBJECT')

    return

  def __triangulate(self,v):

    from scipy.spatial import Delaunay as delaunay
    flags = 'QJ Qc Pp'

    tri = delaunay(v,
                   incremental=False,
                   qhull_options=flags)

    simplex_vertex = tri.simplices
    return simplex_vertex

  def __get_face_centroids(self,vertices,face_vertices):

    from numpy import mean, row_stack

    centroids = {}

    for i,f in face_vertices.items():
      #centroids[i] = mean(row_stack([v.co for v in f]), axis=0)
      centroids[i] = mean([v.co for v in f],axis=0)

    return centroids

  def __get_tri_edges(self,tri_simplex_vertex):

    from numpy import roll, row_stack, all

    _,width = tri_simplex_vertex.shape
    tri_edges = row_stack( [roll(tri_simplex_vertex,i)[:,:2] for i in range(width)] )

    mask = all(tri_edges>-1,axis=1)
    tri_edges = tri_edges[mask]
    tri_edges = row_stack(set((tuple(e) for e in tri_edges)))

    return tri_edges

  #@timeit
  def update_face_structure(self):

    from numpy import row_stack

    bm = self.__get_bmesh()
    self.bm = bm

    vertices = row_stack([v.co for v in bm.verts])
    self.vertices = vertices
    self.vnum = len(vertices)

    faces = [f for f in bm.faces]
    face_vertex_indices = {f.index:[v.index for v in f.verts] for f in faces}
    face_vertices = {f.index:list(f.verts) for f in faces}
    self.face_vertices = face_vertices
    self.face_vertex_indices = face_vertex_indices

    edges = list(bm.edges)
    self.edges = edges #print(list(e.link_faces))

    face_centroids = self.__get_face_centroids(vertices,face_vertices)
    self.face_centroids = face_centroids

    tri_simplex_vertex = self.__triangulate(row_stack([f for i,f in self.face_centroids.items()]))
    tri_edges = self.__get_tri_edges(tri_simplex_vertex) #
    self.tri_edges = tri_edges

    return

  #@timeit
  def balance(self):

    from numpy import zeros, reshape, sum
    from numpy.linalg import norm

    nearl = self.nearl
    farl = self.farl

    vertices = self.vertices
    face_vertices = self.face_vertices
    face_vertex_indices = self.face_vertex_indices
    face_centroids = self.face_centroids
    tri_edges = self.tri_edges
    edges = self.edges
    vnum = self.vnum
    nmax = self.nmax

    dx_attract = zeros((nmax,3),'float')
    dx_reject = zeros((nmax,3),'float')

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

    # reject
    ## TODO: vectorize
    for a,b in tri_edges:
      v1 = face_centroids[a]
      v2 = face_centroids[b]
      vdx = v2-v1
      nrm = norm(vdx)

      if nrm<farl:
        force = (farl/nrm-1.)*vdx
        dx_reject[a,:] -= force
        dx_reject[b,:] += force

    dx_attract *= self.stp_attract
    dx_reject *= self.stp_reject
    dx = dx_attract + dx_reject

    #rsum = norm(abs(dx_reject),axis=0)
    #asum = norm(abs(dx_attract),axis=0)
    #rstr = 'reject: {:4.4f} {:4.4f} {:4.4f}'.format(*rsum)
    #astr = 'attract: {:4.4f} {:4.4f} {:4.4f}'.format(*asum)
    #print(rstr,astr)

    for i,jj in face_vertex_indices.items():
      vertices[jj,:] += dx[i,:]

    return

  def vertex_noise(self):

    from numpy.random import multivariate_normal
    from numpy.linalg import norm
    from numpy import reshape

    vnum = self.vnum
    x = multivariate_normal(mean=[0]*3,cov=[[1,0,0],[0,1,0],[0,0,1]],size=vnum)
    l = reshape(norm(x,axis=1),(vnum,1))
    x /= l

    self.vertices += x*self.noise

    return

  def vertex_update(self):

    vertices = self.vertices
    for i,v in enumerate(self.bm.verts):
      v.co = vertices[i,:]

    return

  def remesh(self):

    bpy.ops.object.modifier_add(type='REMESH')
    self.obj.modifiers['Remesh'].mode = self.remesh_mode
    self.obj.modifiers['Remesh'].scale = self.remesh_scale
    self.obj.modifiers['Remesh'].octree_depth = self.remesh_depth
    bpy.ops.object.modifier_apply(modifier='Remesh',apply_as='DATA')

    return

  #@timeit
  def step(self):

    from time import time

    t1 = time()

    self.itt += 1

    self.update_face_structure()
    self.vertex_noise()
    t2 = time()
    self.balance()
    self.vertex_update()
    self.__to_mesh()

    if not self.itt%self.remesh_itt:
      self.remesh()

    t3 = time()

    print(t3-t1, (t2-t1)/(t3-t1))

    return

def main():

  from time import time
  from numpy import array

  steps = 20

  noise = 0.0008
  stp_attract = 0.02
  stp_reject = array([1,1,0.2],'float')*0.005
  nearl = 0.1
  farl = 4.0

  remesh_mode = 'SMOOTH'
  remesh_scale = 0.65
  remesh_depth = 6
  remesh_itt = 12

  obj_name = 'geom'
  out_fn = 'a_res_ani'

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
      #if not itt%20:
        #fnitt = './res/{:s}_{:05d}.blend'.format(out_fn,itt)
        #S.save(fnitt)
        #print(fnitt)

    except KeyboardInterrupt:
      print('KeyboardInterrupt')
      break


  S.save('./res/{:s}.blend'.format(out_fn))

  print('\ntime:',time()-t1,'\n\n')

  return


if __name__ == '__main__':

  if True:
    import pstats
    import cProfile
    pfilename = 'profile.profile'
    cProfile.run('main()',pfilename)
    p = pstats.Stats(pfilename)
    p.strip_dirs().sort_stats('cumulative').print_stats()
  else:
    main()

