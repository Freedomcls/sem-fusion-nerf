import torch
from collections import defaultdict

import numpy as np
import mcubes
import os.path as osp
from plyfile import PlyData, PlyElement
import open3d as o3d
import time
import skimage

def convert_sdf_samples_to_ply(
    numpy_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
    level=0.0
):
    """
    Convert sdf samples to .ply
    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    # try:
    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level, spacing=[voxel_size] * 3
    )
    # except:
    #     pass

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = PlyElement.describe(verts_tuple, "vertex")
    el_faces = PlyElement.describe(faces_tuple, "face")

    ply_data = PlyData([el_verts, el_faces])
    ply_data.write(ply_filename_out)
    print(f"wrote to {ply_filename_out}")



def gen3d_via_sigmas(sigmas, shapes, 
        sigma_threshold = 1.0, 
        output_dir = "./", 
        name = "unnamed",
        vol_range = [[-1,1], [-1, 1], [-1, 1]], # X Y Z range
    ):
    if isinstance(sigmas, torch.Tensor):
        sigmas = sigmas.cpu().numpy()
    assert isinstance(sigmas, np.ndarray)
    # import pdb; pdb.set_trace()
    sigmas = np.maximum(sigmas, 0)
    nb_grid_x, nb_grid_y, nb_grid_z = shapes
    sigmas = sigmas.reshape(*shapes)
    vertices, triangles = mcubes.marching_cubes(sigmas, sigma_threshold)
    mcubes.export_mesh(vertices, triangles, osp.join(output_dir, f"{name}.dae"))
    vertices_ = (vertices / nb_grid_x).astype(np.float32)
    x_min, x_max = vol_range[0]
    y_min, y_max = vol_range[1]
    z_min, z_max = vol_range[2]

    # denormalize
    x_ = (x_max-x_min) * vertices_[:, 0] + x_min
    y_ = (y_max-y_min) * vertices_[:, 1] + y_min
    vertices_[:, 0] = x_
    vertices_[:, 1] = y_
    vertices_[:, 2] = (z_max-z_min) * vertices_[:, 2] + z_min

    vertices_.dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    face = np.empty(len(triangles), dtype=[('vertex_indices', 'i4', (3,))])
    face['vertex_indices'] = triangles
    # write to ply
    PlyData([PlyElement.describe(vertices_[:, 0], 'vertex'), 
        PlyElement.describe(face, 'face')]).write(osp.join(output_dir, f"{name}_coarse.ply"))    
    
    print('Removing noise ...')
    mesh = o3d.io.read_triangle_mesh(osp.join(output_dir, f"{name}_coarse.ply"))
    idxs, count, _ = mesh.cluster_connected_triangles()
    max_cluster_idx = np.argmax(count)
    triangles_to_remove = [i for i in range(len(face)) if idxs[i] != max_cluster_idx]
    mesh.remove_triangles_by_index(triangles_to_remove)
    mesh.remove_unreferenced_vertices()
    print(f'Mesh has {len(mesh.vertices)/1e6:.2f} M vertices and {len(mesh.triangles)/1e6:.2f} M faces.')



