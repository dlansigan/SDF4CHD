import numpy as np
from sv import meshing
import os
import vtk
import argparse
import sys
import utils
from vtk_utils.vtk_utils import write_vtk_polydata, write_vtu
import glob

# Make sure you are using environment with simvascular installed!
# Usage:
#    simvascular --python -- sv_mesh.py --template_dir data/template_LV_vol

def generate_mesh(fn, ops, q):
    mesher = meshing.create_mesher(meshing.Kernel.TETGEN)  
    mesher.load_model(fn)
    mesher.set_walls([1])
    face_ids = mesher.get_model_face_ids()
    options = meshing.TetGenOptions(**ops)
    options.no_merge = True
    options.no_bisect = True
    options.optimization = 3
    options.quality_ratio = q
    mesher.generate_mesh(options)
    vol_mesh = mesher.get_mesh()
    surf_mesh = mesher.get_surface()
    return vol_mesh, surf_mesh, face_ids

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_dir", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quality", type=float, default=1.0)
    args = parser.parse_args()

    template_dir = args.template_dir 
    overwrite = args.overwrite
    face_names = ["wall","av","mv"]  

    mesh_ops = {
            'surface_mesh_flag': False,
            'volume_mesh_flag': True,
            'global_edge_size': 0.01, 
    }

    surf_fn = os.path.join(args.template_dir,"LV.vtp")
    vol_fn = os.path.join(args.template_dir,"LV.vtu")
    if not os.path.exists(vol_fn) or overwrite: # If volume mesh exists, assumes all others do, too

        vol_mesh, surf_mesh, face_ids = generate_mesh(surf_fn,mesh_ops,args.quality)

        # print(vol_mesh)

        # # Extract faces
        # for i,id in enumerate(face_ids):
        #     face_fn = os.path.join(save_dir,case_dir,"mesh-surfaces","{}.vtp".format(face_names[i]))
        #     face = utils.threshold_polydata(surf_mesh, "ModelFaceID", (id,id))
        #     write_vtk_polydata(face,face_fn)

        # write_vtk_polydata(surf_mesh,surf_fn)
        write_vtu(vol_mesh,vol_fn)
        print(vol_mesh.GetNumberOfCells())