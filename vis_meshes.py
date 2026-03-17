import pyvista as pv
import os
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    mesh_dir = "data/template_LV_vol"
    N_meshes = 10
    N_phases = 10
    N_samp = 100
    np.random.seed(42)

    fig,ax = plt.subplots(2,1)

    ar_all = np.zeros((N_meshes,N_phases*N_samp))
    vol_all = np.zeros((N_meshes,N_phases*N_samp))
    for i in range(N_meshes):
        ar_mesh = []
        vol_mesh = []
        for j in range(N_phases):
            mesh = pv.read(os.path.join(mesh_dir,"LV/mesh_{}/phase{}.vtu".format(i,j)))

            if i==0 and j==0:
                mask = np.random.choice(np.arange(mesh.n_cells),N_samp,replace=False)

            ar = mesh.compute_cell_quality(quality_measure="aspect_ratio")["CellQuality"][mask]
            ar_mesh.append(ar)
            vol = mesh.compute_cell_quality(quality_measure="volume")["CellQuality"][mask]**(1./3.)
            vol_mesh.append(vol)

            # ax[0].scatter(i*np.ones_like(ar)+1,ar,color="C{}".format(j),alpha=0.3,s=2)
            # ax[1].scatter(i*np.ones_like(vol)+1,vol,color="C{}".format(j),alpha=0.3,s=2)
        ar_all[i,:] = np.hstack(ar_mesh)
        vol_all[i,:] = np.hstack(vol_mesh)
    
    ax[0].violinplot(ar_all.T)
    ax[0].set_ylabel("Aspect ratio")
    # ax[0].set_yscale('log')
    ax[0].set_ylim([1,3])
    ax[1].violinplot(vol_all.T)
    ax[1].set_ylabel("Approx. edge length")
    ax[1].set_xlabel("Mesh")
    
    plt.savefig("quality.png")






