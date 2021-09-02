#!/usr/bin/env python3

# Copyright (c) 2019-2021, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#import h5py
import time, sys, copy, math
from dolfinx import Function
from dolfinx.fem import locate_dofs_topological
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np


class ModelOrderReduction():

    def __init__(self, params, comm):
        
        self.numhdms = params['numhdms']
        self.numsnapshots = params['numsnapshots']
        
        try: self.snapshotincr = params['snapshotincr']
        except: self.snapshotincr = 1
        
        try: self.snapshotoffset = params['snapshotoffset']
        except: self.snapshotoffset = 0
        
        try: self.numredbasisvec = params['numredbasisvec']
        except: self.numredbasisvec = self.numsnapshots
        
        try: self.print_eigenproblem = params['print_eigenproblem']
        except: self.print_eigenproblem = False
        
        try: self.eigenvalue_cutoff = params['eigenvalue_cutoff']
        except: self.eigenvalue_cutoff = 0.0
        
        try: self.surface_rom = params['surface_rom']
        except: self.surface_rom = []
        
        try: self.snapshotsource = params['snapshotsource']
        except: self.snapshotsource = 'petscvector'
        
        # some sanity checks
        if self.numhdms <= 0:
            raise ValueError('Number of HDMs has to be > 0!')
        if self.numsnapshots <= 0:
            raise ValueError('Number of snapshots has to be > 0!')
        if self.snapshotincr <= 0:
            raise ValueError('Snapshot increment has to be > 0!')
        if self.numredbasisvec <= 0 or self.numredbasisvec > self.numhdms*self.numsnapshots:
            raise ValueError('Number of reduced-basis vectors has to be > 0 and <= number of HDMs times number of snapshots!')
        
        self.hdmfilenames = params['hdmfilenames']

        self.comm = comm
        
    
    # Proper Orthogonal Decomposition
    def POD(self, pb):
        
        if self.comm.rank==0:
            print("Performing Proper Orthogonal Decomposition (POD) ...")
            sys.stdout.flush()
        
        ts = time.time()
        
        locmatsize_u = pb.V_u.dofmap.index_map.size_local * pb.V_u.dofmap.index_map_bs
        matsize_u = pb.V_u.dofmap.index_map.size_global * pb.V_u.dofmap.index_map_bs
        
        # snapshot matrix
        S_d = PETSc.Mat().createDense(size=((locmatsize_u,matsize_u),(self.numhdms*self.numsnapshots)), bsize=None, array=None, comm=self.comm)
        S_d.setUp()
        
        ss, se = S_d.getOwnershipRange()
        
        # gather snapshots (mostly displacements or velocities)
        for h in range(self.numhdms):
            
            for i in range(self.numsnapshots):
                
                step = self.snapshotoffset + (i+1)*self.snapshotincr
            
                field = Function(pb.V_u)
                
                if self.snapshotsource == 'petscvector':
                    # WARNING: Like this, we can only load data with the same amount of processes as it has been written!
                    viewer = PETSc.Viewer().createMPIIO(self.hdmfilenames[h].replace('*',str(step)), 'r', self.comm)
                    field.vector.load(viewer)
                    
                    field.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                
                elif self.snapshotsource == 'rawtxt':
                    
                    # own read function: requires plain txt format of type valx valy valz x z y
                    pb.io.readfunction(field, pb.V_u, self.hdmfilenames[h].replace('*',str(step)))
                    
                else:
                    raise NameError("Unknown snapshotsource!")

                S_d[ss:se, self.numhdms*h+i] = field.vector[ss:se]

        S_d.assemble()

        # covariance matrix
        C_d = S_d.transposeMatMult(S_d) # S^T * S
        
        # setup eigenvalue problem
        eigsolver = SLEPc.EPS()
        eigsolver.create()
        eigsolver.setOperators(C_d)
        eigsolver.setProblemType(SLEPc.EPS.ProblemType.HEP) # Hermitian problem
        eigsolver.setType(SLEPc.EPS.Type.LAPACK)
        eigsolver.setFromOptions()

        # solve eigenvalue problem
        eigsolver.solve()

        nconv = eigsolver.getConverged()
        
        if self.print_eigenproblem:
            if self.comm.rank==0:
                print("Number of converged eigenpairs: %d" % nconv)
                sys.stdout.flush()

        evecs, evals = [], []
        numredbasisvec_true = 0

        if nconv > 0:
            # create the results vectors
            vr, _ = C_d.getVecs()
            vi, _ = C_d.getVecs()
            
            if self.print_eigenproblem:
                if self.comm.rank==0:
                    print("   k                        ||Ax-kx||/||kx||")
                    print("   ----------------------   ----------------")
                    sys.stdout.flush()
                    
            for i in range(self.numredbasisvec):
                k = eigsolver.getEigenpair(i, vr, vi)
                error = eigsolver.computeError(i)
                if self.print_eigenproblem:
                    if k.imag != 0.0:
                        if self.comm.rank==0:
                            print('{:<3s}{:<4.4e}{:<1s}{:<4.4e}{:<1s}{:<3s}{:<4.4e}'.format(' ',k.real,'+',k.imag,'j',' ',error))
                            sys.stdout.flush()
                    else:
                        if self.comm.rank==0:
                            print('{:<3s}{:<4.4e}{:<15s}{:<4.4e}'.format(' ',k.real,' ',error))
                            sys.stdout.flush()

                # store
                evecs.append(copy.deepcopy(vr)) # need copy here, otherwise reference changes
                evals.append(k.real)
                
                if k.real > self.eigenvalue_cutoff: numredbasisvec_true += 1

        if self.numredbasisvec != numredbasisvec_true:
            if self.comm.rank==0:
                print("Eigenvalues below cutoff tolerance: Number of reduced basis vectors for ROB changed from %i to %i." % (self.numredbasisvec,numredbasisvec_true))
                sys.stdout.flush()
        
        # pop out undesired ones
        for i in range(self.numredbasisvec-numredbasisvec_true):
            evecs.pop(-1)
            evals.pop(-1)

        # eigenvectors, scaled with 1 / sqrt(eigenval)
        # calculate first numredbasisvec_true POD modes
        self.Phi = np.zeros((matsize_u, numredbasisvec_true))
        for i in range(numredbasisvec_true):
            self.Phi[ss:se,i] = S_d * evecs[i] / math.sqrt(evals[i])       

        # build reduced basis - either only on designated surfaces or for the whole model
        if bool(self.surface_rom):
            self.build_reduced_surface_basis(pb,numredbasisvec_true,ts)
        else:
            self.build_reduced_basis(pb,numredbasisvec_true,ts)


    def build_reduced_basis(self, pb, rb, ts):

        locmatsize_u = pb.V_u.dofmap.index_map.size_local * pb.V_u.dofmap.index_map_bs
        matsize_u = pb.V_u.dofmap.index_map.size_global * pb.V_u.dofmap.index_map_bs

        # create aij matrix - important to specify an approximation for nnz (number of non-zeros per row) for efficient value setting
        self.V = PETSc.Mat().createAIJ(size=((locmatsize_u,matsize_u),(rb)), bsize=None, nnz=(rb,locmatsize_u), csr=None, comm=self.comm)
        self.V.setUp()
        
        vrs, vre = self.V.getOwnershipRange()
        
        # set Phi columns
        self.V[vrs:vre,:] = self.Phi[vrs:vre,:]
  
        self.V.assemble()

        te = time.time() - ts
        
        if self.comm.rank==0:
            print("POD done... Created reduced-order basis for ROM. Time: %.4f s" % (te))
            sys.stdout.flush()


    def build_reduced_surface_basis(self, pb, rb, ts):

        locmatsize_u = pb.V_u.dofmap.index_map.size_local * pb.V_u.dofmap.index_map_bs
        matsize_u = pb.V_u.dofmap.index_map.size_global * pb.V_u.dofmap.index_map_bs

        # get boundary dofs which should be reduced
        fn=[]
        for i in range(len(self.surface_rom)):
            
            # these are local node indices!
            fnode_indices_local = locate_dofs_topological(pb.V_u, pb.io.mesh.topology.dim-1, pb.io.mt_b1.indices[pb.io.mt_b1.values == self.surface_rom[i]])

            # get global indices
            fnode_indices = pb.V_u.dofmap.index_map.local_to_global(fnode_indices_local)
            
            # gather indices
            fnode_indices_gathered = self.comm.allgather(fnode_indices)
            
            # flatten indices from all the processes
            fnode_indices_flat = [item for sublist in fnode_indices_gathered for item in sublist]

            # remove duplicates
            fnode_indices_unique = list(dict.fromkeys(fnode_indices_flat))

            fn.append(fnode_indices_unique)
            
        # flatten list
        fn_flat = [item for sublist in fn for item in sublist]

        # remove duplicates
        fn_unique = list(dict.fromkeys(fn_flat))
        
        # now make list of dof indices according to block size
        fd=[]
        for i in range(len(fn_unique)):
            for j in range(pb.V_u.dofmap.index_map_bs):
                fd.append(pb.V_u.dofmap.index_map_bs*fn_unique[i]+j)

        # make set for faster checking
        fd_set = set(fd)

        # number of non-reduced "bulk" dofs
        ndof_bulk = matsize_u - len(fd)

        # row loop to get entries (1's) for "non-reduced" dofs
        nr, a = 0, 0
        row_1, col_1, col_fd = [], [], []

        for row in range(matsize_u):
            
            if row in fd_set:
                # increase counter for number of reduced dofs
                nr += 1
                # column shift if we've exceeded the number of reduced basis vectors
                if nr <= rb: col_fd.append(row)
                if nr > rb: a += 1
            else:
                # column id of non-reduced dof (left-shifted by a)
                col_id = row-a
                # store
                row_1.append(row)
                col_1.append(col_id)
        
        # make set for faster checking
        col_fd_set = set(col_fd)

        # create aij matrix - important to specify an approximation for nnz (number of non-zeros per row) for efficient value setting
        self.V = PETSc.Mat().createAIJ(size=((locmatsize_u,matsize_u),(rb+ndof_bulk)), bsize=None, nnz=(2*rb,locmatsize_u), csr=None, comm=self.comm)
        self.V.setUp()
        
        vrs, vre = self.V.getOwnershipRange()

        # now, eliminate all rows in Phi that do not belong to a surface dof which is reduced
        for i in range(vrs, vre):
            if i not in fd_set:
                self.Phi[i,:] = 0.0

        # now set entries
        for k in range(len(row_1)):
            self.V[row_1[k],col_1[k]] = 1.0

        # column loop to insert columns of Phi
        n=0
        for col in range(rb+ndof_bulk):
            # set Phi column
            if col in col_fd_set:
                self.V[vrs:vre,col] = self.Phi[vrs:vre,n]
                n += 1

        self.V.assemble()

        te = time.time() - ts
        
        if self.comm.rank==0:
            print("POD done... Created reduced-order basis for surface ROM on boundary id(s) "+str(self.surface_rom)+". Time: %.4f s" % (te))
            sys.stdout.flush()