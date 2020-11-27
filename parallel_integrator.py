class IntegratorParallelArbitraryGridOptimized(IntegratorArbitraryGrid):
    # this integrator tries to even the load on each process by evening the amount of uncached points each process has to evaluate
    # this should lead to more even execution time and less waiting for each process
    def __call__(self, f: Callable[[Tuple[float, ...]], float], numPoints: Sequence[int], start: Sequence[float], end: Sequence[float]) -> Sequence[float]:
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        # gathering all work packages (equivalent to IntegratorAbitraryGrid)
        if rank == 0:
            
            dim = len(start)
            offsets = np.ones(dim, dtype=np.int64)
            gridsize = np.int64(1)
            
            for i in range(dim):
                gridsize *= np.int64(numPoints[i])
                if i != 0:
                    offsets[i] = offsets[i - 1] * int(numPoints[i - 1])
            indexvectors = []
            for i in range(gridsize):
                indexvector = np.empty(dim, dtype=int)
                rest = i
                for d in range(dim - 1, -1, -1):
                    indexvector[d] = int(rest / offsets[d])
                    rest = rest % offsets[d]
                indexvectors.append(indexvector)
            
            # dividing into even workload chunks
            cached = [vec for vec in indexvectors if self.iscached(indexvector, f)]
            uncached = [vec for vec in indexvectors if not self.iscached(indexvector, f)]
            chunks = [cached[i::size] + uncached[i::size] for i in range(size)]
        else:
            chunks = None
        
        # distributing work packages
        packet = comm.scatter(chunks, root=0)
        localresult = 0.0
        for vec in packet:
            localresult += self.integrate_point(f, vec)
        
        results = comm.gather(localresult)

       # merge caches 
        if isinstance(f, Function):
            caches : List[dict] = comm.gather(f.f_dict)
            if rank == 0:
                all_keys = set([key for cache in caches for key in cache.keys()])
                new_cache = {}
                for cache in caches:
                    new_cache.update(cache)
            else:
                new_cache = None
            new_cache = comm.bcast(new_cache, root=0)
            f.f_dict.update(new_cache)
            f.old_f_dict = new_cache

        # return the same result to all processes so they can continue
        if rank == 0:
            global_result = sum(results) 
        else:
            global_result = None
        global_result = comm.bcast(global_result) # scattering so each process returns a valid result
        return global_result

    def iscached(self, indexvector, f):
        pos = self.grid.getCoordinate(indexvector)
        return tuple(pos) in f.f_dict


class IntegratorParallelArbitraryGrid(IntegratorArbitraryGrid):
    def __call__(self, f: Callable[[Tuple[float, ...]], float], numPoints: Sequence[int], start: Sequence[float], end: Sequence[float]) -> Sequence[float]:
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        # gathering all work packages (equivalent to IntegratorAbitraryGrid)
        if rank == 0:
            dim = len(start)
            offsets = np.ones(dim, dtype=np.int64)
            gridsize = np.int64(1)
            
            for i in range(dim):
                gridsize *= np.int64(numPoints[i])
                if i != 0:
                    offsets[i] = offsets[i - 1] * int(numPoints[i - 1])
            indexvectors = []
            for i in range(gridsize):
                indexvector = np.empty(dim, dtype=int)
                rest = i
                for d in range(dim - 1, -1, -1):
                    indexvector[d] = int(rest / offsets[d])
                    rest = rest % offsets[d]
                indexvectors.append(indexvector)
                
            chunks = [indexvectors[i::size] for i in range(size)]

        else:
            chunks = None
        
        # distributing work packages
        indexvectors = comm.scatter(chunks, root=0)
        localresult = 0.0
        for vec in indexvectors:
            localresult += self.integrate_point(f, vec)
        results = comm.gather(localresult)

        # merge caches 
        if isinstance(f, Function):
            caches : List[dict] = comm.gather(f.f_dict)
            if rank == 0:
                new_cache = {}
                for cache in caches:
                    new_cache.update(cache)
            else:
                new_cache = None
            new_cache = comm.bcast(new_cache)
            f.f_dict = new_cache

        # return the same result to all processes so they can continue
        if rank == 0:
            global_result = sum(results) 
        else:
            global_result = None
        global_result = comm.bcast(global_result) # scattering so each process returns a valid result
        return global_result

