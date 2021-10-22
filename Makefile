NOWARN=-wd3180
EXEC=lu-omp
OBJ =  $(EXEC) $(EXEC)-debug $(EXEC)-serial

MATRIX_SIZE=1000
MATRIX_HPC_SIZE=7000
MATRIX_BLOCK_SIZE = 100
W :=`grep processor /proc/cpuinfo | wc -l`

CHECKER=inspxe-cl -collect=ti3 -r check
VIEWER=inspxe-gui check

# flags
OPT=-O2 -g
DEBUG=-O0 -g
OMP=-fopenmp

all: $(OBJ)

# build the debug parallel version of the program
$(EXEC)-debug: $(EXEC).cpp
	icpc $(DEBUG) $(OMP) -o $(EXEC)-debug $(EXEC).cpp -lrt


# build the serial version of the program
$(EXEC)-serial: $(EXEC).cpp
	icpc $(OPT) $(NOWARN) -o $(EXEC)-serial $(EXEC).cpp -lrt -liomp5

# build the optimized parallel version of the program
$(EXEC): $(EXEC).cpp
	icpc $(OPT) $(OMP) -o $(EXEC) $(EXEC).cpp -lrt

#run the optimized program in parallel
runp: $(EXEC)
	@echo use make runp W=nworkers
	./$(EXEC) $(MATRIX_SIZE) $(W)

#run the serial version of your program
runs: $(EXEC)-serial
	@echo use make runs
	./$(EXEC)-serial $(MATRIX_SIZE) 1

#run the optimized program with thread checker
check: $(EXEC)
	@echo use make check W=nworkers
	$(CHECKER) ./$(EXEC) $(MATRIX_SIZE) $(W)

#run the hpc checker
runp-hpc: $(EXEC)
	@echo use make runp-hpc W=nworkers
	@/bin/rm -rf $(EXEC).m $(EXEC).d $(EXEC).hpcstruct
	$(XX) hpcrun -e ivb_ep::MEM_LOAD_UOPS_LLC_MISS_RETIRED:REMOTE_DRAM -e ivb_ep::MEM_LOAD_UOPS_LLC_MISS_RETIRED:LOCAL_DRAM -t -o $(EXEC).m ./$(EXEC)  $(MATRIX_HPC_SIZE) $(W) 
	hpcstruct $(EXEC)
	hpcprof -S $(EXEC).hpcstruct -o $(EXEC).d $(EXEC).m
	hpcviewer $(EXEC).d 

#view the thread checker result
view:
	$(VIEWER)


clean:
	/bin/rm -rf $(OBJ) check
