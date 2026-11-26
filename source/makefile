# ============================================================
# Implicit SPH (ISPH) + DeepONet preconditioners
#   PPE : pressure Poisson equation      (BiCGStab, deeponet/ppe_deeponet.inc / deeponet/ppe_deeponet/)
#   VIS : implicit viscosity, solidifying (CG,       deeponet/vis_deeponet.inc / deeponet/vis_deeponet/)
#
#   Source layout:
#     main_implicit.cpp errorfunc.* log.*  core solver
#     fvm/       FVM carrier air (fvm_core, fvm_air, fvm3d_air)
#     spray/     SPH spray + aerosol (spray, aerosol, aerosol_field[3d], aerosol_capillary)
#     deeponet/  DeepONet / POU preconditioners (.inc + Python training dirs, baseline_compare)
#     amgx/      AmgX solvers (amgx_ppe, amgx_vis)
#     body/      solid / rigid bodies, surface tension
#     legacy/    old main.cpp / main_semi.cpp (not built)
#
#   Normal run (Jacobi solvers, unchanged)
#     make GPU=1                         -> SPH_acc
#     make CPU=1                         -> SPH_omp
#
#   Select the solver(s) with PPE=1, VIS=1 or ALL=1 (= PPE + VIS).
#   Without a selector DATASET=1 / ONET=1 / train mean PPE (as before).
#
#   1) Training-data generation (snapshots every EVERY solves -> DATA_ROOT/<case dir name>/)
#     make GPU=1 DATASET=1 PPE=1         -> SPH_acc_dataset       (ppe_*.bin)
#     make GPU=1 DATASET=1 VIS=1         -> SPH_acc_dataset_vis   (vis_*.bin)
#     make GPU=1 DATASET=1 ALL=1         -> SPH_acc_dataset_all   (both, same case directory)
#   2) Training (all cases under DATA_ROOT, or DATA="dir1 dir2 ...")
#     make train PPE=1                   -> deeponet/ppe_deeponet/trained_model/ppe_deeponet.bin
#     make train VIS=1                   -> deeponet/vis_deeponet/trained_model/vis_deeponet.bin
#     make train ALL=1                   (both, one after the other)
#   3) Run with the DeepONet preconditioner(s)
#     make GPU=1 ONET=1 PPE=1            -> SPH_acc_onet
#     make GPU=1 ONET=1 VIS=1            -> SPH_acc_onet_vis
#     make GPU=1 ONET=1 ALL=1            -> SPH_acc_onet_all
#
#   Spray scavenging (FVM air from ../../VOF_FVM_2Phase + SPH spray + aerosol):
#     make GPU=1 SPRAY=1                 -> SPH_acc_spray  (-DSPH_FVM_AIR; combines with ONET/DATASET)
#     make GPU=1 SPRAY=1 DIM=3           -> SPH_acc3d_spray (3D: fvm/fvm3d_air.inc + spray/aerosol_field3d.inc)
#   Options: DIM=2|3 (default 2), GPU_ARCH=cc89, NVCXX=<nvc++>,
#            MODEL=<ppe_deeponet.bin>, VIS_MODEL=<vis_deeponet.bin>,
#            EVERY=<snapshot interval>, VIS_EVERY=<viscosity snapshot interval, default EVERY>,
#            DATA_ROOT=<snapshot root, default ../results/trained_data>,
#            EPOCHS=30, VIS_EPOCHS=$(EPOCHS), VAL_FRACTION=0 (0.2 holds out whole runs)
#   The classic targets below (make SPH_acc, make SPH_gcc, ...) are unchanged.
# ============================================================

CPU      ?= 0
GPU      ?= 0
ONET     ?= 0
DATASET  ?= 0
PPE      ?= 0
VIS      ?= 0
ALL      ?= 0
SPRAY    ?= 0
DIM      ?= 2
GPU_ARCH ?= cc89
EVERY    ?= 50
VIS_EVERY ?= $(EVERY)
MODEL    ?= $(abspath deeponet/ppe_deeponet/trained_model/ppe_deeponet.bin)
VIS_MODEL ?= $(abspath deeponet/vis_deeponet/trained_model/vis_deeponet.bin)
NVCXX    ?= nvc++
PYTHON   ?= python3
DATA_ROOT ?= $(abspath ../results/trained_data)
DATA     ?=
EPOCHS   ?= 30
VIS_EPOCHS ?= $(EPOCHS)
VAL_FRACTION ?= 0

# POU=1  : partition-of-unity coarse space for the viscosity preconditioner (deeponet/vis_pou.inc, key VisPou 1)
# AMGX=1 : AmgX solvers for PPE / viscosity (amgx/amgx_ppe.inc, amgx/amgx_vis.inc, keys PpeAmgx / VisAmgx)
POU      ?= 0
AMGX     ?= 0
AMGX_DIR ?= $(HOME)/opt/amgx
CUDA_DIR ?= /usr/local/cuda
MODE_LIBS :=

ifeq ($(DATASET)$(ONET),11)
  $(error DATASET=1 and ONET=1 cannot be used together)
endif
ifeq ($(CPU)$(GPU),11)
  $(error CPU=1 and GPU=1 cannot be used together)
endif

# solver selection: ALL=1 -> PPE and VIS; nothing selected -> PPE (backward compatible)
ifeq ($(ALL),1)
  override PPE := 1
  override VIS := 1
endif
ifeq ($(PPE)$(VIS),00)
  override PPE := 1
endif
ifeq ($(PPE)$(VIS),11)
  MODE_SFX := _all
else ifeq ($(VIS),1)
  MODE_SFX := _vis
else
  MODE_SFX :=
endif

ifeq ($(DIM),2)
  MODE_DFLAGS := -D TWO_DIMENSIONAL
  MODE_DIMSFX :=
else
  MODE_DFLAGS := -D THREE_DIMENSIONAL
  MODE_DIMSFX := 3d
endif

ifeq ($(GPU),1)
  MODE_CXX    := $(NVCXX)
  MODE_FLAGS  := -O3 -acc -gpu=$(GPU_ARCH) -Minfo=accel
  MODE_BASE   := SPH_acc$(MODE_DIMSFX)
endif
ifeq ($(CPU),1)
  MODE_CXX    := g++
  MODE_FLAGS  := -O3 -fopenmp
  MODE_BASE   := SPH_omp$(MODE_DIMSFX)
endif

ifeq ($(DATASET),1)
  ifeq ($(PPE),1)
    MODE_DFLAGS += -DPPE_DON_DATASET -DPPE_DON_DATASET_EVERY=$(EVERY)
  endif
  ifeq ($(VIS),1)
    MODE_DFLAGS += -DVIS_DON_DATASET -DVIS_DON_DATASET_EVERY=$(VIS_EVERY)
  endif
  MODE_DFLAGS += -DPPE_DON_DATA_ROOT='"$(DATA_ROOT)"'
  MODE_TARGET := $(MODE_BASE)_dataset$(MODE_SFX)
else ifeq ($(ONET),1)
  ifeq ($(PPE),1)
    MODE_DFLAGS += -DPPE_DON_ONET -DPPE_DON_MODEL_DEFAULT='"$(MODEL)"'
  endif
  ifeq ($(VIS),1)
    MODE_DFLAGS += -DVIS_DON_ONET -DVIS_DON_MODEL_DEFAULT='"$(VIS_MODEL)"'
  endif
  MODE_TARGET := $(MODE_BASE)_onet$(MODE_SFX)
else
  MODE_TARGET := $(MODE_BASE)
endif
# MPS=1 : legacy MPS-type pressure operators (default build: ISPH / SPH kernel-gradient operators)
MPS      ?= 0
ifeq ($(MPS),1)
  MODE_DFLAGS += -DLEGACY_MPS_OPERATORS
  MODE_TARGET := $(MODE_TARGET)_mps
endif
ifeq ($(POU),1)
  MODE_DFLAGS += -DVIS_POU
  ifeq ($(GPU),1)
    MODE_LIBS += -cudalib=cusolver
  endif
  MODE_TARGET := $(MODE_TARGET)_pou
endif
ifeq ($(AMGX),1)
  MODE_DFLAGS += -DUSE_AMGX -I$(AMGX_DIR)/include
  MODE_FLAGS  += -cuda
  MODE_LIBS   += -L$(AMGX_DIR)/lib -lamgxsh -Wl,-rpath,$(AMGX_DIR)/lib $(CUDA_DIR)/lib64/libnvJitLink.so.12 -Wl,-rpath,$(CUDA_DIR)/lib64
  MODE_TARGET := $(MODE_TARGET)_amgx
endif

ifeq ($(SPRAY),1)
  MODE_DFLAGS += -DSPH_FVM_AIR -std=c++17
  MODE_TARGET := $(MODE_TARGET)_spray
endif

ifneq ($(CPU)$(GPU),00)
.DEFAULT_GOAL := mode_build
endif

.PHONY: mode_build train train_ppe train_vis
MODULE_INC := $(wildcard fvm/*.inc spray/*.inc deeponet/*.inc amgx/*.inc body/*.inc)

mode_build: main_implicit.cpp $(MODULE_INC) errorfunc.cpp log.cpp errorfunc.h log.h
	$(MODE_CXX) $(MODE_FLAGS) $(MODE_DFLAGS) -o $(MODE_TARGET) main_implicit.cpp errorfunc.cpp log.cpp -lm $(MODE_LIBS)
	@echo "built $(MODE_TARGET)"
ifeq ($(ONET),1)
ifeq ($(PPE),1)
	@test -f "$(MODEL)" || echo "WARNING: PPE model $(MODEL) not found (make train PPE=1, or pass MODEL=...)"
endif
ifeq ($(VIS),1)
	@test -f "$(VIS_MODEL)" || echo "WARNING: viscosity model $(VIS_MODEL) not found (make train VIS=1, or pass VIS_MODEL=...)"
endif
endif

TRAIN_DATA = $(or $(DATA),$(wildcard $(DATA_ROOT)/*/))
TRAIN_GOALS :=
ifeq ($(PPE),1)
  TRAIN_GOALS += train_ppe
endif
ifeq ($(VIS),1)
  TRAIN_GOALS += train_vis
endif
train: $(TRAIN_GOALS)

train_ppe:
	@test -n "$(TRAIN_DATA)" || (echo 'no snapshots under $(DATA_ROOT) (or pass DATA="dir1 dir2 ...")'; exit 1)
	$(PYTHON) deeponet/ppe_deeponet/train_ppe_don.py --data $(TRAIN_DATA) --out deeponet/ppe_deeponet/trained_model --epochs $(EPOCHS) --val-fraction $(VAL_FRACTION) --fourier-max 10

# directories without vis_*.bin (PPE-only runs) are skipped by the trainer
train_vis:
	@test -n "$(TRAIN_DATA)" || (echo 'no snapshots under $(DATA_ROOT) (or pass DATA="dir1 dir2 ...")'; exit 1)
	$(PYTHON) deeponet/vis_deeponet/train_vis_don.py --data $(TRAIN_DATA) --out deeponet/vis_deeponet/trained_model --epochs $(VIS_EPOCHS) --val-fraction $(VAL_FRACTION)

# ============================================================
# Classic targets
# ============================================================
TARGET          = SPHExplicit
TARGET_3D       = SPHExplicit_3d
TARGET_GCC      = SPH_gcc
TARGET_GCC3D    = SPH_gcc3d
TARGET_ICC      = SPH_icc
TARGET_ICC3D    = SPH3D_icc3d
TARGET_ACC      = SPH_acc
TARGET_ACC3D    = SPH_acc3d
TARGET_CUDA     = SPH_cuda
TARGET_CUDA3D   = SPH_cuda3d
TARGET_MPIACC   = SPH_mpiacc
TARGET_MPIACC3D = SPH_mpiacc3d

SRC = main_implicit.cpp errorfunc.cpp log.cpp

DFLAGS = -D TWO_DIMENSIONAL

LDFLAGS = -lm


###### For single CPU calculation compiled with g++ ######
CC = g++
OBJE = $(patsubst %.cpp,%.o,$(SRC))
OBJE_3D = $(patsubst %.cpp,%_3d.o,$(SRC))

CFLAGS  = -O3

$(TARGET): $(OBJE)
	$(CC) $(CFLAGS) $(DFLAGS) -o $@ $(OBJE) $(LDFLAGS)

.cpp.o:
	$(CC) $(CFLAGS) $(DFLAGS) -c $<

$(TARGET_3D): $(OBJE_3D)
	$(CC) $(CFLAGS) -o $@ $(OBJE_3D) $(LDFLAGS)

%_3d.o : %.cpp
	$(CC) $(CFLAGS) -o $@ -c $<


###### For OpenMP calculation compiled with g++ ######
GCC = g++
OBJE_GCC   = $(patsubst %.cpp,%_gcc.o,$(SRC))
OBJE_GCC3D = $(patsubst %.cpp,%_gcc3d.o,$(SRC))
CFLAGS_GCC = -O3 -fopenmp

$(TARGET_GCC): $(OBJE_GCC)
	$(GCC) $(CFLAGS_GCC) $(DFLAGS) -o $@ $(OBJE_GCC) $(LDFLAGS)

%_gcc.o : %.cpp
	$(GCC) $(CFLAGS_GCC) $(DFLAGS) -o $@ -c $<

$(TARGET_GCC3D): $(OBJE_GCC3D)
	$(GCC) $(CFLAGS_GCC) -o $@ $(OBJE_GCC3D) $(LDFLAGS)

%_gcc3d.o : %.cpp
	$(GCC) $(CFLAGS_GCC) -o $@ -c $<


###### For OpenMP calculation compiled with icpx ######
ICC = icpx
OBJE_ICC   = $(patsubst %.cpp,%_icc.o,$(SRC))
OBJE_ICC3D = $(patsubst %.cpp,%_icc3d.o,$(SRC))
CFLAGS_ICC = -O3 -fiopenmp

$(TARGET_ICC): $(OBJE_ICC)
	$(ICC) $(CFLAGS_ICC) $(DFLAGS) -o $@ $(OBJE_ICC) $(LDFLAGS)

%_icc.o : %.cpp
	$(ICC) $(CFLAGS_ICC) $(DFLAGS) -o $@ -c $<

$(TARGET_ICC3D): $(OBJE_ICC3D)
	$(ICC) $(CFLAGS_ICC) -o $@ $(OBJE_ICC3D) $(LDFLAGS)

%_icc3d.o : %.cpp
	$(ICC) $(CFLAGS_ICC) -o $@ -c $<


###### For OpenACC calculation compiled with nvc++ ######
ACC = nvc++
OBJE_ACC   = $(patsubst %.cpp,%_acc.o,$(SRC))
OBJE_ACC3D = $(patsubst %.cpp,%_acc3d.o,$(SRC))
CFLAGS_ACC = -O3 -acc -Minfo=accel

$(TARGET_ACC): $(OBJE_ACC)
	$(ACC) $(CFLAGS_ACC) $(DFLAGS) -o $@ $(OBJE_ACC) $(LDFLAGS)

%_acc.o : %.cpp
	$(ACC) $(CFLAGS_ACC) $(DFLAGS) -o $@ -c $<

$(TARGET_ACC3D): $(OBJE_ACC3D)
	$(ACC) $(CFLAGS_ACC) -o $@ $(OBJE_ACC3D) $(LDFLAGS)

%_acc3d.o : %.cpp
	$(ACC) $(CFLAGS_ACC) -o $@ -c $<


###### For OpenACC+CUDA calculation compiled with nvc++ ######
CUDA = nvc++
OBJE_CUDA    = $(patsubst %.cpp,%_cuda.o,$(SRC))
OBJE_CUDA3D  = $(patsubst %.cpp,%_cuda3d.o,$(SRC))
CFLAGS_CUDA  = -O3 -acc -Minfo=accel -gpu=cc70,cc80 -Mcuda
CFLAGS_CUDA3D = $(CFLAGS_CUDA)
LDFLAGS_CUDA = -lm -lcublas -lcusparse

$(TARGET_CUDA): $(OBJE_CUDA)
	$(CUDA) $(CFLAGS_CUDA) $(DFLAGS) -o $@ $(OBJE_CUDA) $(LDFLAGS_CUDA)

%_cuda.o : %.cpp
	$(CUDA) $(CFLAGS_CUDA) $(DFLAGS) -o $@ -c $<

$(TARGET_CUDA3D): $(OBJE_CUDA3D)
	$(CUDA) $(CFLAGS_CUDA3D) -o $@ $(OBJE_CUDA3D) $(LDFLAGS_CUDA)

%_cuda3d.o : %.cpp
	$(CUDA) $(CFLAGS_CUDA3D) -o $@ -c $<


###### For MPI + OpenACC calculation compiled with nvc++ ######
MPIACC = mpicxx
OBJE_MPIACC   = $(patsubst %.cpp,%_mpiacc.o,$(SRC))
OBJE_MPIACC3D = $(patsubst %.cpp,%_mpiacc3d.o,$(SRC))
CFLAGS_MPIACC = -O3 -acc -Minfo=accel -gpu=cc89
CFLAGS_MPIACC3D = $(CFLAGS_MPIACC)

$(TARGET_MPIACC): $(OBJE_MPIACC)
	$(MPIACC) $(CFLAGS_MPIACC) $(DFLAGS) -o $@ $(OBJE_MPIACC) $(LDFLAGS)

%_mpiacc.o : %.cpp
	$(MPIACC) $(CFLAGS_MPIACC) $(DFLAGS) -o $@ -c $<

$(TARGET_MPIACC3D): $(OBJE_MPIACC3D)
	$(MPIACC) $(CFLAGS_MPIACC3D) -o $@ $(OBJE_MPIACC3D) $(LDFLAGS)

%_mpiacc3d.o : %.cpp
	$(MPIACC) $(CFLAGS_MPIACC3D) -o $@ -c $<


clean:
	/bin/rm -f *.o *~ SPH_acc* SPH_omp* \
	$(foreach b,SPH_acc SPH_omp SPH_acc3d SPH_omp3d,$(foreach m,dataset onet,$(b)_$(m) $(b)_$(m)_vis $(b)_$(m)_all)) \
	$(TARGET) $(TARGET_3D) \
	$(TARGET_GCC) $(TARGET_GCC3D) \
	$(TARGET_ICC) $(TARGET_ICC3D) \
	$(TARGET_ACC) $(TARGET_ACC3D) \
	$(TARGET_CUDA) $(TARGET_CUDA3D) \
	$(TARGET_MPIACC) $(TARGET_MPIACC3D)


errorfunc.o : errorfunc.h
log.o       : errorfunc.h log.h

# DeepONet PPE / viscosity preconditioners are #included by main_implicit.cpp
main_implicit.o main_implicit_3d.o main_implicit_gcc.o main_implicit_gcc3d.o main_implicit_icc.o main_implicit_icc3d.o main_implicit_acc.o main_implicit_acc3d.o main_implicit_cuda.o main_implicit_cuda3d.o main_implicit_mpiacc.o main_implicit_mpiacc3d.o : $(MODULE_INC)
