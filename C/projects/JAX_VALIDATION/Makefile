# Oracle harness for the DALEC_1100 JAX port.
#
# FLAG POLICY: identical to BASH/CARDAMOM_COMPILE.sh:78 — plain $(CC) with no
# optimization flag, -lm, and nc-config's libs+cflags, in that order. The
# equivalence claim is stamped against these flags; do not add -O/-ffast-math/
# -march here without regenerating every golden and rerunning the suite.

# '=' not '?=': make predefines CC=cc, so ?= would never select gcc.
# (On this host cc symlinks gcc, but the fingerprint should say gcc explicitly.)
CC = gcc
NC_CONFIG ?= $(shell which nc-config)
NETCDF_LIB_FLAGS := $(shell $(NC_CONFIG) --libs) $(shell $(NC_CONFIG) --cflags)

all: oracle_1100.exe

oracle_1100.exe: oracle_1100.c
	$(CC) oracle_1100.c -o oracle_1100.exe -lm $(NETCDF_LIB_FLAGS)

# Tap build (P3): same flags + the ORACLE_TAP macro switch.
oracle_1100_tap.exe: oracle_1100.c
	$(CC) -DCARDAMOM_ORACLE_TAP oracle_1100.c -o oracle_1100_tap.exe -lm $(NETCDF_LIB_FLAGS)

fingerprint:
	@echo "cc:      $$($(CC) --version | head -1)"
	@echo "glibc:   $$(ldd --version | head -1)"
	@echo "nc:      $$($(NC_CONFIG) --version)"
	@echo "flags:   $(CC) oracle_1100.c -o oracle_1100.exe -lm $(NETCDF_LIB_FLAGS)"
	@echo "c_tree:  $$(git -C ../../.. rev-parse HEAD 2>/dev/null || echo untracked)"

clean:
	rm -f oracle_1100.exe oracle_1100_tap.exe

.PHONY: all clean fingerprint
