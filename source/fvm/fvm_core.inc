// =============================================================================
// fvm_core.inc -- the VOF/FVM solver of VOF_FVM_2Phase as a library
//
// Included by main_implicit.cpp BEFORE its own macros (DIM, Mod, ...), so the
// FVM source is compiled unchanged inside namespace fvm. The standard headers
// are included here first; their include guards make the copies inside the
// namespace no-ops.
// =============================================================================
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <queue>
#include <sstream>
#include <string>
#include <vector>
#ifdef _OPENACC
#include <openacc.h>
#endif

#define VOF_NO_MAIN
namespace fvm {
#include "../../../VOF_FVM_2Phase/source/main.cpp"
}
