/* oracle_1100.c — golden-reference harness for the DALEC_1100 JAX port.
 *
 * Compiled with the SAME flags as the production executables (plain gcc,
 * no -O flag: see BASH/CARDAMOM_COMPILE.sh) so the reference numerics are
 * identical to what CARDAMOM_RUN_MODEL.exe / CARDAMOM_MDF.exe compute.
 *
 * Subcommands
 *   manifest
 *       Print a JSON description of every exposed module: input/output field
 *       names in C-struct declaration order. The fixture generator and the
 *       pytest loader both consume this — field order is never hand-coded
 *       on the Python side.
 *   module <NAME> <in.bin> <out.bin>
 *       Batch-evaluate one leaf module. in.bin is n_cases x n_in doubles
 *       (raw little-endian, row-major; n_cases inferred from file size);
 *       out.bin is written as n_cases x n_out doubles.
 *   trajectory <cbf.nc> <params.bin> <pools.bin> <fluxes.bin>
 *       Call DALEC_1100 DIRECTLY (DALEC->dalec) for each parameter vector
 *       (params.bin = n x nopars doubles) and append raw trajectories:
 *       pools.bin gains (Ntimesteps+1) x nopools doubles per sample,
 *       fluxes.bin gains Ntimesteps x nofluxes doubles per sample.
 *       NOTE: this bypasses DALEC_MLF2 on purpose — MLF2 skips the model for
 *       prerun-EDC-failing samples and CARDAMOM_RUN_MODEL.exe then writes the
 *       PREVIOUS sample's trajectory (stale-output bug). Never use RUN_MODEL
 *       output as a reference.
 *   mlf <cbf.nc> <params.bin> <out.bin>
 *       Full DALEC_MLF2 path per sample; out.bin gains
 *       [noedcs M_EDCs | nolikelihoods M_LIKELIHOODS | P] doubles per sample.
 *
 * Buffer policy (BUG_COMPAT: per_sample_buffer_zeroing): the production
 * drivers calloc M_POOLS/M_FLUXES once per PROCESS, so a sample whose time
 * loop breaks early (isfinite check, DALEC_1100.c:1137-1141) inherits the
 * previous sample's values in its tail — i.e. in-process C output is sample-
 * order dependent. This harness zeroes the model buffers before EVERY sample,
 * which reproduces first-sample-of-process semantics deterministically. The
 * JAX port freezes to zero after a break, matching this canonical form.
 */
#include "../../auxi_fun/okcheck.c"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../CARDAMOM_GENERAL/CARDAMOM_READ_BINARY_DATA.c"
#include <netcdf.h>

/* ------------------------------------------------------------------ */
/* Module registry                                                     */
/* ------------------------------------------------------------------ */

typedef void (*ADAPTER)(const double *in, double *out);

typedef struct {
    const char *name;
    int n_in;
    int n_out;
    const char **in_names;
    const char **out_names;
    ADAPTER run;
} MODULE_SPEC;

/* --- scalar-function modules ------------------------------------- */

static const char *IN_EWT2MOI[] = {"ewt", "p", "z"};
static const char *OUT_EWT2MOI[] = {"moi"};
static void run_EWT2MOI(const double *in, double *out)
{ out[0] = HYDROFUN_EWT2MOI(in[0], in[1], in[2]); }

static const char *IN_MOI2EWT[] = {"moi", "p", "z"};
static const char *OUT_MOI2EWT[] = {"ewt"};
static void run_MOI2EWT(const double *in, double *out)
{ out[0] = HYDROFUN_MOI2EWT(in[0], in[1], in[2]); }

static const char *IN_MOI2CON[] = {"moi", "k0", "b"};
static const char *OUT_MOI2CON[] = {"con"};
static void run_MOI2CON(const double *in, double *out)
{ out[0] = HYDROFUN_MOI2CON(in[0], in[1], in[2]); }

static const char *IN_MOI2PSI[] = {"moi", "psi_porosity", "b"};
static const char *OUT_MOI2PSI[] = {"psi"};
static void run_MOI2PSI(const double *in, double *out)
{ out[0] = HYDROFUN_MOI2PSI(in[0], in[1], in[2]); }

static const char *IN_PSI2MOI[] = {"psi", "psi_porosity", "b"};
static const char *OUT_PSI2MOI[] = {"moi"};
static void run_PSI2MOI(const double *in, double *out)
{ out[0] = HYDROFUN_PSI2MOI(in[0], in[1], in[2]); }

static const char *IN_DRAINAGE[] = {"sm", "Qexcess", "psi_field", "psi_porosity", "b"};
static const char *OUT_DRAINAGE[] = {"drainage"};
static void run_DRAINAGE(const double *in, double *out)
{ out[0] = DRAINAGE(in[0], in[1], in[2], in[3], in[4]); }

static const char *IN_IEPLM[] = {"TEMP"};
static const char *OUT_IEPLM[] = {"U"};
static void run_IEPLM(const double *in, double *out)
{ out[0] = INTERNAL_ENERGY_PER_LIQUID_H2O_UNIT_MASS(in[0]); }

static const char *IN_INITSOILE[] = {"internal_energy_per_mm_H2O", "H2O_mm",
                                     "dry_soil_vol_heat_capacity", "depth"};
static const char *OUT_INITSOILE[] = {"TOTAL_E"};
static void run_INITSOILE(const double *in, double *out)
{ out[0] = INITIALIZE_INTERNAL_SOIL_ENERGY(in[0], in[1], in[2], in[3]); }

static const char *IN_MINQUAD[] = {"x", "y", "eta"};
static const char *OUT_MINQUAD[] = {"mins"};
static void run_MINQUAD(const double *in, double *out)
{ out[0] = MinQuadraticSmooth(in[0], in[1], in[2]); }

static const char *IN_MAXEXP[] = {"x", "y", "x0"};
static const char *OUT_MAXEXP[] = {"maxx"};
static void run_MAXEXP(const double *in, double *out)
{ out[0] = MaxExponentialSmooth(in[0], in[1], in[2]); }

static const char *IN_DAYL[] = {"latitude", "DOY"};
static const char *OUT_DAYL[] = {"dayl"};
static void run_DAYL(const double *in, double *out)
{
    double pars[3];
    pars[0] = in[0]; pars[1] = in[1]; pars[2] = DGCM_PI;
    out[0] = ComputeDaylightHours(pars);
}

/* --- .IN/.OUT struct modules (field order = declaration order) ---- */

static const char *IN_SOILTEMP[] = {"dry_soil_vol_heat_capacity", "depth",
                                    "soil_water", "internal_energy"};
static const char *OUT_SOILTEMP[] = {"TEMP", "LF"};
static void run_SOILTEMP(const double *in, double *out)
{
    SOIL_TEMP_AND_LIQUID_FRAC_STRUCT S;
    memset(&S, 0, sizeof(S));
    S.IN.dry_soil_vol_heat_capacity = in[0];
    S.IN.depth = in[1];
    S.IN.soil_water = in[2];
    S.IN.internal_energy = in[3];
    SOIL_TEMP_AND_LIQUID_FRAC(&S);
    out[0] = S.OUT.TEMP;
    out[1] = S.OUT.LF;
}

static const char *IN_HETRESP[] = {"TEMP", "SM", "LF", "S_FV", "SM_OPT",
                                   "FWC", "R_CH4", "Q10CH4", "Q10CO2"};
static const char *OUT_HETRESP[] = {"aerobic_tr", "anaerobic_tr",
                                    "anaerobic_co2_c_ratio",
                                    "anaerobic_ch4_c_ratio", "fT", "fV", "fW"};
static void run_HETRESP(const double *in, double *out)
{
    HET_RESP_RATES_JCR_STRUCT S;
    memset(&S, 0, sizeof(S));
    S.IN.TEMP = in[0]; S.IN.SM = in[1]; S.IN.LF = in[2];
    S.IN.S_FV = in[3]; S.IN.SM_OPT = in[4]; S.IN.FWC = in[5];
    S.IN.R_CH4 = in[6]; S.IN.Q10CH4 = in[7]; S.IN.Q10CO2 = in[8];
    HET_RESP_RATES_JCR(&S);
    out[0] = S.OUT.aerobic_tr;
    out[1] = S.OUT.anaerobic_tr;
    out[2] = S.OUT.anaerobic_co2_c_ratio;
    out[3] = S.OUT.anaerobic_ch4_c_ratio;
    out[4] = S.OUT.fT;
    out[5] = S.OUT.fV;
    out[6] = S.OUT.fW;
}

static const char *IN_KNORR[] = {"temp", "deltat", "n", "latitude", "DOY",
                                 "lambda", "lambda_max", "T_phi", "T_r",
                                 "plgr", "k_L", "pasm", "transp", "tau_W",
                                 "t_c", "t_r", "T_memory", "lambda_max_memory"};
static const char *OUT_KNORR[] = {"lambda_next", "T", "laim", "dlambdadt",
                                  "f_T", "f_d", "lambda_tilde_max", "lambda_W"};
static void run_KNORR(const double *in, double *out)
{
    KNORR_ALLOCATION_STRUCT K;
    memset(&K, 0, sizeof(K));
    K.IN.temp = in[0]; K.IN.deltat = in[1]; K.IN.n = in[2];
    K.IN.latitude = in[3]; K.IN.DOY = in[4]; K.IN.lambda = in[5];
    K.IN.lambda_max = in[6]; K.IN.T_phi = in[7]; K.IN.T_r = in[8];
    K.IN.plgr = in[9]; K.IN.k_L = in[10]; K.IN.pasm = in[11];
    K.IN.transp = in[12]; K.IN.tau_W = in[13]; K.IN.t_c = in[14];
    K.IN.t_r = in[15]; K.IN.T_memory = in[16]; K.IN.lambda_max_memory = in[17];
    KNORR_ALLOCATION(&K);
    out[0] = K.OUT.lambda_next; out[1] = K.OUT.T; out[2] = K.OUT.laim;
    out[3] = K.OUT.dlambdadt; out[4] = K.OUT.f_T; out[5] = K.OUT.f_d;
    out[6] = K.OUT.lambda_tilde_max; out[7] = K.OUT.lambda_W;
}

static const char *IN_ALLOC[] = {"deltat", "TEMP", "C_LIVE_W", "C_LIVE_R",
                                 "NSC", "GPP", "Rd", "mr_r", "mr_w", "gr",
                                 "Q10mr", "ALLOC_FOL_POT", "ALLOC_WOO_POT",
                                 "ALLOC_ROO_POT"};
static const char *OUT_ALLOC[] = {"F_LABPROD", "F_LABREL_ACTUAL",
                                  "AUTO_RESP_MAINTENANCE", "AUTO_RESP_GROWTH",
                                  "ALLOC_FOL_ACTUAL", "ALLOC_WOO_ACTUAL",
                                  "ALLOC_ROO_ACTUAL", "AUTO_RESP_TOTAL",
                                  "NPP", "CUE", "NONLEAF_MORTALITY_FACTOR"};
static void run_ALLOC(const double *in, double *out)
{
    ALLOC_AND_AUTO_RESP_FLUXES_STRUCT S;
    memset(&S, 0, sizeof(S));
    S.IN.deltat = in[0]; S.IN.TEMP = in[1]; S.IN.C_LIVE_W = in[2];
    S.IN.C_LIVE_R = in[3]; S.IN.NSC = in[4]; S.IN.GPP = in[5];
    S.IN.Rd = in[6]; S.IN.mr_r = in[7]; S.IN.mr_w = in[8];
    S.IN.gr = in[9]; S.IN.Q10mr = in[10]; S.IN.ALLOC_FOL_POT = in[11];
    S.IN.ALLOC_WOO_POT = in[12]; S.IN.ALLOC_ROO_POT = in[13];
    ALLOC_AND_AUTO_RESP_FLUXES(&S);
    out[0] = S.OUT.F_LABPROD; out[1] = S.OUT.F_LABREL_ACTUAL;
    out[2] = S.OUT.AUTO_RESP_MAINTENANCE; out[3] = S.OUT.AUTO_RESP_GROWTH;
    out[4] = S.OUT.ALLOC_FOL_ACTUAL; out[5] = S.OUT.ALLOC_WOO_ACTUAL;
    out[6] = S.OUT.ALLOC_ROO_ACTUAL; out[7] = S.OUT.AUTO_RESP_TOTAL;
    out[8] = S.OUT.NPP; out[9] = S.OUT.CUE;
    out[10] = S.OUT.NONLEAF_MORTALITY_FACTOR;
}

static const char *IN_LIU[] = {"SRAD", "VPD", "TEMP", "vcmax25", "co2",
                               "beta_factor", "g1", "LAI", "ga", "VegK",
                               "Tupp", "Tdown", "C3_frac", "clumping",
                               "leaf_refl_par", "leaf_refl_nir", "maxPevap",
                               "precip", "q10canopy", "q10canopyRd",
                               "canopyRdsf", "NSC", "deltat"};
static const char *OUT_LIU[] = {"An", "Ag", "Rd", "transp", "evap",
                                "LEAF_MORTALITY_FACTOR"};
static void run_LIU(const double *in, double *out)
{
    LIU_AN_ET_STRUCT A;
    memset(&A, 0, sizeof(A));
    A.IN.SRAD = in[0]; A.IN.VPD = in[1]; A.IN.TEMP = in[2];
    A.IN.vcmax25 = in[3]; A.IN.co2 = in[4]; A.IN.beta_factor = in[5];
    A.IN.g1 = in[6]; A.IN.LAI = in[7]; A.IN.ga = in[8];
    A.IN.VegK = in[9]; A.IN.Tupp = in[10]; A.IN.Tdown = in[11];
    A.IN.C3_frac = in[12]; A.IN.clumping = in[13];
    A.IN.leaf_refl_par = in[14]; A.IN.leaf_refl_nir = in[15];
    A.IN.maxPevap = in[16]; A.IN.precip = in[17];
    A.IN.q10canopy = in[18]; A.IN.q10canopyRd = in[19];
    A.IN.canopyRdsf = in[20]; A.IN.NSC = in[21]; A.IN.deltat = in[22];
    LIU_AN_ET(&A);
    out[0] = A.OUT.An; out[1] = A.OUT.Ag; out[2] = A.OUT.Rd;
    out[3] = A.OUT.transp; out[4] = A.OUT.evap;
    out[5] = A.OUT.LEAF_MORTALITY_FACTOR;
}

#define NMODULES 16
static const MODULE_SPEC MODULES[NMODULES] = {
    {"HYDROFUN_EWT2MOI", 3, 1, IN_EWT2MOI, OUT_EWT2MOI, run_EWT2MOI},
    {"HYDROFUN_MOI2EWT", 3, 1, IN_MOI2EWT, OUT_MOI2EWT, run_MOI2EWT},
    {"HYDROFUN_MOI2CON", 3, 1, IN_MOI2CON, OUT_MOI2CON, run_MOI2CON},
    {"HYDROFUN_MOI2PSI", 3, 1, IN_MOI2PSI, OUT_MOI2PSI, run_MOI2PSI},
    {"HYDROFUN_PSI2MOI", 3, 1, IN_PSI2MOI, OUT_PSI2MOI, run_PSI2MOI},
    {"DRAINAGE", 5, 1, IN_DRAINAGE, OUT_DRAINAGE, run_DRAINAGE},
    {"INTERNAL_ENERGY_PER_LIQUID_H2O_UNIT_MASS", 1, 1, IN_IEPLM, OUT_IEPLM, run_IEPLM},
    {"INITIALIZE_INTERNAL_SOIL_ENERGY", 4, 1, IN_INITSOILE, OUT_INITSOILE, run_INITSOILE},
    {"MIN_QUADRATIC_SMOOTH", 3, 1, IN_MINQUAD, OUT_MINQUAD, run_MINQUAD},
    {"MAX_EXPONENTIAL_SMOOTH", 3, 1, IN_MAXEXP, OUT_MAXEXP, run_MAXEXP},
    {"COMPUTE_DAYLIGHT_HOURS", 2, 1, IN_DAYL, OUT_DAYL, run_DAYL},
    {"SOIL_TEMP_AND_LIQUID_FRAC", 4, 2, IN_SOILTEMP, OUT_SOILTEMP, run_SOILTEMP},
    {"HET_RESP_RATES_JCR", 9, 7, IN_HETRESP, OUT_HETRESP, run_HETRESP},
    {"KNORR_ALLOCATION", 18, 8, IN_KNORR, OUT_KNORR, run_KNORR},
    {"ALLOC_AND_AUTO_RESP_FLUXES", 14, 11, IN_ALLOC, OUT_ALLOC, run_ALLOC},
    {"LIU_AN_ET", 23, 6, IN_LIU, OUT_LIU, run_LIU},
};

/* ------------------------------------------------------------------ */
/* Subcommand implementations                                          */
/* ------------------------------------------------------------------ */

static int cmd_manifest(void)
{
    int m, i;
    printf("{\n  \"format\": \"raw little-endian float64, row-major, "
           "n_cases x n_fields\",\n");
    printf("  \"constants\": {\"DGCM_PI\": %.17g, \"DGCM_T3\": %.17g, "
           "\"DGCM_TK0C\": %.17g, \"DGCM_SPECIFIC_HEAT_ICE\": %.17g, "
           "\"DGCM_SPECIFIC_HEAT_WATER\": %.17g, "
           "\"DGCM_LATENT_HEAT_FUSION_3\": %.17g, "
           "\"DGCM_T_LIQUID_H2O_ZERO_ENERGY\": %.17g, "
           "\"DGCM_SEC_DAY\": %.17g},\n",
           (double)DGCM_PI, (double)DGCM_T3, (double)DGCM_TK0C,
           (double)DGCM_SPECIFIC_HEAT_ICE, (double)DGCM_SPECIFIC_HEAT_WATER,
           (double)DGCM_LATENT_HEAT_FUSION_3,
           (double)DGCM_T_LIQUID_H2O_ZERO_ENERGY, (double)DGCM_SEC_DAY);
    printf("  \"modules\": [\n");
    for (m = 0; m < NMODULES; m++) {
        printf("    {\"name\": \"%s\", \"inputs\": [", MODULES[m].name);
        for (i = 0; i < MODULES[m].n_in; i++)
            printf("%s\"%s\"", i ? ", " : "", MODULES[m].in_names[i]);
        printf("], \"outputs\": [");
        for (i = 0; i < MODULES[m].n_out; i++)
            printf("%s\"%s\"", i ? ", " : "", MODULES[m].out_names[i]);
        printf("]}%s\n", m < NMODULES - 1 ? "," : "");
    }
    printf("  ]\n}\n");
    return 0;
}

static long file_size(FILE *f)
{
    long sz;
    fseek(f, 0L, SEEK_END);
    sz = ftell(f);
    fseek(f, 0L, SEEK_SET);
    return sz;
}

static int cmd_module(const char *name, const char *infile, const char *outfile)
{
    const MODULE_SPEC *spec = NULL;
    int m;
    for (m = 0; m < NMODULES; m++)
        if (strcmp(MODULES[m].name, name) == 0) { spec = &MODULES[m]; break; }
    if (!spec) {
        fprintf(stderr, "oracle_1100: unknown module '%s' (run 'manifest')\n", name);
        return 2;
    }

    FILE *fi = fopen(infile, "rb");
    if (!fi) { fprintf(stderr, "oracle_1100: cannot open %s\n", infile); return 2; }
    long sz = file_size(fi);
    long rowbytes = (long)spec->n_in * (long)sizeof(double);
    if (sz % rowbytes != 0) {
        fprintf(stderr, "oracle_1100: %s size %ld not divisible by %d doubles\n",
                infile, sz, spec->n_in);
        fclose(fi);
        return 2;
    }
    long n_cases = sz / rowbytes;

    double *in = malloc(rowbytes);
    double *out = malloc((size_t)spec->n_out * sizeof(double));
    FILE *fo = fopen(outfile, "wb");
    if (!fo) { fprintf(stderr, "oracle_1100: cannot open %s\n", outfile); return 2; }

    long c;
    for (c = 0; c < n_cases; c++) {
        if (fread(in, sizeof(double), spec->n_in, fi) != (size_t)spec->n_in) {
            fprintf(stderr, "oracle_1100: short read at case %ld\n", c);
            return 2;
        }
        memset(out, 0, (size_t)spec->n_out * sizeof(double));
        spec->run(in, out);
        fwrite(out, sizeof(double), spec->n_out, fo);
    }
    fclose(fi);
    fclose(fo);
    free(in);
    free(out);
    fprintf(stderr, "oracle_1100: %s evaluated %ld cases\n", spec->name, n_cases);
    return 0;
}

/* Zero every per-sample model buffer (see buffer policy in the header). */
static void zero_model_buffers(DATA *D)
{
    int T = D->ncdf_data.Ntimesteps;
    memset(D->M_FLUXES, 0, (size_t)T * D->nofluxes * sizeof(double));
    memset(D->M_POOLS, 0, (size_t)(T + 1) * D->nopools * sizeof(double));
    memset(D->M_EDCs, 0, (size_t)D->noedcs * sizeof(double));
    memset(D->M_LIKELIHOODS, 0, (size_t)D->nolikelihoods * sizeof(double));
    D->M_P[0] = 0;
    memset(D->M_ABGB, 0, (size_t)T * sizeof(double));
    memset(D->M_CH4, 0, (size_t)T * sizeof(double));
    memset(D->M_CWOO, 0, (size_t)T * sizeof(double));
    memset(D->M_DOM, 0, (size_t)T * sizeof(double));
    memset(D->M_ET, 0, (size_t)T * sizeof(double));
    memset(D->M_LE, 0, (size_t)T * sizeof(double));
    memset(D->M_H, 0, (size_t)T * sizeof(double));
    memset(D->M_EWT, 0, (size_t)T * sizeof(double));
    memset(D->M_GPP, 0, (size_t)T * sizeof(double));
    memset(D->M_SIF, 0, (size_t)T * sizeof(double));
    memset(D->M_LAI, 0, (size_t)T * sizeof(double));
    memset(D->M_FIR, 0, (size_t)T * sizeof(double));
    memset(D->M_NBE, 0, (size_t)T * sizeof(double));
    memset(D->M_ROFF, 0, (size_t)T * sizeof(double));
    memset(D->M_SCF, 0, (size_t)T * sizeof(double));
    memset(D->M_SWE, 0, (size_t)T * sizeof(double));
}

static double *read_params(const char *paramfile, int nopars, long *n_samples)
{
    FILE *fp = fopen(paramfile, "rb");
    if (!fp) { fprintf(stderr, "oracle_1100: cannot open %s\n", paramfile); exit(2); }
    long sz = file_size(fp);
    long rowbytes = (long)nopars * (long)sizeof(double);
    if (sz % rowbytes != 0) {
        fprintf(stderr, "oracle_1100: %s size %ld not divisible by %d doubles\n",
                paramfile, sz, nopars);
        exit(2);
    }
    *n_samples = sz / rowbytes;
    double *pars = malloc((size_t)sz);
    if (fread(pars, 1, (size_t)sz, fp) != (size_t)sz) {
        fprintf(stderr, "oracle_1100: short read on %s\n", paramfile);
        exit(2);
    }
    fclose(fp);
    return pars;
}

static int cmd_trajectory(const char *cbf, const char *paramfile,
                          const char *poolsfile, const char *fluxesfile)
{
    DATA D;
    CARDAMOM_READ_BINARY_DATA((char *)cbf, &D);
    DALEC *MODEL = (DALEC *)D.MODEL;
    int T = D.ncdf_data.Ntimesteps;

    long n_samples;
    double *pars = read_params(paramfile, D.nopars, &n_samples);

    FILE *fpo = fopen(poolsfile, "wb");
    FILE *ffl = fopen(fluxesfile, "wb");
    if (!fpo || !ffl) { fprintf(stderr, "oracle_1100: cannot open output\n"); return 2; }

    long n;
    for (n = 0; n < n_samples; n++) {
        zero_model_buffers(&D);
        MODEL->dalec(D, pars + (size_t)n * D.nopars);
        fwrite(D.M_POOLS, sizeof(double), (size_t)(T + 1) * D.nopools, fpo);
        fwrite(D.M_FLUXES, sizeof(double), (size_t)T * D.nofluxes, ffl);
    }
    fclose(fpo);
    fclose(ffl);
    fprintf(stderr,
            "oracle_1100: trajectory ran %ld samples (T=%d, nopools=%d, nofluxes=%d)\n",
            n_samples, T, D.nopools, D.nofluxes);
    return 0;
}

static int cmd_mlf(const char *cbf, const char *paramfile, const char *outfile)
{
    DATA D;
    CARDAMOM_READ_BINARY_DATA((char *)cbf, &D);

    long n_samples;
    double *pars = read_params(paramfile, D.nopars, &n_samples);

    FILE *fo = fopen(outfile, "wb");
    if (!fo) { fprintf(stderr, "oracle_1100: cannot open %s\n", outfile); return 2; }

    long n;
    for (n = 0; n < n_samples; n++) {
        zero_model_buffers(&D);
        double P = D.MLF(D, pars + (size_t)n * D.nopars);
        fwrite(D.M_EDCs, sizeof(double), (size_t)D.noedcs, fo);
        fwrite(D.M_LIKELIHOODS, sizeof(double), (size_t)D.nolikelihoods, fo);
        fwrite(&P, sizeof(double), 1, fo);
    }
    fclose(fo);
    fprintf(stderr,
            "oracle_1100: mlf ran %ld samples (noedcs=%d, nolikelihoods=%d, "
            "row = %d doubles)\n",
            n_samples, D.noedcs, D.nolikelihoods, D.noedcs + D.nolikelihoods + 1);
    return 0;
}

/* Central finite-difference gradient of the MLF log-posterior w.r.t. all
 * parameters: the C-side baseline for "time to gradient" and an
 * independent numerical check of the JAX autodiff. Per sample this costs
 * 2*nopars MLF evaluations (each with per-sample buffer zeroing). Relative
 * step hrel applied per parameter: h = hrel*|p_k| (or hrel if p_k == 0). */
static int cmd_fdgrad(const char *cbf, const char *paramfile,
                      const char *outfile, double hrel)
{
    DATA D;
    CARDAMOM_READ_BINARY_DATA((char *)cbf, &D);

    long n_samples;
    double *pars = read_params(paramfile, D.nopars, &n_samples);
    double *work = malloc((size_t)D.nopars * sizeof(double));
    double *grad = malloc((size_t)D.nopars * sizeof(double));

    FILE *fo = fopen(outfile, "wb");
    if (!fo) { fprintf(stderr, "oracle_1100: cannot open %s\n", outfile); return 2; }

    long n;
    int k;
    for (n = 0; n < n_samples; n++) {
        const double *p0 = pars + (size_t)n * D.nopars;
        for (k = 0; k < D.nopars; k++) {
            double h = hrel * fabs(p0[k]);
            if (h == 0) h = hrel;
            memcpy(work, p0, (size_t)D.nopars * sizeof(double));
            work[k] = p0[k] + h;
            zero_model_buffers(&D);
            double Pp = D.MLF(D, work);
            work[k] = p0[k] - h;
            zero_model_buffers(&D);
            double Pm = D.MLF(D, work);
            grad[k] = (Pp - Pm) / (2 * h);
        }
        fwrite(grad, sizeof(double), (size_t)D.nopars, fo);
    }
    fclose(fo);
    fprintf(stderr, "oracle_1100: fdgrad ran %ld samples x %d params "
            "(2 MLF evals each, hrel=%g)\n", n_samples, D.nopars, hrel);
    return 0;
}

/* ------------------------------------------------------------------ */

int main(int argc, char *argv[])
{
    if (argc >= 2 && strcmp(argv[1], "manifest") == 0)
        return cmd_manifest();
    if (argc == 5 && strcmp(argv[1], "module") == 0)
        return cmd_module(argv[2], argv[3], argv[4]);
    if (argc == 6 && strcmp(argv[1], "trajectory") == 0)
        return cmd_trajectory(argv[2], argv[3], argv[4], argv[5]);
    if (argc == 5 && strcmp(argv[1], "mlf") == 0)
        return cmd_mlf(argv[2], argv[3], argv[4]);
    if ((argc == 5 || argc == 6) && strcmp(argv[1], "fdgrad") == 0)
        return cmd_fdgrad(argv[2], argv[3], argv[4],
                          argc == 6 ? atof(argv[5]) : 1e-6);

    fprintf(stderr,
        "usage:\n"
        "  oracle_1100 manifest\n"
        "  oracle_1100 module <NAME> <in.bin> <out.bin>\n"
        "  oracle_1100 trajectory <cbf.nc> <params.bin> <pools.bin> <fluxes.bin>\n"
        "  oracle_1100 mlf <cbf.nc> <params.bin> <out.bin>\n"
        "  oracle_1100 fdgrad <cbf.nc> <params.bin> <grads.bin> [hrel=1e-6]\n");
    return 1;
}
