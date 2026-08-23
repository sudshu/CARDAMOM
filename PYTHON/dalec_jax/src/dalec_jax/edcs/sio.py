"""Flux source/sink incidence — transcription of DALEC_1100_FLUX_SOURCES_SINKS
(DALEC_1100.c:82-236), inverted per DALEC_STATE_SOURCE_SINK_MATRIX_CONFIG.c
(ascending flux-index order, as the C builds it).
"""
from ..indices import F, S

# flux -> pool it feeds (C: FIOMATRIX.SINK[f])
SINK = {
    F.gpp: S.C_lab,
    F.foliar_prod: S.C_fol,
    F.root_prod: S.C_roo,
    F.wood_prod: S.C_woo,
    F.labyield2lit: S.C_lit, F.lab2lit: S.C_lit, F.fx_lab2lit: S.C_lit,
    F.ph_fol2lit: S.C_lit, F.folyield2lit: S.C_lit, F.fol2lit: S.C_lit,
    F.fx_fol2lit: S.C_lit, F.rooyield2lit: S.C_lit, F.roo2lit: S.C_lit,
    F.fx_roo2lit: S.C_lit,
    F.wooyield2cwd: S.C_cwd, F.woo2cwd: S.C_cwd, F.fx_woo2cwd: S.C_cwd,
    F.cwd2som: S.C_som, F.fx_cwd2som: S.C_som, F.lit2som: S.C_som,
    F.fx_lit2som: S.C_som,
    F.snowfall: S.H2O_SWE,
    F.infil: S.H2O_LY1,
    F.ly1xly2: S.H2O_LY2,
    F.ly2xly3: S.H2O_LY3,
    F.gh_in: S.E_LY1, F.infil_e: S.E_LY1,
    F.ly1xly2_e: S.E_LY2, F.ly1xly2_th_e: S.E_LY2,
    F.ly2xly3_e: S.E_LY3, F.geological: S.E_LY3, F.ly2xly3_th_e: S.E_LY3,
}

# flux -> pool it drains (C: FIOMATRIX.SOURCE[f])
SOURCE = {
    F.resp_auto_maint: S.C_lab, F.Rd: S.C_lab, F.foliar_prod: S.C_lab,
    F.root_prod: S.C_lab, F.wood_prod: S.C_lab, F.resp_auto_growth: S.C_lab,
    F.f_lab: S.C_lab, F.lab2lit: S.C_lab, F.labyield2lit: S.C_lab,
    F.fx_lab2lit: S.C_lab, F.dist_lab: S.C_lab,
    F.fol2lit: S.C_fol, F.ph_fol2lit: S.C_fol, F.folyield2lit: S.C_fol,
    F.f_fol: S.C_fol, F.fx_fol2lit: S.C_fol, F.dist_fol: S.C_fol,
    F.roo2lit: S.C_roo, F.rooyield2lit: S.C_roo, F.f_roo: S.C_roo,
    F.fx_roo2lit: S.C_roo, F.dist_roo: S.C_roo,
    F.woo2cwd: S.C_woo, F.wooyield2cwd: S.C_woo, F.f_woo: S.C_woo,
    F.fx_woo2cwd: S.C_woo, F.dist_woo: S.C_woo,
    F.ae_rh_lit: S.C_lit, F.an_rh_lit: S.C_lit, F.f_lit: S.C_lit,
    F.lit2som: S.C_lit, F.fx_lit2som: S.C_lit,
    F.ae_rh_cwd: S.C_cwd, F.an_rh_cwd: S.C_cwd, F.f_cwd: S.C_cwd,
    F.cwd2som: S.C_cwd, F.fx_cwd2som: S.C_cwd,
    F.ae_rh_som: S.C_som, F.an_rh_som: S.C_som, F.f_som: S.C_som,
    F.melt: S.H2O_SWE, F.sublimation: S.H2O_SWE,
    F.evap: S.H2O_LY1, F.transp1: S.H2O_LY1, F.ly1xly2: S.H2O_LY1,
    F.q_ly1: S.H2O_LY1,
    F.transp2: S.H2O_LY2, F.ly2xly3: S.H2O_LY2, F.q_ly2: S.H2O_LY2,
    F.q_ly3: S.H2O_LY3,
    F.evap_e: S.E_LY1, F.transp1_e: S.E_LY1, F.q_ly1_e: S.E_LY1,
    F.ly1xly2_e: S.E_LY1, F.ly1xly2_th_e: S.E_LY1,
    F.transp2_e: S.E_LY2, F.q_ly2_e: S.E_LY2, F.ly2xly3_e: S.E_LY2,
    F.ly2xly3_th_e: S.E_LY2,
    F.q_ly3_e: S.E_LY3,
}


def state_input_fluxes(pool: int) -> list[int]:
    return sorted(f for f, p in SINK.items() if p == pool)


def state_output_fluxes(pool: int) -> list[int]:
    return sorted(f for f, p in SOURCE.items() if p == pool)
