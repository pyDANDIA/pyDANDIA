from sys import argv
from os import getcwd, path, remove, environ
import numpy as np
from astropy import table
from pyDANDIA import  photometry_classes as pc
from pyDANDIA import  logs
from pyDANDIA import  event_colour_analysis
from pyDANDIA import  spectral_type_data
from pyDANDIA import  red_clump_utilities
from pyDANDIA import  config_utils
from pyDANDIA import  lens_properties
import pyslalib

def run_event_colour_analysis():

    config = get_args()

    log = logs.start_stage_log( config['output_dir'], 'analyse_event_colours' )

    (source, blend, RC, event_model) = load_event_data(config,log)

    (source,blend) = calc_phot_properties(source, blend, RC, log)

    (source, blend) = event_colour_analysis.match_source_blend_isochrones(config,source,blend,log)

    (source, blend) = event_colour_analysis.calc_source_blend_ang_radii(source, blend, log)

    (source, blend) = event_colour_analysis.calc_source_blend_physical_radii(source, blend, log)

    (source,blend) = event_colour_analysis.calc_source_blend_distance(source, blend, RC, log)

    lens = calc_lens_parameters(config, event_model, source, RC, log)

    output_source_blend_data_latex(config,source,blend,log)

    output_lens_parameters_latex(config,source,lens,log)

    logs.close_log(log)

def get_args():

    if len(argv) == 1:

        config_file = input('Please enter the path to the configuration file: ')

    else:

        config_file = argv[1]

    config = config_utils.build_config_from_json(config_file)

    return config

def load_event_data(config,log):
    """Function to load the parameters of the source, blend and Red Clump"""

    source = pc.Star(file_path=config['source_parameters_file'])
    log.info('Loaded source data:')
    log.info(source.summary(show_mags=True))
    log.info(source.summary(show_mags=False,show_colours=True))
    log.info(source.summary(show_mags=False,johnsons=True))

    blend = pc.Star(file_path=config['blend_parameters_file'])
    log.info('Loaded blend data:')
    log.info(blend.summary(show_mags=True))
    log.info(blend.summary(show_mags=False,show_colours=True))
    log.info(blend.summary(show_mags=False,johnsons=True))

    RC = pc.Star(file_path=config['red_clump_parameters_file'])
    log.info('Loaded Red Clump data:')
    log.info(RC.summary(show_mags=True))
    log.info(RC.summary(show_mags=False,show_colours=True))
    log.info(RC.summary(show_mags=False,johnsons=True))

    log.info('\n')
    log.info('Extinction, d(g) = '+str(RC.A_g)+' +/- '+str(RC.sig_A_g)+'mag')
    log.info('Extinction, d(r) = '+str(RC.A_r)+' +/- '+str(RC.sig_A_r)+'mag')
    log.info('Extinction, d(i) = '+str(RC.A_i)+' +/- '+str(RC.sig_A_i)+'mag')
    log.info('Reddening, E(g-r) = '+str(RC.Egr)+' +/- '+str(RC.sig_Egr)+'mag')
    log.info('Reddening, E(g-i) = '+str(RC.Egi)+' +/- '+str(RC.sig_Egi)+'mag')
    log.info('Reddening, E(r-i) = '+str(RC.Eri)+' +/- '+str(RC.sig_Eri)+'mag')

    log.info('\n')
    log.info('Extinction, d(V) = '+str(RC.A_V)+' +/- '+str(RC.sig_A_V)+'mag')
    log.info('Extinction, d(I) = '+str(RC.A_I)+' +/- '+str(RC.sig_A_I)+'mag')
    log.info('Reddening, E(V-I) = '+str(RC.EVI)+' +/- '+str(RC.sig_EVI)+'mag')

    event_model = config_utils.load_event_model(config['event_model_parameters_file'],log)

    return source, blend, RC, event_model

def calc_phot_properties(source, blend, RC, log):
    """Function to calculate the de-reddened and extinction-corrected
    photometric properties of the target
    """

    source.calibrate_phot_properties(RC,log=log)
    blend.calibrate_phot_properties(RC,log=log)

    log.info('\nSource star extinction-corrected magnitudes and de-reddened colours:\n')
    log.info(source.summary(show_mags=True))
    log.info(source.summary(show_mags=False,show_colours=True))
    log.info(source.summary(show_mags=False,show_cal=True))
    log.info(source.summary(show_mags=False,show_cal=True,show_colours=True))
    log.info(source.summary(show_mags=False,johnsons=True,show_cal=True))

    log.info('\nBlend extinction-corrected magnitudes and de-reddened colours:\n')
    log.info(blend.summary(show_mags=True))
    log.info(blend.summary(show_mags=False,show_colours=True))
    log.info(blend.summary(show_mags=False,show_cal=True))
    log.info(blend.summary(show_mags=False,show_cal=True,show_colours=True))
    log.info(blend.summary(show_mags=False,johnsons=True,show_cal=True))

    return source,blend

def calc_lens_parameters(params, event_model, source, RC, log):
    """Function to compute the physical parameters of the lens"""

    earth_position = pyslalib.slalib.sla_epv(event_model['t0']-2400000.0)

    v_earth = earth_position[1]     # Earth's heliocentric velocity vector

    pi_E = [ event_model['pi_E_N'], event_model['pi_E_E'] ]
    sig_pi_E = [ event_model['sig_pi_E_N'], event_model['sig_pi_E_E'] ]

    lens = lens_properties.Lens()
    lens.ra = params['target_ra']
    lens.dec = params['target_dec']
    lens.tE = event_model['tE']
    lens.sig_tE = event_model['sig_tE']
    lens.t0 = event_model['t0']
    lens.sig_t0 = event_model['sig_t0']
    lens.rho = event_model['rho']
    lens.sig_rho = event_model['sig_rho']
    lens.pi_E = np.array([ event_model['pi_E_N'], event_model['pi_E_E'] ])
    lens.sig_pi_E = np.array([ event_model['sig_pi_E_N'], event_model['sig_pi_E_E'] ])

    lens.calc_angular_einstein_radius(source.ang_radius,source.sig_ang_radius,log=log)

    lens.calc_distance(RC.D,0.0,log)
    lens.calc_distance_modulus(log)
    lens.calc_einstein_radius(log)

    if event_model['logq'] != None and event_model['logs'] != None:
        lens.q = 10**(event_model['logq'])
        lens.sig_q = (event_model['sig_logq']/event_model['logq']) * lens.q
        lens.s = 10**(event_model['logs'])
        lens.sig_s = (event_model['sig_logs']/event_model['logs']) * lens.s

        lens.calc_projected_separation(log)

    lens.calc_masses(log)

    if event_model['dsdt'] != None and event_model['dalphadt'] != None:
        lens.dsdt = event_model['dsdt']
        lens.sig_dsdt = event_model['sig_dsdt']
        lens.dalphadt = event_model['dalphadt']
        lens.sig_dalphadt = event_model['sig_dalphadt']

        lens.calc_orbital_energies(log)

    lens.calc_rel_proper_motion(log)

    return lens

def output_source_blend_data_latex(params,source,blend,log):
    """Function to output a LaTex format table with the source and blend data"""

    file_path = path.join(params['output_dir'],'source_blend_data_table.tex')

    t = open(file_path, 'w')

    t.write('\\begin{table}[h!]\n')
    t.write('\\centering\n')
    t.write('\\caption{Photometric properties of the source star (S) and blend (b).} \label{tab:targetphot}\n')
    t.write('\\begin{tabular}{llll}\n')
    t.write('\\hline\n')
    t.write('\\hline\n')
    t.write('$m_{g,\\rm S}$ & '+pc.convert_ndp(source.g,3)+' $\pm$ '+pc.convert_ndp(source.sig_g,3)+'\,mag & $m_{g,b}$ & '+pc.convert_ndp(blend.g,3)+' $\pm$ '+pc.convert_ndp(blend.sig_g,3)+'\,mag\\\\\n')
    t.write('$m_{r,\\rm S}$ & '+pc.convert_ndp(source.r,3)+' $\pm$ '+pc.convert_ndp(source.sig_r,3)+'\,mag & $m_{r,b}$ & '+pc.convert_ndp(blend.r,3)+' $\pm$ '+pc.convert_ndp(blend.sig_r,3)+'\,mag\\\\\n')
    t.write('$m_{i,\\rm S}$ & '+pc.convert_ndp(source.i,3)+' $\pm$ '+pc.convert_ndp(source.sig_i,3)+'\,mag & $m_{i,b}$ & '+pc.convert_ndp(blend.i,3)+' $\pm$ '+pc.convert_ndp(blend.sig_i,3)+'\,mag\\\\\n')
    t.write('$(g-r)_{\\rm S}$ & '+pc.convert_ndp(source.gr,3)+' $\pm$ '+pc.convert_ndp(source.sig_gr,3)+'\,mag & $(g-r)_{b}$ & '+pc.convert_ndp(blend.gr,3)+' $\pm$ '+pc.convert_ndp(blend.sig_gr,3)+'\,mag\\\\\n')
    t.write('$(g-i)_{\\rm S}$ & '+pc.convert_ndp(source.gi,3)+' $\pm$ '+pc.convert_ndp(source.sig_gi,3)+'\,mag & $(g-i)_{b}$ & '+pc.convert_ndp(blend.gi,3)+' $\pm$ '+pc.convert_ndp(blend.sig_gi,3)+'\,mag\\\\\n')
    t.write('$(r-i)_{\\rm S}$ & '+pc.convert_ndp(source.ri,3)+' $\pm$ '+pc.convert_ndp(source.sig_ri,3)+'\,mag & $(r-i)_{b}$ & '+pc.convert_ndp(blend.ri,3)+' $\pm$ '+pc.convert_ndp(blend.sig_ri,3)+'\,mag\\\\\n')
#    t.write('$m_{g,s,0}$ & '+pc.convert_ndp(source.g_0,3)+' $\pm$ '+pc.convert_ndp(source.sig_g_0,3)+'\,mag & $m_{g,b,0}$ & '+pc.convert_ndp(blend.g_0,3)+' $\pm$ '+pc.convert_ndp(blend.sig_g_0,3)+'\,mag\\\\\n')
#    t.write('$m_{r,s,0}$ & '+pc.convert_ndp(source.r_0,3)+' $\pm$ '+pc.convert_ndp(source.sig_r_0,3)+'\,mag & $m_{r,b,0}$ & '+pc.convert_ndp(blend.r_0,3)+' $\pm$ '+pc.convert_ndp(blend.sig_r_0,3)+'\,mag\\\\\n')
#    t.write('$m_{i,s,0}$ & '+pc.convert_ndp(source.i_0,3)+' $\pm$ '+pc.convert_ndp(source.sig_i_0,3)+'\,mag & $m_{i,b,0}$ & '+pc.convert_ndp(blend.i_0,3)+' $\pm$ '+pc.convert_ndp(blend.sig_i_0,3)+'\,mag\\\\\n')
#    t.write('$(g-r)_{s,0}$ & '+pc.convert_ndp(source.gr_0,3)+' $\pm$ '+pc.convert_ndp(source.sig_gr_0,3)+'\,mag & $(g-r)_{b,0}$ & '+pc.convert_ndp(blend.gr_0,3)+' $\pm$ '+pc.convert_ndp(blend.sig_gr_0,3)+'\,mag\\\\\n')
#    t.write('$(r-i)_{s,0}$ & '+pc.convert_ndp(source.ri_0,3)+' $\pm$ '+pc.convert_ndp(source.sig_ri_0,3)+'\,mag & $(r-i)_{b,0}$ & '+pc.convert_ndp(blend.ri_0,3)+' $\pm$ '+pc.convert_ndp(blend.sig_ri_0,3)+'\,mag\\\\\n')
    t.write('$m_{g,\\rm S,0}$ & '+pc.convert_ndp(source.g_0,3)+' $\pm$ '+pc.convert_ndp(source.sig_g_0,3)+'\,mag &  & \\\\\n')
    t.write('$m_{r,\\rm S,0}$ & '+pc.convert_ndp(source.r_0,3)+' $\pm$ '+pc.convert_ndp(source.sig_r_0,3)+'\,mag &  & \\\\\n')
    t.write('$m_{i,\\rm S,0}$ & '+pc.convert_ndp(source.i_0,3)+' $\pm$ '+pc.convert_ndp(source.sig_i_0,3)+'\,mag &  & \\\\\n')
    t.write('$(g-r)_{\\rm S,0}$ & '+pc.convert_ndp(source.gr_0,3)+' $\pm$ '+pc.convert_ndp(source.sig_gr_0,3)+'\,mag &  & \\\\\n')
    t.write('$(g-i)_{\\rm S,0}$ & '+pc.convert_ndp(source.gi_0,3)+' $\pm$ '+pc.convert_ndp(source.sig_gi_0,3)+'\,mag &  & \\\\\n')
    t.write('$(r-i)_{\\rm S,0}$ & '+pc.convert_ndp(source.ri_0,3)+' $\pm$ '+pc.convert_ndp(source.sig_ri_0,3)+'\,mag &  & \\\\\n')
    t.write('\\hline\n')
    t.write('\\end{tabular}\n')
    t.write('\\end{table}\n')

    t.close()

    log.info('Output source and blend data in laTex table to '+file_path)

def output_lens_parameters_latex(params,source,lens,log):
    """Function to output a LaTex format table with the lens parameters"""

    file_path = path.join(params['output_dir'],'lens_data_table.tex')

    t = open(file_path, 'w')

    t.write('\\begin{table}[h!]\n')
    t.write('\\centering\n')
    t.write('\\caption{Physical properties of the source and lens system} \\label{tab:lensproperties}\n')
    t.write('\\begin{tabular}{lll}\n')
    t.write('\\hline\n')
    t.write('\\hline\n')
    t.write('Parameter   &   Units    &   Value \\\\\n')
    t.write('$\\theta_{\\rm{S}}$  & $\\mu$as     & '+pc.convert_ndp(source.ang_radius,3)+'$\pm$'+pc.convert_ndp(source.sig_ang_radius,3)+'\\\\\n')
    t.write('$\\theta_{\\rm{E}}$  & $\\mu$as     & '+pc.convert_ndp(lens.thetaE,3)+'$\pm$'+pc.convert_ndp(lens.sig_thetaE,3)+'\\\\\n')
    t.write('$R_{\\rm{S}}$       & $R_{\\odot}$ & '+pc.convert_ndp(source.radius,3)+'$\pm$'+pc.convert_ndp(source.sig_radius,3)+'\\\\\n')
    t.write('$M_{L,tot}$        & $M_{\\odot}$ & '+pc.convert_ndp(lens.ML,3)+'$\pm$'+pc.convert_ndp(lens.sig_ML,3)+'\\\\\n')
    if 'q' in dir(lens):
        t.write('$M_{L,1}$          & $M_{\\odot}$ & '+pc.convert_ndp(lens.M1,3)+'$\pm$'+pc.convert_ndp(lens.sig_M1,3)+'\\\\\n')
        t.write('$M_{L,2}$          & $M_{\\odot}$ & '+pc.convert_ndp(lens.M2,3)+'$\pm$'+pc.convert_ndp(lens.sig_M2,3)+'\\\\\n')
        t.write('$a_{\\perp}$       & AU          & '+pc.convert_ndp(lens.a_proj,3)+'$\pm$'+pc.convert_ndp(lens.sig_a_proj,3)+'\\\\\n')
    t.write('$D_{L}$            & Kpc         & '+pc.convert_ndp(lens.D,3)+'$\pm$'+pc.convert_ndp(lens.sig_D,3)+'\\\\\n')
#    t.write('KE/PE              &             & '+pc.convert_ndp(lens.kepe,3)+'$\pm$'+pc.convert_ndp(lens.sig_kepe,3)+'\\\\\n')
    t.write('$\mu$              & mas yr$^{-1}$ & '+pc.convert_ndp(lens.mu_rel,2)+'$\pm$'+pc.convert_ndp(lens.sig_mu_rel,2)+'\\\\\n')
    t.write('\\hline\n')
    t.write('\\end{tabular}\n')
    t.write('\\end{table}\n')

    t.close()

    log.info('Output lens parameters in laTex table to '+file_path)

if __name__ == '__main__':
    run_event_colour_analysis()
