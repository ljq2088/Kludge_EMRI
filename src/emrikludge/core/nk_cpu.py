def generate_nk_waveform_cpu(params: EMRIParameters, config: WaveformConfig):
    # 1. initial_consts = from_p_e_iota_to_consts(...)
    # 2. trajectory = integrate_kerr_geodesic(...)
    # 3. possibly apply flux evolution (nk_fluxes)
    # 4. waveform = generate_nk_waveform(trajectory, params, config)
    # 5. if include_lisa_response: use project_to_lisa_channels(...)
    # return t, hI, hII, h_plus, h_cross
