def integrate_kerr_geodesic(params: EMRIParameters, initial_consts: OrbitConstants, config: WaveformConfig) -> GeodesicTrajectory:
    """
    输入：
      params：EMRIParameters（含 M, a,…）
      initial_consts：OrbitConstants (E, Lz, Q) 或 (p,e,iota) 映射过来
      config：WaveformConfig（dt, T 等）
    输出：
      GeodesicTrajectory 包含 t, r(t), θ(t), φ(t), constants(t)（若演化）.
    """
    ...
