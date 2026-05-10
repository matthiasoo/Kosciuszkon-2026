# Honeywell Gold Dataset — PX4 telemetry for GPS spoofing detection

Real autopilot telemetry merged from PX4 uORB topics (`vehicle_attitude`, `vehicle_global_position`, `vehicle_gps_position`, `vehicle_local_position`). Each row is one synchronised sample of a single drone in flight; rows from four flight scenarios are concatenated end-to-end.

## Dataset overview

| | |
|---|---|
| File | `honeywell_gold_dataset.csv` |
| Rows | 24,992 |
| Columns | 84 (83 features + 1 target: `label`) |
| Nulls | 0 |
| Drones | 1 |
| Reference origin | `ref_lat=36.2048055, ref_lon=138.2529121, ref_alt=51.6988 m` |

### Label distribution

| label | count | fraction |
|---|---|---|
| 0 | 11 870 | 47.5% |
| 1 | 13 122 | 52.5% |

The dataset contains rows from four contiguous flight blocks (each 6 248 rows). Some blocks are entirely clean (`label=0`), while others contain spoofing windows where `label=1`.

## Important caveats

1. **Train/test splitting** — the data is stored as four contiguous blocks, *not* shuffled. A random `train_test_split` may leak rows from the same flight block into both folds and inflate scores. Consider using block-aware splitting strategies.
2. **`timestamp` is a sample index, not real time.** It runs `0..24991` with step 1 inside each block (not a wall clock). `time_utc_usec` and `timestamp_time_relative` are constant zero in this file, so there is no absolute time reference. If sampling rate matters for your features, treat the index as ticks and assume PX4-typical 50 Hz / 100 Hz.
3. **35 of 85 columns are constant** in this file (mostly EKF reset counters and validity flags). They carry no information; drop them up front. The full list is at the bottom of this document.
4. **EKF (`*_x`) and raw GPS (`*_y`) use different units**:
   - `lat_x`, `lon_x` — degrees (float)
   - `lat_y`, `lon_y` — degrees × 1e7 (integer-encoded sensor format)
   - `alt_x`, `alt_ellipsoid_x` — metres
   - `alt_y`, `alt_ellipsoid_y` — millimetres

   To compare EKF vs GPS divergence (a strong spoofing signal), normalise:
   ```python
   df['lat_diff_m']  = (df['lat_x'] - df['lat_y']  / 1e7) * 111_000
   df['lon_diff_m']  = (df['lon_x'] - df['lon_y']  / 1e7) * 111_000 * np.cos(np.radians(df['lat_x']))
   df['alt_diff_m']  =  df['alt_x'] - df['alt_y']  / 1000.0
   ```

## Column reference

### Sample / timeline
- **`timestamp`** — sample index within the file (0..24991, step 1). Not seconds.
- **`time_utc_usec`** — GPS UTC microseconds. Constant `0` here.
- **`timestamp_time_relative`** — relative time field from `sensor_gps`. Constant `0` here.

### Attitude (`vehicle_attitude`)
- **`q[0]..q[3]`** — vehicle attitude quaternion (`[w, x, y, z]`, normalised).
- **`delta_q_reset[0]..delta_q_reset[3]`** — quaternion of the last attitude estimator reset (constant in this file).
- **`quat_reset_counter`** — increments on every attitude reset (constant `3` here).

### EKF global position (`vehicle_global_position`, suffix `_x`)
- **`lat_x`** — fused latitude estimate, degrees.
- **`lon_x`** — fused longitude estimate, degrees.
- **`alt_x`** — fused altitude above mean sea level, metres.
- **`alt_ellipsoid_x`** — fused altitude above WGS-84 ellipsoid, metres.
- **`delta_alt`** — altitude delta on last reset (constant `0` here).
- **`eph_x`** — standard deviation of horizontal position estimate, metres.
- **`epv_x`** — standard deviation of vertical position estimate, metres.
- **`terrain_alt`** — terrain altitude estimate (AMSL), metres.
- **`terrain_alt_valid`** — terrain estimate valid (1) / invalid (0).
- **`lat_lon_reset_counter`** — increments on each lat/lon reset (constant `2`).
- **`alt_reset_counter`** — increments on each altitude reset (constant `0`).
- **`dead_reckoning`** — `1` if the EKF has no GPS and is dead-reckoning (constant `0`).

### Raw GPS sensor (`vehicle_gps_position`, suffix `_y`)
- **`lat_y`** — GPS latitude, degrees × 1e7.
- **`lon_y`** — GPS longitude, degrees × 1e7.
- **`alt_y`** — GPS altitude AMSL, millimetres.
- **`alt_ellipsoid_y`** — GPS altitude above WGS-84 ellipsoid, millimetres.
- **`s_variance_m_s`** — variance of GPS speed estimate, (m/s)².
- **`c_variance_rad`** — variance of GPS course estimate, rad².
- **`eph_y`** — GPS reported horizontal position uncertainty (1σ), metres.
- **`epv_y`** — GPS reported vertical position uncertainty (1σ), metres.
- **`hdop`** — horizontal dilution of precision (unitless; lower is better).
- **`vdop`** — vertical dilution of precision.
- **`noise_per_ms`** — average receiver background noise per millisecond. Higher means a noisier RF environment; jamming pushes this up.
- **`jamming_indicator`** — receiver-side jamming likelihood, 0..255 (higher = more interference).
- **`vel_m_s`** — GPS total speed over ground, m/s.
- **`vel_n_m_s`** — GPS velocity north, m/s.
- **`vel_e_m_s`** — GPS velocity east, m/s.
- **`vel_d_m_s`** — GPS velocity down, m/s.
- **`cog_rad`** — GPS course over ground, radians.
- **`heading_offset`** — magnetic heading offset (rad). Constant `0` here.
- **`fix_type`** — GPS fix type: `0` no fix, `2` 2D, `3` 3D, `4` DGPS, `5` RTK float, `6` RTK fixed. Constant `3` (3D fix) here.
- **`vel_ned_valid`** — NED-velocity valid flag (constant `1` here).
- **`satellites_used`** — number of GPS satellites used in the fix.

### EKF local frame (`vehicle_local_position`)
Local NED frame anchored at `ref_lat / ref_lon / ref_alt`.

- **`ref_lat`**, **`ref_lon`**, **`ref_alt`** — origin of the local frame (constant per-flight here).
- **`x`** — north position from origin, metres.
- **`y`** — east position from origin, metres.
- **`z`** — down position from origin (negative = above origin), metres.
- **`delta_xy[0]`**, **`delta_xy[1]`** — north/east position delta on last EKF reset, metres.
- **`delta_z`** — down delta on last EKF reset, metres.
- **`vx`** — north velocity, m/s.
- **`vy`** — east velocity, m/s.
- **`vz`** — down velocity, m/s.
- **`z_deriv`** — finite-difference vertical velocity (`dz/dt`), m/s.
- **`delta_vxy[0]`**, **`delta_vxy[1]`** — north/east velocity delta on last reset.
- **`delta_vz`** — down velocity delta on last reset.
- **`ax`** — north acceleration, m/s².
- **`ay`** — east acceleration, m/s².
- **`az`** — down acceleration, m/s².
- **`heading_y`** — vehicle yaw, radians.
- **`delta_heading`** — yaw delta on last reset, radians.
- **`dist_bottom`** — estimated distance to ground (e.g. range finder fused), metres.
- **`eph`** — local horizontal position uncertainty (1σ), metres.
- **`epv`** — local vertical position uncertainty (1σ), metres.
- **`evh`** — local horizontal velocity uncertainty (1σ), m/s.
- **`evv`** — local vertical velocity uncertainty (1σ), m/s.
- **`xy_valid`** — local horizontal position is valid (constant `1`).
- **`z_valid`** — local vertical position is valid (constant `1`).
- **`v_xy_valid`** — local horizontal velocity is valid (constant `1`).
- **`v_z_valid`** — local vertical velocity is valid (constant `1`).
- **`xy_reset_counter`**, **`z_reset_counter`**, **`vxy_reset_counter`**, **`vz_reset_counter`**, **`heading_reset_counter`** — increment on each respective EKF reset (all constant in this file).
- **`xy_global`** — local x/y are tied to the global frame (constant `1`).
- **`z_global`** — local z is tied to the global frame (constant `1`).
- **`dist_bottom_valid`** — `dist_bottom` is valid (constant `1.0`).

### Target
- **`label`** — binary anomaly flag for this row.
  - `1` — spoofing window active in this sample.
  - `0` — clean (during clean flights, or before the spoofing onset in attack scenarios).

## Constant columns to drop up front

These 35 columns have a single unique value across the file and contribute nothing to a per-row classifier:

```
delta_q_reset[0..3], quat_reset_counter,
delta_alt, lat_lon_reset_counter, alt_reset_counter, dead_reckoning,
time_utc_usec, timestamp_time_relative, heading_offset,
fix_type, vel_ned_valid,
ref_lat, ref_lon, ref_alt,
delta_xy[0], delta_xy[1], delta_z,
delta_vxy[0], delta_vxy[1], delta_vz,
delta_heading,
xy_valid, z_valid, v_xy_valid, v_z_valid,
xy_reset_counter, z_reset_counter, vxy_reset_counter, vz_reset_counter, heading_reset_counter,
xy_global, z_global
```

After dropping these, ~48 informative input columns remain.

## Suggested feature engineering for spoofing detection

The strongest spoofing-specific signals in this schema are:

- **EKF-vs-GPS divergence** — `lat_x` / `lon_x` (EKF) vs `lat_y/1e7` / `lon_y/1e7` (raw GPS), and `alt_x` vs `alt_y/1000`. Differences are zero in clean flight and grow during all three attack scenarios.
- **EKF-vs-GPS velocity divergence** — `vx, vy, vz` (local NED, EKF) vs `vel_n_m_s, vel_e_m_s, vel_d_m_s` (raw GPS).
- **GPS uncertainty fields** — `eph_y`, `epv_y`, `s_variance_m_s`, `c_variance_rad` often spike under spoofing.
- **Receiver health** — `noise_per_ms`, `jamming_indicator`, `satellites_used` for jamming-style attacks (less relevant for pure spoofing).
- **Rolling statistics** of the divergence and velocity fields (windowed mean / std over 10–100 samples) significantly help, given the gradual onset of `horizontal_drift` and `circular_spoof`.
