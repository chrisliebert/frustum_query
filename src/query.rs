// Copyright (C) 2017-2026 Chris Liebert

use crate::frustum::Frustum;

/// `FrustumQueryObject`
///
/// A data structure describing an object's local center and bounding radius
///
#[derive(Copy, Clone, Debug)]
pub struct FrustumQueryObject {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub r: f32,
}

/// `FrustumQuery`
///
/// Stores objects in Structure-of-Arrays (SoA) layout for efficient SIMD loads.
///
#[allow(unused)]
pub struct FrustumQuery {
    xs: Vec<f32>,
    ys: Vec<f32>,
    zs: Vec<f32>,
    rs: Vec<f32>,
}

#[allow(unused)]
impl FrustumQuery {
    /// Create a new FrustumQuery from a collection of objects.
    /// Converts AoS → SoA once at construction time.
    pub fn new(objects: Vec<FrustumQueryObject>) -> FrustumQuery {
        let n = objects.len();
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        let mut zs = Vec::with_capacity(n);
        let mut rs = Vec::with_capacity(n);
        for o in objects {
            xs.push(o.x);
            ys.push(o.y);
            zs.push(o.z);
            rs.push(o.r);
        }
        FrustumQuery { xs, ys, zs, rs }
    }

    /// Number of objects in the query.
    pub fn len(&self) -> usize {
        self.xs.len()
    }

    /// Query if a list of objects are in the frustum (scalar).
    pub fn execute(&self, frustum: &Frustum, result: &mut Vec<bool>) {
        let n = self.len();
        assert_eq!(result.len(), n, "result Vec must have the same length as the number of objects");
        for i in 0..n {
            result[i] = frustum.sphere_intersecting(&self.xs[i], &self.ys[i], &self.zs[i], &self.rs[i]);
        }
    }

    /// Query if a list of objects are in the frustum (in parallel) with Rayon.
    #[cfg(feature = "multithreaded_rayon")]
    pub fn execute_parallel_rayon(&self, frustum: &Frustum, result: &mut Vec<bool>) {
        let n = self.len();
        assert_eq!(result.len(), n, "result Vec must have the same length as the number of objects");
        use rayon::prelude::*;
        (0..n)
            .into_par_iter()
            .map(|i| {
                frustum.sphere_intersecting(&self.xs[i], &self.ys[i], &self.zs[i], &self.rs[i])
            })
            .collect_into_vec(result);
    }

    /// SIMD-accelerated query (processes 8 spheres at a time using `wide::f32x8`).
    /// Requires the `simd` feature. Falls back to scalar for the remainder.
    ///
    /// Data is already in SoA form, so loads are contiguous.
    /// For best performance on AVX-512, compile with:
    /// `RUSTFLAGS="-C target-feature=+avx512f,+avx512dq,+avx512vl,+avx512bw"`
    #[cfg(feature = "simd")]
    pub fn execute_simd(&self, frustum: &Frustum, result: &mut Vec<bool>) {
        let n = self.len();
        assert_eq!(result.len(), n, "result Vec must have the same length as the number of objects");
        use wide::f32x8;
        use std::ops::BitOr;

        // Pre-splat the six planes once (constant for the whole query)
        let mut px = [f32x8::ZERO; 6];
        let mut py = [f32x8::ZERO; 6];
        let mut pz = [f32x8::ZERO; 6];
        let mut pd = [f32x8::ZERO; 6];
        for p in 0..6 {
            let plane = frustum.planes[p];
            px[p] = f32x8::splat(plane[0]);
            py[p] = f32x8::splat(plane[1]);
            pz[p] = f32x8::splat(plane[2]);
            pd[p] = f32x8::splat(plane[3]);
        }

        let mut i = 0;
        // Process full groups of 8 with contiguous SoA loads
        while i + 8 <= n {
            let xs_arr: [f32; 8] = self.xs[i..i + 8].try_into().unwrap();
            let ys_arr: [f32; 8] = self.ys[i..i + 8].try_into().unwrap();
            let zs_arr: [f32; 8] = self.zs[i..i + 8].try_into().unwrap();
            let rs_arr: [f32; 8] = self.rs[i..i + 8].try_into().unwrap();
            let vx = f32x8::from(xs_arr);
            let vy = f32x8::from(ys_arr);
            let vz = f32x8::from(zs_arr);
            let neg_r = -f32x8::from(rs_arr);
            let mut outside_mask = f32x8::ZERO;

            for p in 0..6 {
                // FMA chain: dist = ((px*vx + py*vy) + pz*vz) + pd
                let dist = px[p].mul_add(vx, py[p].mul_add(vy, pz[p].mul_add(vz, pd[p])));
                let fail = dist.simd_le(neg_r);
                outside_mask = outside_mask.bitor(fail);
            }

            let inside_mask = outside_mask.simd_eq(f32x8::ZERO);
            let bits = inside_mask.to_bitmask();
            // Unrolled write-back
            result[i + 0] = (bits & 1) != 0;
            result[i + 1] = (bits & 2) != 0;
            result[i + 2] = (bits & 4) != 0;
            result[i + 3] = (bits & 8) != 0;
            result[i + 4] = (bits & 16) != 0;
            result[i + 5] = (bits & 32) != 0;
            result[i + 6] = (bits & 64) != 0;
            result[i + 7] = (bits & 128) != 0;
            i += 8;
        }

        // Masked SIMD remainder (avoids scalar fallback for the tail)
        if i < n {
            let mut xs_arr = [0.0f32; 8];
            let mut ys_arr = [0.0f32; 8];
            let mut zs_arr = [0.0f32; 8];
            let mut rs_arr = [0.0f32; 8];
            let rem = n - i;
            xs_arr[..rem].copy_from_slice(&self.xs[i..n]);
            ys_arr[..rem].copy_from_slice(&self.ys[i..n]);
            zs_arr[..rem].copy_from_slice(&self.zs[i..n]);
            rs_arr[..rem].copy_from_slice(&self.rs[i..n]);
            let vx = f32x8::from(xs_arr);
            let vy = f32x8::from(ys_arr);
            let vz = f32x8::from(zs_arr);
            let neg_r = -f32x8::from(rs_arr);
            let mut outside_mask = f32x8::ZERO;
            for p in 0..6 {
                let dist = px[p].mul_add(vx, py[p].mul_add(vy, pz[p].mul_add(vz, pd[p])));
                let fail = dist.simd_le(neg_r);
                outside_mask = outside_mask.bitor(fail);
            }
            let inside_mask = outside_mask.simd_eq(f32x8::ZERO);
            let bits = inside_mask.to_bitmask();
            for k in 0..rem {
                result[i + k] = (bits & (1 << k)) != 0;
            }
        }
    }
}
