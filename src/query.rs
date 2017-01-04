// Copyright (C) 2017 Chris Liebert

use frustum::Frustum;

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

/// `FrustumQueryObject`
///
/// A data structure with a collection of frustum query objects
///
#[allow(unused)]
pub struct FrustumQuery {
    objects: Vec<FrustumQueryObject>,
}

#[allow(unused)]
impl FrustumQuery {
    /// Create a new FurstumQuery from a colletion of objects
    pub fn new(objects: Vec<FrustumQueryObject>) -> FrustumQuery {
        FrustumQuery { objects: objects }
    }
    
    /// Query if a list of objects are in the frustum
    ///
    pub fn execute(&self, frustum: &Frustum, result: &mut Vec<bool>) {
        for i in 0 .. self.objects.len() {
            let o = &self.objects[i];
            result[i] = frustum.sphere_intersecting(&o.x, &o.y, &o.z, &o.r);
        }
    }
    
    /// Query if a list of objects are in the frustum (in parallel) with Rayon
    ///
    #[cfg(feature = "multithreaded_rayon")]
    pub fn execute_parallel_rayon(&self, frustum: &Frustum, result: &mut Vec<bool>) {
        use rayon::prelude::*;
        self.objects.par_iter().map(|&o| {
            frustum.sphere_intersecting(&o.x, &o.y, &o.z, &o.r)
        }).collect_into(result);
    }
}