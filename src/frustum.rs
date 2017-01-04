// Copyright (C) 2017 Chris Liebert

// Frustum extracting and testing algorithm used from
// http://www.racer.nl/reference/vfc_markmorley.htm

/// `Frustum`
///
/// A data structure for six planes of the view frustum
///
#[allow(unused)]
#[derive(Debug)]
pub struct Frustum {
    // right left bottom top near and far planes
    planes:	[[f32; 4]; 6],
}

#[allow(unused)]
impl Frustum {
    /// Create a frustum given the product of the modelview and projection as a float array
    ///
    #[inline]
    pub fn from_modelview_projection(clip: &[f32; 16]) -> Frustum {
        // Extract the numbers for the RIGHT plane
        let rdx: f32 = clip[3] - clip[0];
        let rdy: f32 = clip[7] - clip[4];
        let rdz: f32 = clip[11] - clip[8];
        let rdw: f32 = clip[15] - clip[12];

        // Divisor to normilize right plane
        let rd: f32 = (rdx.powi(2) + rdy.powi(2) + rdz.powi(2)).sqrt();
        debug_assert!(rd > 0.0f32, "Invalid modelview projection matrix: right plane divisor cannot be 0");

        // Extract the numbers for the LEFT plane
        let ldx: f32 = clip[3] + clip[0];
        let ldy: f32 = clip[7] + clip[4];
        let ldz: f32 = clip[11] + clip[8];
        let ldw: f32 = clip[15] + clip[12];

        // Divisor to normilize left plane
        let ld: f32 = (ldx.powi(2) + ldy.powi(2) + ldz.powi(2)).sqrt();
        debug_assert!(rd > 0.0f32, "Invalid modelview projection matrix: left plane divisor cannot be 0");

        // Extract the BOTTOM plane
        let bdx: f32 = clip[3] + clip[1];
        let bdy: f32 = clip[7] + clip[5];
        let bdz: f32 = clip[11] + clip[9];
        let bdw: f32 = clip[15] + clip[13];

        // Divisor to normilize bottom plane
        let bd: f32 = (bdx.powi(2) + bdy.powi(2) + bdz.powi(2)).sqrt();
        debug_assert!(rd > 0.0f32, "Invalid modelview projection matrix: bottom plane divisor cannot be 0");

        // Extract the TOP plane
        let tdx: f32 = clip[3] - clip[1];
        let tdy: f32 = clip[7] - clip[5];
        let tdz: f32 = clip[11] - clip[9];
        let tdw: f32 = clip[15] - clip[13];

        // Divisor to normilize top plane
        let td: f32 = (tdx.powi(2) + tdy.powi(2) + tdz.powi(2)).sqrt();
        debug_assert!(rd > 0.0f32, "Invalid modelview projection matrix: top plane divisor cannot be 0");

        // Extract the FAR plane
        let fdx: f32 = clip[3] - clip[2];
        let fdy: f32 = clip[7] - clip[6];
        let fdz: f32 = clip[11] - clip[10];
        let fdw: f32 = clip[15] - clip[14];

        // Divisor to normilize far plane
        let fd: f32 = (fdx.powi(2) + fdy.powi(2) + fdz.powi(2)).sqrt();
        debug_assert!(rd > 0.0f32, "Invalid modelview projection matrix: far plane divisor cannot be 0");

        // Extract the NEAR plane
        let ndx: f32 = clip[3] + clip[2];
        let ndy: f32 = clip[7] + clip[6];
        let ndz: f32 = clip[11] + clip[10];
        let ndw: f32 = clip[15] + clip[14];

        // Divisor to normilize near plane
        let nd: f32 = (ndx.powi(2) + ndy.powi(2) + ndz.powi(2)).sqrt();
        debug_assert!(rd > 0.0f32, "Invalid modelview projection matrix: near plane divisor cannot be 0");

        Frustum {
            planes: [[rdx / rd, rdy / rd, rdz / rd, rdw / rd],
                     [ldx / ld, ldy / ld, ldz / ld, ldw / ld],
                     [bdx / bd, bdy / bd, bdz / bd, bdw / bd],
                     [tdx / td, tdy / td, tdz / td, tdw / td],
                     [fdx / fd, fdy / fd, fdz / fd, fdw / fd],
                     [ndx / nd, ndy / nd, ndz / nd, ndw / nd]],
        }
    }
    
    /// Create a frustum from column major modelview and projection float arrays matrices
    ///
    pub fn from_modelview_and_projection(modl: &[f32; 16], proj: &[f32; 16]) -> Frustum {
                
        // clip = projection * modelview modelview
        let clip: [f32; 16] = [
            modl[0]		* proj[0] + modl[1] 	* proj[4] + modl[2]		* proj[8]	+ modl[3]	* proj[12],
            modl[0]		* proj[1] + modl[1] 	* proj[5] + modl[2] 	* proj[9]	+ modl[3]	* proj[13],
            modl[0]		* proj[2] + modl[1] 	* proj[6] + modl[2] 	* proj[10]	+ modl[3]	* proj[14],
            modl[0]		* proj[3] + modl[1] 	* proj[7] + modl[2] 	* proj[11]	+ modl[3]	* proj[15],
            modl[4]		* proj[0] + modl[5] 	* proj[4] + modl[6] 	* proj[8]	+ modl[7]	* proj[12],
            modl[4]		* proj[1] + modl[5] 	* proj[5] + modl[6] 	* proj[9]	+ modl[7]	* proj[13],
            modl[4]		* proj[2] + modl[5] 	* proj[6] + modl[6] 	* proj[10]	+ modl[7]	* proj[14],
            modl[4]		* proj[3] + modl[5] 	* proj[7] + modl[6] 	* proj[11]	+ modl[7]	* proj[15],
            modl[8]		* proj[0] + modl[9] 	* proj[4] + modl[10]	* proj[8]	+ modl[11]	* proj[12],
            modl[8]		* proj[1] + modl[9] 	* proj[5] + modl[10] 	* proj[9]	+ modl[11]	* proj[13],
            modl[8]		* proj[2] + modl[9] 	* proj[6] + modl[10] 	* proj[10]	+ modl[11]	* proj[14],
            modl[8]		* proj[3] + modl[9] 	* proj[7] + modl[10] 	* proj[11]	+ modl[11]	* proj[15],
            modl[12]	* proj[0] + modl[13]	* proj[4] + modl[14] 	* proj[8]	+ modl[15]	* proj[12],
            modl[12]	* proj[1] + modl[13]	* proj[5] + modl[14] 	* proj[9]	+ modl[15]	* proj[13],
            modl[12]	* proj[2] + modl[13]	* proj[6] + modl[14] 	* proj[10]	+ modl[15]	* proj[14],
            modl[12]	* proj[3] + modl[13]	* proj[7] + modl[14] 	* proj[11]	+ modl[15]	* proj[15],
        ];
        Frustum::from_modelview_projection(&clip)
    }

    /// Create a frustum from 2-dimensional column-major modelview and projection float arrays
    ///
    pub fn from_modelview_and_projection_2d(modl: &[[f32; 4]; 4], proj: &[[f32; 4]; 4]) -> Frustum {
        let modl_1d: [f32; 16] = [
            modl[0][0],
            modl[0][1],
            modl[0][2],
            modl[0][3],
                modl[1][0],
                modl[1][1],
                modl[1][2],
                modl[1][3],
                    modl[2][0],
                    modl[2][1],
                    modl[2][2],
                    modl[2][3],
                        modl[3][0],
                        modl[3][1],
                        modl[3][2],
                        modl[3][3],
        ];
        let proj_1d: [f32; 16] = [
            proj[0][0],
            proj[0][1],
            proj[0][2],
            proj[0][3],
                proj[1][0],
                proj[1][1],
                proj[1][2],
                proj[1][3],
                    proj[2][0],
                    proj[2][1],
                    proj[2][2],
                    proj[2][3],
                        proj[3][0],
                        proj[3][1],
                        proj[3][2],
                        proj[3][3],
        ];
        
        return Frustum::from_modelview_and_projection(&modl_1d, &proj_1d);
    }

    /// A point is intersecting the renderable region of the frustum
    ///
    pub fn point_intersecting(&self, x: &f32, y: &f32, z: &f32) -> bool {
        for p in 0..6 {
            if (self.planes[p][0] * x + self.planes[p][1] * y + self.planes[p][2] * z + self.planes[p][3]) <= 0.0f32 {
                return false;
            }
        }
        true
    }

    /// A sphere is intersecting the renderable region of the frustum
    ///
    pub fn sphere_intersecting(&self, x: &f32, y: &f32, z: &f32, r: &f32) -> bool {
        for p in 0..6 {
            if self.planes[p][0] * x + self.planes[p][1] * y + self.planes[p][2] * z + self.planes[p][3] <= (0.0f32 - r) {
                return false;
            }
        }
        true
    }
}