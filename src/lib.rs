// Copyright (C) 2017 Chris Liebert

#![crate_name = "frustum_query"]
#![crate_type = "rlib"]

#[cfg(feature = "multithreaded_rayon")]
extern crate rayon;

pub mod frustum;
pub mod query;

#[cfg(test)]
mod tests {
    use crate::frustum::Frustum;
    use crate::query::FrustumQuery;
    use crate::query::FrustumQueryObject;
    /// A basic unit test (no multithreading)
    
    fn get_test_objects(n: usize) -> Vec<FrustumQueryObject> {
        let mut objects: Vec<FrustumQueryObject> = Vec::with_capacity(n * 4);
        let copy = vec![
            FrustumQueryObject { x: 1.0f32, y: 1.0f32, z: 1.0f32, r: 1.0f32},
            FrustumQueryObject { x: 2.0f32, y: 3.0f32, z: 4.0f32, r: 2.0f32},
            FrustumQueryObject { x: 3.0f32, y: 4.0f32, z: 5.0f32, r: 3.0f32},
            FrustumQueryObject { x: 4.0f32, y: 5.0f32, z: 6.0f32, r: 7.0f32},
        ];
        // Make the array large
        for _ in 0 .. n {
            objects.append(
                &mut copy.clone()
            );
        }
        println!("Using {} objects", objects.len());
        objects
    }
    
    #[test]
    fn simple() {
        let mvp: [f32; 16] = [1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32, 9.0f32, 10.0f32, 11.0f32, 12.0f32, 13.0f32, 14.0f32, 15.0f32, 16.0f32];
        let frustum: Frustum = Frustum::from_modelview_projection(&mvp);
        let n = 15000; // 60,000 objects
        let objects: Vec<FrustumQueryObject> = get_test_objects(n);
        let num_objects = objects.len();                
        let mut query_result1: Vec<bool> = Vec::with_capacity(num_objects);
        let mut query_result2: Vec<bool> = Vec::with_capacity(num_objects);
        for _ in 0 .. num_objects {
            query_result1.push(true);
            query_result2.push(true);
        }
        let frustum_query: FrustumQuery = FrustumQuery::new(objects);
        frustum_query.execute(&frustum, &mut query_result1);
        assert!(query_result1 == query_result2);   
    }
    
    /// This compairison test can be run with `cargo test --features multithreaded_rayon -- --nocapture` to display output
    #[cfg(feature = "multithreaded_rayon")]
    #[test]
    fn multithreaded_compairison_rayon() {
        use std::time::Instant;
        let mvp: [f32; 16] = [1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32, 9.0f32, 10.0f32, 11.0f32, 12.0f32, 13.0f32, 14.0f32, 15.0f32, 16.0f32];
        
        let frustum: Frustum = Frustum::from_modelview_projection(&mvp);
        
        // When tested on an Intel i7 with 4 cores / 8 threads notebook processor on Windows 8.1 Visual Studio 2015 x64 Rust 1.13,
        // at least ~60,000 objects are required before multithreaded version shows improvement
        
        let n = 15000; // with 60,000 objects start to see improved multithreaded performance on average
        // multithreaded performaced is much less predictable and will occationally take longer even with ~60,000
        
        //let n = 250000; // with 1 million objects, multithreaded version is ~4x faster
        // occational low performance spikes are comparable to single-threaded times
        
        //let n = 250000000; // with 1 billion objects, multithreaded version is ~4-8x faster
        
        let objects: Vec<FrustumQueryObject> = get_test_objects(n);
        let num_objects = objects.len();
        let mut query_result1: Vec<bool> = Vec::with_capacity(num_objects);
        let mut query_result2: Vec<bool> = Vec::with_capacity(num_objects);
        for _ in 0 .. num_objects {
            query_result1.push(true);
            query_result2.push(true);
        }
        
        let start = Instant::now();
        
        let frustum_query: FrustumQuery = FrustumQuery::new(objects.clone());
        
        frustum_query.execute(&frustum, &mut query_result1);
        
        let _end = Instant::now();
        let single_thread_time = start.elapsed();
        
        
        let start = Instant::now();
        
        frustum_query.execute_parallel_rayon(&frustum, &mut query_result2);
        
        let _end = Instant::now();
        let multi_thread_time = start.elapsed();
        
        println!("Single-threaded time:\t{:?}", single_thread_time);
        println!("Multi-threaded time:\t{:?}", multi_thread_time);
        assert!(query_result1 == query_result2);   
    }

    #[cfg(feature = "simd")]
    #[test]
    fn simd_matches_scalar() {
        let mvp: [f32; 16] = [1.0, 0.0, 0.0, 0.0,
                              0.0, 1.0, 0.0, 0.0,
                              0.0, 0.0, 1.0, 0.0,
                              0.0, 0.0, 0.0, 1.0];
        let frustum = Frustum::from_modelview_projection(&mvp);
        let objects = get_test_objects(100); // 400 objects
        let n = objects.len();
        let mut scalar = vec![false; n];
        let mut simd = vec![false; n];
        let q = FrustumQuery::new(objects);
        q.execute(&frustum, &mut scalar);
        q.execute_simd(&frustum, &mut simd);
        assert_eq!(scalar, simd, "SIMD path must match scalar path");
    }

    #[cfg(feature = "simd")]
    #[test]
    fn simd_performance_harness() {
        use std::time::Instant;
        let mvp: [f32; 16] = [1.0f32; 16];
        let frustum = Frustum::from_modelview_projection(&mvp);
        let n = 250_000; // 1 million objects
        let objects = get_test_objects(n);
        let num = objects.len();
        let mut res_scalar = vec![false; num];
        let mut res_simd = vec![false; num];
        let q = FrustumQuery::new(objects);

        // Warmup
        q.execute(&frustum, &mut res_scalar);
        q.execute_simd(&frustum, &mut res_simd);

        let start = Instant::now();
        for _ in 0..5 {
            q.execute(&frustum, &mut res_scalar);
        }
        let scalar_t = start.elapsed() / 5;

        let start = Instant::now();
        for _ in 0..5 {
            q.execute_simd(&frustum, &mut res_simd);
        }
        let simd_t = start.elapsed() / 5;

        println!("Scalar: {:?}  SIMD: {:?}  speedup: {:.2}x",
                 scalar_t, simd_t,
                 scalar_t.as_secs_f64() / simd_t.as_secs_f64());
        assert!(simd_t < scalar_t * 2, "SIMD should not be dramatically slower");
        // Note: actual speedup depends on CPU features enabled at compile time
    }
}
