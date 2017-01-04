// Copyright (C) 2017 Chris Liebert

#[cfg(feature = "multithreaded_rayon")]
extern crate time;
#[cfg(feature = "multithreaded_rayon")]
extern crate rayon;

mod frustum;
mod query;

#[cfg(test)]
mod tests {
    use frustum::Frustum;
    use query::FrustumQuery;
    use query::FrustumQueryObject;
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
        use time::PreciseTime;
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
        
        let start = PreciseTime::now();
        
        let frustum_query: FrustumQuery = FrustumQuery::new(objects.clone());
        
        frustum_query.execute(&frustum, &mut query_result1);
        
        let end = PreciseTime::now();
        let single_thread_time = start.to(end);
        
        
        let start = PreciseTime::now();
        
        frustum_query.execute_parallel_rayon(&frustum, &mut query_result2);
        
        let end = PreciseTime::now();
        let multi_thread_time = start.to(end);
        
        println!("Single-threaded time:\t{}", single_thread_time);
        println!("Multi-threaded time:\t{}", multi_thread_time);
        assert!(query_result1 == query_result2);   
    }
}
