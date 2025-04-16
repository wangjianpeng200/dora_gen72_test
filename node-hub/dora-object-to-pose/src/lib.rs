use core::f32;
use dora_node_api::{
    arrow::{
        array::{AsArray, Float64Array, UInt8Array},
        datatypes::{Float32Type, Int64Type},
    },
    dora_core::config::DataId,
    DoraNode, Event, IntoArrow, Parameter,
};
use eyre::Result;
use std::collections::HashMap;

fn points_to_pose(points: &[(f32, f32, f32)]) -> Vec<f32> {
    let (_x, _y, _z, sum_xy, sum_x2, sum_y2, n, x_min, x_max, y_min, y_max, z_min, z_max) =
        points.iter().fold(
            (
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, -10.0, 10.0, -10.0, 10., -10.0,
            ),
            |(
                acc_x,
                acc_y,
                acc_z,
                acc_xy,
                acc_x2,
                acc_y2,
                acc_n,
                acc_x_min,
                acc_x_max,
                acc_y_min,
                acc_y_max,
                acc_z_min,
                acc_z_max,
            ),
             (x, y, z)| {
                (
                    acc_x + x,
                    acc_y + y,
                    acc_z + z,
                    acc_xy + x * y,
                    acc_x2 + x * x,
                    acc_y2 + y * y,
                    acc_n + 1.,
                    f32::min(acc_x_min, *x),
                    f32::max(acc_x_max, *x),
                    f32::min(acc_y_min, *y),
                    f32::max(acc_y_max, *y),
                    f32::min(acc_z_min, *z),
                    f32::max(acc_z_max, *z),
                )
            },
        );
    let (mean_x, mean_y, mean_z) = (
        (x_max + x_min) / 2.,
        (y_max + y_min) / 2.,
        (z_max + z_min) / 2.,
    );

    // Compute covariance and standard deviations
    let cov = sum_xy / n - mean_x * mean_y;
    let std_x = (sum_x2 / n - mean_x * mean_x).sqrt();
    let std_y = (sum_y2 / n - mean_y * mean_y).sqrt();
    let corr = cov / (std_x * std_y);

    vec![mean_x, mean_y, mean_z, 0., 0., corr * f32::consts::PI / 2.]
}

pub fn lib_main() -> Result<()> {
    let (mut node, mut events) = DoraNode::init_from_env()?;
    let mut image_cache = HashMap::new();

    let mut depth_frame = None;
    let mut width = 640;
    let mut height = 640;
    let mut focal_length = vec![605, 605];
    let mut resolution = vec![605, 605];
    let camera_pitch = std::env::var("CAMERA_PITCH")
        .unwrap_or("2.47".to_string())
        .parse::<f32>()
        .unwrap();
    let cos_theta = camera_pitch.cos(); // np.cos(np.deg2rad(180-38))
    let sin_theta = camera_pitch.sin(); // np.sin(np.deg2rad(180-38))
                                        // (0.32489833, -0.25068134, 0.4761387)

    while let Some(event) = events.recv() {
        if let Event::Input { id, metadata, data } = event {
            match id.as_str() {
                "image" => {
                    let buffer: &UInt8Array = data.as_any().downcast_ref().unwrap();
                    image_cache.insert(id.clone(), buffer.values().to_vec());
                }
                "depth" => {
                    height = if let Some(Parameter::Integer(height)) =
                        metadata.parameters.get("height")
                    {
                        *height
                    } else {
                        height
                    };
                    width =
                        if let Some(Parameter::Integer(width)) = metadata.parameters.get("width") {
                            *width
                        } else {
                            width
                        };
                    focal_length = if let Some(Parameter::ListInt(focals)) =
                        metadata.parameters.get("focal")
                    {
                        focals.to_vec()
                    } else {
                        vec![605, 605]
                    };
                    resolution = if let Some(Parameter::ListInt(resolution)) =
                        metadata.parameters.get("resolution")
                    {
                        resolution.to_vec()
                    } else {
                        vec![640, 480]
                    };
                    let buffer: &Float64Array = data.as_any().downcast_ref().unwrap();
                    depth_frame = Some(buffer.clone());
                }
                "masks" => {
                    let masks = if let Some(data) = data.as_primitive_opt::<Float32Type>() {
                        let data = data
                            .iter()
                            .map(|x| if let Some(x) = x { x > 0. } else { false })
                            .collect::<Vec<_>>();
                        data
                    } else if let Some(data) = data.as_boolean_opt() {
                        let data = data
                            .iter()
                            .map(|x| x.unwrap_or_default())
                            .collect::<Vec<_>>();
                        data
                    } else {
                        println!("Got unexpected data type: {}", data.data_type());
                        continue;
                    };

                    let outputs: Vec<Vec<f32>> = masks
                        .chunks(height as usize * width as usize)
                        .filter_map(|data| {
                            let mut points = vec![];
                            let mut z_total = 0.;
                            let mut n = 0.;

                            if let Some(depth_frame) = &depth_frame {
                                depth_frame.iter().enumerate().for_each(|(i, z)| {
                                    let u = i as f32 % width as f32; // Calculate x-coordinate (u)
                                    let v = i as f32 / width as f32; // Calculate y-coordinate (v)

                                    if let Some(z) = z {
                                        let z = z as f32;
                                        // Skip points that have empty depth or is too far away
                                        if z == 0. || z > 1.0 {
                                            return;
                                        } 
                                        
                                        if data[i] {
                                            let x = (u - resolution[0] as f32) * z
                                                / focal_length[0] as f32;
                                            let y = (v - resolution[1] as f32) * z
                                                / focal_length[1] as f32;
                                            let new_x = x;
                                            let new_y = y;
                                            let new_z = z;

                                            points.push((new_x, new_y, new_z));
                                            z_total += new_z;
                                            n += 1.;
                                        }
                                    }
                                });
                            } else {
                                println!("No depth frame found");
                                return None;
                            }
                            if points.is_empty() {
                                println!("No points in mask found");
                                return None;
                            }
                            Some(points_to_pose(&points))
                        })
                        .collect();
                    let flatten_data = outputs.into_iter().flatten().collect::<Vec<_>>();
                    let mut metadata = metadata.parameters.clone();
                    metadata.insert(
                        "encoding".to_string(),
                        Parameter::String("xyzrpy".to_string()),
                    );
                    println!("Got data: {:?}", flatten_data);

                    node.send_output(
                        DataId::from("pose".to_string()),
                        metadata,
                        flatten_data.into_arrow(),
                    )?;
                }
                "boxes2d" => {
                    if let Some(data) = data.as_primitive_opt::<Int64Type>() {
                        let rgb_width = 1280;
                        let rgb_height = 720;
                        // println!("更新后的height: {}, width: {}", height, width);
                        let scale_x = width as f32 / rgb_width as f32;
                        let scale_y = height as f32 / rgb_height as f32;
                        println!("scale_x: {}, scale_y: {}", scale_x, scale_y);
                        let values = data.values();
                        let mut pose_data = vec![];
                        let outputs: Vec<Vec<f32>> = values
                            .chunks(4)
                            .filter_map(|data| {
                                // 原始边界坐标
                                let mut is_active: bool = true;
                                let x_min_orig = (data[0] as f32) * scale_x;
                                let y_min_orig = (data[1] as f32) * scale_y;
                                let x_max_orig = (data[2] as f32) * scale_x;
                                let y_max_orig = (data[3] as f32) * scale_y;
                                
                                // 计算中心点和缩小后的尺寸
                                let center_x = ((x_min_orig + x_max_orig) / 2.0 ) as usize;
                                let center_y = ((y_min_orig + y_max_orig) / 2.0 ) as usize;
                                let center_x = center_x as f32;
                                let center_y = center_y as f32;
                                let new_width = (x_max_orig - x_min_orig) / 2.0;  
                                let new_height = (y_max_orig - y_min_orig) / 2.0;
                                let new_width = (x_max_orig - x_min_orig) / 2.0;  
                                let new_height = (y_max_orig - y_min_orig) / 2.0;
                                let x_min = center_x - new_width / 2.0;
                                let x_max = center_x + new_width / 2.0;
                                let y_min = center_y - new_height / 2.0;
                                let y_max = center_y + new_height / 2.0;

                                let mut points = vec![];
                                let mut z_min = 100.;
                                let mut z_total = 0.;
                                let mut n = 0.;

                                if let Some(depth_frame) = &depth_frame {
                                    
                                    depth_frame.iter().enumerate().for_each(|(i, z)| {
                                        
                                        if i >= (width as usize * height as usize) {
                                            return;
                                        }
                                        
                                        let u =(i % width as usize) as f32; 
                                        let v =(i / width as usize) as f32;
                                        

                                        if let Some(z) = z {
                                            let z = z as f32;

                                            if (u>=center_x-1.0 && u<=center_x+1.0) && (v>=center_y-1.0 && v<=center_y+1.0) && is_active
                                            {
                                                println!("u: {}, v: {}", u, v);
                                                let x_test = (u - resolution[0] as f32) * z / focal_length[0] as f32;
                                                let y_test = (v - resolution[1] as f32) * z / focal_length[1] as f32;
                                                println!("x: {}, y: {}, z: {}", x_test, y_test, z);
                                                is_active = false;
                                                pose_data.extend(vec![
                                                    x_test, 
                                                    y_test, 
                                                    z, 
                                                    0.0, 
                                                    0.0, 
                                                    0.0
                                                ]);
                                            }

                                            
                                            if z == 0. || z > 1.0 {
                                                return;
                                            }
                                            if u > x_min && u < x_max && v > y_min && v < y_max {
                                                let x = (u - resolution[0] as f32) * z
                                                    / focal_length[0] as f32;
                                                let y = (v - resolution[1] as f32) * z
                                                    / focal_length[1] as f32;
                                                let new_x = x;
                                                let new_y = y;
                                                let new_z = z;
                                                if  new_z < z_min 
                                                {
                                                    z_min = new_z;
                                                }
                                                points.push((new_x, new_y, new_z));
                                                z_total += new_z;
                                                n += 1.;
                                            }
                                        }
                                    });
                                } else {
                                    println!("No depth frame found");
                                    return None;
                                }
                                if points.is_empty() {
                                    return None;
                                }
                                let raw_mean_z = z_total / n as f32;
                                let threshold = (raw_mean_z + z_min) / 2.;
                                let points = points
                                    .into_iter()
                                    .filter(|(_x, _y, z)| z > &threshold)
                                    .collect::<Vec<_>>();
                                Some(points_to_pose(&points))
                            })
                            .collect();
                        // let flatten_data = pose_data.clone();
                        let flatten_data = outputs.into_iter().flatten().collect::<Vec<_>>();
                        // let flatten_data = pose_data.into_iter().flatten().collect::<Vec<_>>();
                        // let flatten_data_test=pose_data.into_iter().flatten().collect::<Vec<_>>();
                        // println!("pose_data: {:?}", pose_data);
                        println!("Got data: {:?}", flatten_data);
                        let mut metadata = metadata.parameters.clone();
                        metadata.insert(
                            "encoding".to_string(),
                            Parameter::String("xyzrpy".to_string()),
                        );

                        node.send_output(
                            DataId::from("pose".to_string()),
                            metadata,
                            flatten_data.into_arrow(),
                        )?;
                    }
                }
                other => eprintln!("Received input `{other}`"),
            }
        }
    }

    Ok(())
}

#[cfg(feature = "python")]
use pyo3::{
    pyfunction, pymodule,
    types::{PyModule, PyModuleMethods},
    wrap_pyfunction, Bound, PyResult, Python,
};

#[cfg(feature = "python")]
#[pyfunction]
fn py_main(_py: Python) -> eyre::Result<()> {
    lib_main()
}

/// A Python module implemented in Rust.
#[cfg(feature = "python")]
#[pymodule]
fn dora_object_to_pose(_py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_main, &m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}