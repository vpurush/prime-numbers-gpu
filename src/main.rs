extern crate ocl;

use std::time::Instant;

use ocl::{flags, Buffer, Context, Device, Kernel, Platform, ProQue, Program, Queue};

fn main() {
    let now = Instant::now();

    let platforms = Platform::list();

    let platform = platforms[1];

    let device = Device::first(platform).unwrap();

    println!("Platforms {:?}", platforms[1].name());
    println!("Platform {}", platform.name().unwrap());
    println!("Device {}", device.name().unwrap());

    let src = r#"
        __kernel void mod(__global float* inputOne, __global float* inputTwo, __global float* output) {
            
            int xint = (int)(inputOne[get_global_id(0)]);
            int yint = (int)(inputTwo[get_global_id(0)]);
            output[get_global_id(0)] = xint % yint;
        }
    "#;

    let context = Context::builder()
        .platform(platform)
        .devices(device.clone())
        .build()
        .unwrap();

    let program = Program::builder()
        .devices(device)
        .src(src)
        .build(&context)
        .unwrap();

    let queue = Queue::new(&context, device, None).unwrap();

    // let host_input_one: Vec<f32> = vec![3.0, 4.0, 4.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0];
    // let host_input_two: Vec<f32> = vec![2.0, 2.0, 3.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0];

    let mut host_input_one: Vec<f32> = vec![];
    let mut host_input_two: Vec<f32> = vec![];

    for x in 2..100000 {
        for y in 2..x {
            host_input_one.push(x as f32);
            host_input_two.push(y as f32);
        }
    }

    let dims: usize = host_input_one.len();

    // host_input_one.
    let input_one = Buffer::<f32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_ONLY)
        .len(dims)
        .copy_host_slice(host_input_one.as_slice())
        // .fill_val(0f32)
        .build()
        .unwrap();

    let input_two = Buffer::<f32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_ONLY)
        .len(dims)
        .copy_host_slice(host_input_two.as_slice())
        .build()
        .unwrap();

    let output = Buffer::<f32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_ONLY)
        .len(dims)
        .fill_val(0.0f32)
        .build()
        .unwrap();

    let kernel = Kernel::builder()
        .program(&program)
        .name("mod")
        .queue(queue.clone())
        .global_work_size(dims)
        .arg(&input_one)
        .arg(&input_two)
        .arg(&output)
        .build()
        .unwrap();

    unsafe {
        kernel
            .cmd()
            .queue(&queue)
            .global_work_offset(kernel.default_global_work_offset())
            .global_work_size(dims)
            .local_work_size(kernel.default_local_work_size())
            .enq()
            .unwrap();
    }

    let mut output_vec = vec![0.0f32; dims];
    output
        .cmd()
        .queue(&queue)
        .offset(0)
        .read(&mut output_vec)
        .enq()
        .unwrap();

    let mut prime_numbers = vec![];
    let mut current_integer: f32 = -1.0;
    let mut current_integer_is_prime = true;
    output_vec.into_iter().enumerate().for_each(|(i, elm)| {
        let integer = host_input_one[i];
        if current_integer != -1.0 && current_integer != integer {
            if current_integer_is_prime {
                // println!("{} is prime", current_integer);
                prime_numbers.push(current_integer);
            } else {
                // println!("{} is not prime", current_integer);
            }

            current_integer_is_prime = true;
        }

        if elm == 0.0 {
            current_integer_is_prime = false;
        }

        current_integer = integer;
    });

    let elapsed = now.elapsed();
    println!("Primes are {:?}", prime_numbers);
    println!("Time to execute {:?}", elapsed);
    // Print an element:
    // println!("output_vec {:?}", output_vec);
}
