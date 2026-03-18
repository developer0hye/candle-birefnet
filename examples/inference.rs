//! Example: BiRefNet inference on a real image.
//!
//! Downloads the pretrained model from HuggingFace and runs segmentation.
//!
//! Usage:
//!   cargo run --example inference --release -- --image examples/Helicopter.jpg

use anyhow::Result;
use candle_birefnet::BiRefNet;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use image::{ImageBuffer, Luma, RgbImage};

fn load_and_preprocess(path: &str, size: (u32, u32)) -> Result<(Tensor, RgbImage)> {
    let img = image::open(path)?.resize_exact(size.0, size.1, image::imageops::Triangle);
    let rgb = img.to_rgb8();

    let (w, h) = (size.0 as usize, size.1 as usize);
    let mut data = vec![0f32; 3 * h * w];

    // ImageNet normalization
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];

    for y in 0..h {
        for x in 0..w {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                data[c * h * w + y * w + x] = (pixel[c] as f32 / 255.0 - mean[c]) / std[c];
            }
        }
    }

    let tensor = Tensor::from_vec(data, (1, 3, h, w), &Device::Cpu)?;
    Ok((tensor, rgb))
}

fn mask_to_image(mask: &Tensor, h: u32, w: u32) -> Result<ImageBuffer<Luma<u8>, Vec<u8>>> {
    // Apply sigmoid
    let mask = candle_nn::ops::sigmoid(mask)?;
    let mask = mask.squeeze(0)?.squeeze(0)?; // [H, W]
    let data = mask.to_vec2::<f32>()?;

    let mut img = ImageBuffer::new(w, h);
    for y in 0..h as usize {
        for x in 0..w as usize {
            let v = (data[y][x] * 255.0).clamp(0.0, 255.0) as u8;
            img.put_pixel(x as u32, y as u32, Luma([v]));
        }
    }
    Ok(img)
}

fn create_comparison(
    input: &RgbImage,
    mask: &ImageBuffer<Luma<u8>, Vec<u8>>,
    output_path: &str,
) -> Result<()> {
    let (w, h) = (input.width(), input.height());

    // Side by side: Input | Mask | Composite
    let mut canvas = RgbImage::new(w * 3, h);

    // Input
    for y in 0..h {
        for x in 0..w {
            canvas.put_pixel(x, y, *input.get_pixel(x, y));
        }
    }

    // Mask (grayscale -> RGB)
    for y in 0..h {
        for x in 0..w {
            let v = mask.get_pixel(x, y)[0];
            canvas.put_pixel(x + w, y, image::Rgb([v, v, v]));
        }
    }

    // Composite (foreground on white)
    for y in 0..h {
        for x in 0..w {
            let alpha = mask.get_pixel(x, y)[0] as f32 / 255.0;
            let fg = input.get_pixel(x, y);
            let r = (fg[0] as f32 * alpha + 255.0 * (1.0 - alpha)) as u8;
            let g = (fg[1] as f32 * alpha + 255.0 * (1.0 - alpha)) as u8;
            let b = (fg[2] as f32 * alpha + 255.0 * (1.0 - alpha)) as u8;
            canvas.put_pixel(x + w * 2, y, image::Rgb([r, g, b]));
        }
    }

    canvas.save(output_path)?;
    println!("Saved: {output_path}");
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let image_path = args
        .iter()
        .position(|a| a == "--image")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("examples/Helicopter.jpg");

    let res_str = args
        .iter()
        .position(|a| a == "--size")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("1024");
    let res: u32 = res_str.parse().unwrap_or(1024);
    let size: (u32, u32) = (res, res);

    let is_lite = args.iter().any(|a| a == "--lite");

    // Download model from HuggingFace
    let model_id = if is_lite {
        "ZhengPeng7/BiRefNet_lite"
    } else {
        "ZhengPeng7/BiRefNet"
    };
    println!("Loading model from HuggingFace ({model_id})...");
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model(model_id.to_string());
    let model_path = repo.get("model.safetensors")?;

    let device = &Device::Cpu;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, device)? };
    let model = if is_lite {
        BiRefNet::new_lite(vb)?
    } else {
        BiRefNet::new(vb)?
    };

    // Preprocess
    println!("Processing: {image_path} at {}x{}", size.0, size.1);
    let (input, input_img) = load_and_preprocess(image_path, size)?;

    // Inference
    println!("Running inference...");
    let outputs = model.forward(&input)?;
    let mask_tensor = &outputs[0];

    // Postprocess
    let mask_img = mask_to_image(mask_tensor, size.1, size.0)?;

    let stem = std::path::Path::new(image_path)
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap()
        .to_lowercase();

    let variant = if is_lite { "_lite" } else { "" };
    let mask_path = format!("examples/{stem}_mask_candle{variant}_{res_str}.png");
    mask_img.save(&mask_path)?;
    println!("Mask saved: {mask_path}");

    let result_path = format!("examples/{stem}_result_candle{variant}_{res_str}.png");
    create_comparison(&input_img, &mask_img, &result_path)?;

    println!("Done!");
    Ok(())
}
