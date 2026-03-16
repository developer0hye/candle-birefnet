//! Debug test: compare intermediate outputs at 1024x1024.

use candle_core::{DType, Device, Module, Result, Tensor, Var};
use candle_nn::{VarBuilder, VarMap};
use std::collections::HashMap;

fn load_test_case(name: &str) -> HashMap<String, Tensor> {
    let path = format!("test-data/{name}.safetensors");
    let data = std::fs::read(&path).unwrap_or_else(|_| panic!("missing test data: {path}"));
    let tensors = safetensors::SafeTensors::deserialize(&data).unwrap();
    let device = &Device::Cpu;
    tensors
        .tensors()
        .into_iter()
        .map(|(name, view)| {
            let dtype = match view.dtype() {
                safetensors::Dtype::F32 => DType::F32,
                safetensors::Dtype::F64 => DType::F64,
                safetensors::Dtype::I64 => DType::I64,
                safetensors::Dtype::U32 => DType::U32,
                dt => panic!("unsupported dtype {dt:?} for tensor {name}"),
            };
            let tensor = Tensor::from_raw_buffer(view.data(), dtype, view.shape(), device).unwrap();
            (name.to_string(), tensor)
        })
        .collect()
}

fn compare(label: &str, actual: &Tensor, expected: &Tensor) {
    let diff = actual
        .to_dtype(DType::F64)
        .unwrap()
        .sub(&expected.to_dtype(DType::F64).unwrap())
        .unwrap()
        .abs()
        .unwrap();
    let max_diff: f64 = diff
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f64>()
        .unwrap();
    let mean_diff: f64 = diff.mean_all().unwrap().to_scalar::<f64>().unwrap();
    let status = if max_diff < 1e-2 { "OK" } else { "DIVERGED" };
    eprintln!(
        "  {label}: shape={:?} max_diff={max_diff:.4e} mean_diff={mean_diff:.4e} [{status}]",
        actual.dims()
    );
}

#[test]
fn debug_birefnet_1024_intermediates() -> Result<()> {
    // Load model weights
    let model_data = load_test_case("birefnet_1024");
    let varmap = VarMap::new();
    {
        let mut dm = varmap.data().lock().unwrap();
        for (k, t) in &model_data {
            if let Some(pn) = k.strip_prefix("param.") {
                dm.insert(pn.to_string(), Var::from_tensor(t).unwrap());
            }
        }
    }
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

    // Load intermediates from PyTorch
    let inter = load_test_case("birefnet_1024_intermediates");
    let x = &model_data["input"];

    eprintln!("\n=== BiRefNet 1024x1024 Debug ===\n");

    // 1. Swin backbone
    let swin_config = candle_swin::swin_transformer::SwinTransformerConfig::large();
    let bb = candle_swin::SwinTransformer::new(&swin_config, vb.pp("bb"))?;
    let features = bb.forward(x)?;

    let x1 = &features[0];
    let x4 = &features[3];

    compare("backbone x1", x1, &inter["x1"].narrow(1, 0, 192)?);
    compare(
        "backbone x4 (pre-cat)",
        x4,
        &inter["x4_after_squeeze"].narrow(1, 0, 1536)?,
    );

    // 2. Multi-scale + cat
    let (_, _, h, w) = x.dims4()?;
    let x_half = x.upsample_bilinear2d(h / 2, w / 2, true)?;
    let features_half = bb.forward(&x_half)?;

    let x1 = Tensor::cat(
        &[
            &features[0],
            &features_half[0].upsample_bilinear2d(
                features[0].dims()[2],
                features[0].dims()[3],
                true,
            )?,
        ],
        1,
    )?;
    let x2 = Tensor::cat(
        &[
            &features[1],
            &features_half[1].upsample_bilinear2d(
                features[1].dims()[2],
                features[1].dims()[3],
                true,
            )?,
        ],
        1,
    )?;
    let x3 = Tensor::cat(
        &[
            &features[2],
            &features_half[2].upsample_bilinear2d(
                features[2].dims()[2],
                features[2].dims()[3],
                true,
            )?,
        ],
        1,
    )?;
    let x4 = Tensor::cat(
        &[
            &features[3],
            &features_half[3].upsample_bilinear2d(
                features[3].dims()[2],
                features[3].dims()[3],
                true,
            )?,
        ],
        1,
    )?;

    compare("x1 after cat", &x1, &inter["x1"]);
    compare(
        "x4 after cat (pre-ctx)",
        &x4,
        &inter["x4_after_squeeze"].narrow(1, 0, x4.dims()[1])?,
    );

    // 3. Context augmentation
    let x4_h = x4.dims()[2];
    let x4_w = x4.dims()[3];
    let x4 = Tensor::cat(
        &[
            &x1.upsample_bilinear2d(x4_h, x4_w, true)?,
            &x2.upsample_bilinear2d(x4_h, x4_w, true)?,
            &x3.upsample_bilinear2d(x4_h, x4_w, true)?,
            &x4,
        ],
        1,
    )?;
    eprintln!("  x4 after context: {:?} ch={}", x4.dims(), x4.dims()[1]);

    // 4. Squeeze
    let squeeze =
        candle_birefnet::decoder::BasicDecBlk::new(5760, 3072, vb.pp("squeeze_module.0"))?;
    let x4 = squeeze.forward(&x4)?;
    compare("x4 after squeeze", &x4, &inter["x4_after_squeeze"]);

    // 5. image2patches
    let grid_h = h / x4.dims()[2];
    let grid_w = w / x4.dims()[3];
    eprintln!("  grid for patches: {grid_h}x{grid_w}");
    let patches = candle_birefnet::birefnet::image2patches(x, grid_h, grid_w)?;
    compare("patches_stage4", &patches, &inter["patches_stage4"]);

    // 6. Stage 4 decoder block internals
    let stage4 = load_test_case("birefnet_1024_stage4");

    let ipt5 =
        candle_birefnet::decoder::SimpleConvs::new(3072, 384, 64, vb.pp("decoder.ipt_blk5"))?;
    let ipt_feat =
        ipt5.forward(&patches.upsample_bilinear2d(x4.dims()[2], x4.dims()[3], true)?)?;
    let x4_cat = Tensor::cat(&[&x4, &ipt_feat], 1)?;
    compare("x4_cat (before decoder_block4)", &x4_cat, &stage4["x4_cat"]);

    // BasicDecBlk internals
    let dec4 =
        candle_birefnet::decoder::BasicDecBlk::new(3456, 1536, vb.pp("decoder.decoder_block4"))?;
    let p4 = dec4.forward(&x4_cat)?;
    compare("p4 (decoder_block4 output)", &p4, &stage4["p4"]);

    // 7. GDT attention on p4
    let gdt4 = candle_birefnet::decoder::GradientAttention::new(1536, "4", vb.pp("decoder"))?;
    let attn_4 = gdt4.forward(&p4)?;
    let p4 = p4.broadcast_mul(&attn_4)?;

    let stage3 = load_test_case("birefnet_1024_stage3");
    compare("p4 after gdt", &p4, &stage3["p4_after_gdt"]);

    // 8. Upsample p4 to x3 size
    let p4_up = p4.upsample_bilinear2d(x3.dims()[2], x3.dims()[3], true)?;
    compare("p4 upsampled to x3", &p4_up, &stage3["p4_up"]);

    // 9. Lateral + add
    let lat4 =
        candle_birefnet::decoder::BasicLatBlk::new(1536, 1536, vb.pp("decoder.lateral_block4"))?;
    let lat4_out = lat4.forward(&x3)?;
    compare("lateral_block4(x3)", &lat4_out, &stage3["lat4"]);

    let _p3 = (p4_up + lat4_out)?;
    compare("_p3 before cat", &_p3, &stage3["_p3_before_cat"]);

    // 10. Stage 3 decoder
    let ipt4 = candle_birefnet::decoder::SimpleConvs::new(768, 384, 64, vb.pp("decoder.ipt_blk4"))?;
    let patches3 =
        candle_birefnet::birefnet::image2patches(x, h / _p3.dims()[2], w / _p3.dims()[3])?;
    let ipt_feat3 =
        ipt4.forward(&patches3.upsample_bilinear2d(_p3.dims()[2], _p3.dims()[3], true)?)?;
    let _p3 = Tensor::cat(&[&_p3, &ipt_feat3], 1)?;
    // Stage 3: compare patches3 directly
    let patches3_ref = load_test_case("birefnet_1024_patches3");
    compare(
        "patches3 (image2patches)",
        &patches3,
        &patches3_ref["patches3"],
    );

    let patches3_interp = patches3.upsample_bilinear2d(_p3.dims()[2], _p3.dims()[3], true)?;
    eprintln!("  patches3_interp shape: {:?}", patches3_interp.dims());

    // Stage 3 decoder block internals
    let detail3 = load_test_case("birefnet_1024_stage3_detail");
    // Verify ipt_blk4 weights loaded correctly
    {
        let dm = varmap.data().lock().unwrap();
        let w = dm.get("decoder.ipt_blk4.conv1.weight").unwrap();
        let expected_w = &model_data["param.decoder.ipt_blk4.conv1.weight"];
        let wdiff = w
            .as_tensor()
            .to_dtype(DType::F64)
            .unwrap()
            .sub(&expected_w.to_dtype(DType::F64).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f64>()
            .unwrap();
        eprintln!("  ipt_blk4.conv1.weight diff: {wdiff:.6e}");
    }
    // Run ipt_blk4 step by step
    let ipt4_conv1 = candle_nn::conv2d(
        768,
        64,
        3,
        candle_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        },
        vb.pp("decoder.ipt_blk4.conv1"),
    )?;
    let ipt4_conv_out = candle_nn::conv2d(
        64,
        384,
        3,
        candle_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        },
        vb.pp("decoder.ipt_blk4.conv_out"),
    )?;
    let step1 = ipt4_conv1.forward(&patches3)?;
    eprintln!(
        "  ipt_blk4 conv1 output: shape={:?} abs_mean={:.6}",
        step1.dims(),
        step1
            .to_dtype(DType::F64)?
            .abs()?
            .mean_all()?
            .to_scalar::<f64>()?
    );

    let step2 = ipt4_conv_out.forward(&step1)?;
    eprintln!(
        "  ipt_blk4 conv_out output: shape={:?} abs_mean={:.6}",
        step2.dims(),
        step2
            .to_dtype(DType::F64)?
            .abs()?
            .mean_all()?
            .to_scalar::<f64>()?
    );
    compare("ipt_feat3 (manual)", &step2, &detail3["ipt_feat3"]);
    compare("_p3 after cat", &_p3, &detail3["_p3_cat"]);

    // Manually run decoder_block3 conv_in + bn + relu
    use candle_core::ModuleT;
    let conv_in = candle_nn::conv2d(
        1920,
        64,
        3,
        candle_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        },
        vb.pp("decoder.decoder_block3.conv_in"),
    )?;
    let bn_in = candle_nn::batch_norm(64, 1e-5, vb.pp("decoder.decoder_block3.bn_in"))?;
    let t = conv_in.forward(&_p3)?;
    let t = bn_in.forward_t(&t, false)?;
    let t = t.relu()?;
    compare(
        "dec3 after conv_in+bn+relu",
        &t,
        &detail3["after_conv_in_bn_relu"],
    );

    // ASPP at 64x64 spatial
    let aspp3 = candle_birefnet::aspp::ASPPDeformable::new(
        64,
        64,
        vb.pp("decoder.decoder_block3.dec_att"),
    )?;
    let t_aspp = aspp3.forward(&t)?;
    compare("dec3 after ASPP", &t_aspp, &detail3["after_aspp"]);

    eprintln!("\n=== Done ===");
    Ok(())
}
