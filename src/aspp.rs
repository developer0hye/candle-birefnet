//! ASPPDeformable: Atrous Spatial Pyramid Pooling with deformable convolutions.
//!
//! Reference: BiRefNet `models/modules/aspp.py`

use candle_core::{Module, ModuleT, Result, Tensor};
use candle_nn::{BatchNorm, Conv2d, Conv2dConfig, Dropout, VarBuilder};

use crate::deform_conv2d::DeformableConv2d;

/// Single ASPP branch with DeformableConv2d.
struct ASPPModuleDeformable {
    atrous_conv: DeformableConv2d,
    bn: Option<BatchNorm>,
}

impl ASPPModuleDeformable {
    fn new(
        in_channels: usize,
        planes: usize,
        kernel_size: usize,
        padding: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let atrous_conv = DeformableConv2d::new(
            in_channels,
            planes,
            kernel_size,
            1,
            padding,
            false,
            vb.pp("atrous_conv"),
        )?;
        // BiRefNet uses BN when batch_size > 1, Identity otherwise.
        // For inference with batch_size=1, skip BN. Load it if weights exist.
        let bn = candle_nn::batch_norm(planes, 1e-5, vb.pp("bn")).ok();
        Ok(Self { atrous_conv, bn })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x: Tensor = self.atrous_conv.forward(x)?;
        let x: Tensor = if let Some(ref bn) = self.bn {
            bn.forward_t(&x, false)?
        } else {
            x
        };
        x.relu()
    }
}

/// ASPP with deformable convolution branches.
///
/// Architecture:
/// - 1 deformable conv (k=1)
/// - N deformable convs (k=1,3,7 by default)
/// - 1 global average pooling branch
/// - Concat all → Conv(1x1) → BN → ReLU → Dropout
pub struct ASPPDeformable {
    aspp1: ASPPModuleDeformable,
    aspp_deforms: Vec<ASPPModuleDeformable>,
    global_avg_pool_conv: Conv2d,
    global_avg_pool_bn: Option<BatchNorm>,
    conv1: Conv2d,
    bn1: Option<BatchNorm>,
    dropout: Dropout,
}

impl ASPPDeformable {
    pub fn new(in_channels: usize, out_channels: usize, vb: VarBuilder) -> Result<Self> {
        let inter: usize = 256;
        let parallel_block_sizes: Vec<usize> = vec![1, 3, 7];

        let aspp1 = ASPPModuleDeformable::new(in_channels, inter, 1, 0, vb.pp("aspp1"))?;

        let mut aspp_deforms: Vec<ASPPModuleDeformable> = Vec::new();
        for (i, &ks) in parallel_block_sizes.iter().enumerate() {
            let padding: usize = ks / 2;
            aspp_deforms.push(ASPPModuleDeformable::new(
                in_channels,
                inter,
                ks,
                padding,
                vb.pp(format!("aspp_deforms.{i}")),
            )?);
        }

        let gap_cfg = Conv2dConfig::default();
        let global_avg_pool_conv =
            candle_nn::conv2d_no_bias(in_channels, inter, 1, gap_cfg, vb.pp("global_avg_pool.1"))?;
        let global_avg_pool_bn =
            candle_nn::batch_norm(inter, 1e-5, vb.pp("global_avg_pool.2")).ok();

        let num_branches: usize = 2 + parallel_block_sizes.len(); // aspp1 + deforms + gap
        let conv1 = candle_nn::conv2d_no_bias(
            inter * num_branches,
            out_channels,
            1,
            gap_cfg,
            vb.pp("conv1"),
        )?;
        let bn1 = candle_nn::batch_norm(out_channels, 1e-5, vb.pp("bn1")).ok();

        Ok(Self {
            aspp1,
            aspp_deforms,
            global_avg_pool_conv,
            global_avg_pool_bn,
            conv1,
            bn1,
            dropout: Dropout::new(0.5),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x1: Tensor = self.aspp1.forward(x)?;

        let mut branches: Vec<Tensor> = vec![x1.clone()];
        for deform in &self.aspp_deforms {
            branches.push(deform.forward(x)?);
        }

        // Global average pooling: mean over spatial dims → conv → bn → relu → upsample
        let (_, _, _h, _w) = x.dims4()?;
        let x5: Tensor = x.mean_keepdim(2)?.mean_keepdim(3)?; // [B, C, 1, 1]
        let x5: Tensor = self.global_avg_pool_conv.forward(&x5)?;
        let x5: Tensor = if let Some(ref bn) = self.global_avg_pool_bn {
            bn.forward_t(&x5, false)?
        } else {
            x5
        };
        let x5: Tensor = x5.relu()?;
        let (target_h, target_w) = (x1.dims()[2], x1.dims()[3]);
        let x5: Tensor = x5.upsample_bilinear2d(target_h, target_w, true)?;
        branches.push(x5);

        let branch_refs: Vec<&Tensor> = branches.iter().collect();
        let x: Tensor = Tensor::cat(&branch_refs, 1)?;

        let x: Tensor = self.conv1.forward(&x)?;
        let x: Tensor = if let Some(ref bn) = self.bn1 {
            bn.forward_t(&x, false)?
        } else {
            x
        };
        let x: Tensor = x.relu()?;
        self.dropout.forward(&x, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn test_aspp_deformable_output_shape() -> Result<()> {
        let device = &Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

        let aspp = ASPPDeformable::new(64, 64, vb)?;
        let input = Tensor::randn(0f32, 1.0, (1, 64, 8, 8), device)?;
        let output = aspp.forward(&input)?;

        assert_eq!(output.dims(), &[1, 64, 8, 8]);
        Ok(())
    }
}
