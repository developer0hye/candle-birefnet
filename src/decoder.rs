//! Decoder building blocks: BasicDecBlk, BasicLatBlk, SimpleConvs.
//!
//! Reference: BiRefNet `models/modules/decoder_blocks.py`, `lateral_blocks.py`, `birefnet.py`

use candle_core::{Module, ModuleT, Result, Tensor};
use candle_nn::{BatchNorm, Conv2d, Conv2dConfig, VarBuilder};

use crate::aspp::ASPPDeformable;

/// BasicDecBlk: Conv(3x3) → BN → ReLU → ASPPDeformable → Conv(3x3) → BN
pub struct BasicDecBlk {
    conv_in: Conv2d,
    bn_in: Option<BatchNorm>,
    dec_att: ASPPDeformable,
    conv_out: Conv2d,
    bn_out: Option<BatchNorm>,
}

impl BasicDecBlk {
    pub fn new(in_channels: usize, out_channels: usize, vb: VarBuilder) -> Result<Self> {
        let inter_channels: usize = 64; // config.dec_channels_inter == 'fixed'
        let cfg3x3 = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };

        let conv_in = candle_nn::conv2d(in_channels, inter_channels, 3, cfg3x3, vb.pp("conv_in"))?;
        let bn_in = candle_nn::batch_norm(inter_channels, 1e-5, vb.pp("bn_in")).ok();
        let dec_att = ASPPDeformable::new(inter_channels, inter_channels, vb.pp("dec_att"))?;
        let conv_out =
            candle_nn::conv2d(inter_channels, out_channels, 3, cfg3x3, vb.pp("conv_out"))?;
        let bn_out = candle_nn::batch_norm(out_channels, 1e-5, vb.pp("bn_out")).ok();

        Ok(Self {
            conv_in,
            bn_in,
            dec_att,
            conv_out,
            bn_out,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x: Tensor = self.conv_in.forward(x)?;
        let x: Tensor = if let Some(ref bn) = self.bn_in {
            bn.forward_t(&x, false)?
        } else {
            x
        };
        let x: Tensor = x.relu()?;
        let x: Tensor = self.dec_att.forward(&x)?;
        let x: Tensor = self.conv_out.forward(&x)?;
        if let Some(ref bn) = self.bn_out {
            bn.forward_t(&x, false)
        } else {
            Ok(x)
        }
    }
}

/// BasicLatBlk: Conv2d(1x1) channel projection.
pub struct BasicLatBlk {
    conv: Conv2d,
}

impl BasicLatBlk {
    pub fn new(in_channels: usize, out_channels: usize, vb: VarBuilder) -> Result<Self> {
        let conv = candle_nn::conv2d(
            in_channels,
            out_channels,
            1,
            Default::default(),
            vb.pp("conv"),
        )?;
        Ok(Self { conv })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.conv.forward(x)
    }
}

/// SimpleConvs: Conv(3x3) → Conv(3x3) for decoder input processing.
pub struct SimpleConvs {
    conv1: Conv2d,
    conv_out: Conv2d,
}

impl SimpleConvs {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        inter_channels: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv1 = candle_nn::conv2d(in_channels, inter_channels, 3, cfg, vb.pp("conv1"))?;
        let conv_out = candle_nn::conv2d(inter_channels, out_channels, 3, cfg, vb.pp("conv_out"))?;
        Ok(Self { conv1, conv_out })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x: Tensor = self.conv1.forward(x)?;
        self.conv_out.forward(&x)
    }

    /// Access conv1 weight for debugging.
    pub fn conv1_weight(&self) -> Result<Tensor> {
        Ok(self.conv1.weight().clone())
    }

    /// Access conv_out weight for debugging.
    pub fn conv_out_weight(&self) -> Result<Tensor> {
        Ok(self.conv_out.weight().clone())
    }
}

/// Gradient-attention module used in decoder when `out_ref=True`.
/// Conv(3x3) → BN → ReLU for feature extraction, then two 1x1 conv heads
/// for prediction and attention gating.
pub struct GradientAttention {
    convs: Conv2d,
    bn: Option<BatchNorm>,
    convs_attn: Conv2d,
}

impl GradientAttention {
    /// `suffix` corresponds to the decoder stage number (e.g., "4", "3", "2").
    /// Keys will map to PyTorch's `gdt_convs_{suffix}.0`, `gdt_convs_attn_{suffix}.0`.
    pub fn new(in_channels: usize, suffix: &str, vb: VarBuilder) -> Result<Self> {
        let n: usize = 16;
        let cfg3 = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let convs = candle_nn::conv2d(
            in_channels,
            n,
            3,
            cfg3,
            vb.pp(format!("gdt_convs_{suffix}.0")),
        )?;
        let bn = candle_nn::batch_norm(n, 1e-5, vb.pp(format!("gdt_convs_{suffix}.1"))).ok();
        let convs_attn = candle_nn::conv2d(
            n,
            1,
            1,
            Default::default(),
            vb.pp(format!("gdt_convs_attn_{suffix}.0")),
        )?;
        Ok(Self {
            convs,
            bn,
            convs_attn,
        })
    }

    /// Returns sigmoid attention map `[B, 1, H, W]` to multiply with decoder features.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x: Tensor = self.convs.forward(x)?;
        let x: Tensor = if let Some(ref bn) = self.bn {
            bn.forward_t(&x, false)?
        } else {
            x
        };
        let x: Tensor = x.relu()?;
        candle_nn::ops::sigmoid(&self.convs_attn.forward(&x)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn test_basic_dec_blk_shape() -> Result<()> {
        let device = &Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

        let blk = BasicDecBlk::new(128, 64, vb)?;
        let input = Tensor::randn(0f32, 1.0, (1, 128, 8, 8), device)?;
        let output = blk.forward(&input)?;

        assert_eq!(output.dims(), &[1, 64, 8, 8]);
        Ok(())
    }

    #[test]
    fn test_basic_lat_blk_shape() -> Result<()> {
        let device = &Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

        let blk = BasicLatBlk::new(256, 128, vb)?;
        let input = Tensor::randn(0f32, 1.0, (1, 256, 16, 16), device)?;
        let output = blk.forward(&input)?;

        assert_eq!(output.dims(), &[1, 128, 16, 16]);
        Ok(())
    }

    #[test]
    fn test_simple_convs_shape() -> Result<()> {
        let device = &Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

        let sc = SimpleConvs::new(192, 48, 64, vb)?;
        let input = Tensor::randn(0f32, 1.0, (1, 192, 8, 8), device)?;
        let output = sc.forward(&input)?;

        assert_eq!(output.dims(), &[1, 48, 8, 8]);
        Ok(())
    }
}
