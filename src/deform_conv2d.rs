//! DeformableConv2d wrapper: generates offsets and modulation mask internally,
//! then delegates to `candle_dcnv2::deform_conv2d`.
//!
//! Reference: BiRefNet `models/modules/deform_conv.py`

use candle_core::{Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder};

/// Deformable Convolution v2 with learned offset and modulation.
///
/// Wraps three Conv2d layers (offset, modulator, regular) and calls
/// `candle_dcnv2::deform_conv2d` in the forward pass.
pub struct DeformableConv2d {
    offset_conv: Conv2d,
    modulator_conv: Conv2d,
    weight: Tensor,
    bias: Option<Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
}

impl DeformableConv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        use_bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let kk: usize = kernel_size * kernel_size;
        let conv_cfg = Conv2dConfig {
            stride,
            padding,
            ..Default::default()
        };

        let offset_conv = candle_nn::conv2d(
            in_channels,
            2 * kk,
            kernel_size,
            conv_cfg,
            vb.pp("offset_conv"),
        )?;
        let modulator_conv = candle_nn::conv2d(
            in_channels,
            kk,
            kernel_size,
            conv_cfg,
            vb.pp("modulator_conv"),
        )?;

        let weight = vb.get(
            (out_channels, in_channels, kernel_size, kernel_size),
            "regular_conv.weight",
        )?;
        let bias = if use_bias {
            Some(vb.get(out_channels, "regular_conv.bias")?)
        } else {
            None
        };

        Ok(Self {
            offset_conv,
            modulator_conv,
            weight,
            bias,
            stride: (stride, stride),
            padding: (padding, padding),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let offset: Tensor = self.offset_conv.forward(x)?;
        // modulator = 2 * sigmoid(conv(x))
        let modulator: Tensor = (candle_nn::ops::sigmoid(&self.modulator_conv.forward(x)?)? * 2.0)?;

        candle_dcnv2::deform_conv2d(
            x,
            &offset,
            &self.weight,
            self.bias.as_ref(),
            Some(&modulator),
            self.stride,
            self.padding,
            (1, 1),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn test_deform_conv2d_output_shape() -> Result<()> {
        let device = &Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

        let dcn = DeformableConv2d::new(64, 128, 3, 1, 1, false, vb)?;
        let input = Tensor::randn(0f32, 1.0, (1, 64, 16, 16), device)?;
        let output = dcn.forward(&input)?;

        assert_eq!(output.dims(), &[1, 128, 16, 16]);
        Ok(())
    }

    #[test]
    fn test_deform_conv2d_kernel_1x1() -> Result<()> {
        let device = &Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

        let dcn = DeformableConv2d::new(32, 64, 1, 1, 0, false, vb)?;
        let input = Tensor::randn(0f32, 1.0, (1, 32, 8, 8), device)?;
        let output = dcn.forward(&input)?;

        assert_eq!(output.dims(), &[1, 64, 8, 8]);
        Ok(())
    }

    #[test]
    fn test_deform_conv2d_kernel_7x7() -> Result<()> {
        let device = &Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

        let dcn = DeformableConv2d::new(32, 64, 7, 1, 3, false, vb)?;
        let input = Tensor::randn(0f32, 1.0, (1, 32, 8, 8), device)?;
        let output = dcn.forward(&input)?;

        assert_eq!(output.dims(), &[1, 64, 8, 8]);
        Ok(())
    }
}
