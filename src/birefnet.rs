//! BiRefNet: Bilateral Reference Network for image segmentation.
//!
//! Inference-only implementation targeting the default config:
//! - Backbone: Swin-V1-Large
//! - Multi-scale input: cat mode
//! - Decoder attention: ASPPDeformable
//! - Squeeze block: BasicDecBlk x1
//! - Decoder input patching: split mode
//! - Gradient attention: enabled (out_ref=True)
//!
//! Reference: https://github.com/ZhengPeng7/BiRefNet

use candle_core::{Module, Result, Tensor};
use candle_nn::{Conv2d, VarBuilder};
use candle_swin::swin_transformer::{SwinTransformer, SwinTransformerConfig};

use crate::decoder::{BasicDecBlk, BasicLatBlk, GradientAttention, SimpleConvs};

/// Splits image into a grid of patches and merges channels.
///
/// `[B, C, H, W]` → `[B, C*grid_h*grid_w, H/grid_h, W/grid_w]`
///
/// Equivalent to: `einops.rearrange(x, 'b c (hg h) (wg w) -> b (c hg wg) h w', hg=gh, wg=gw)`
pub fn image2patches(x: &Tensor, grid_h: usize, grid_w: usize) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let ph: usize = h / grid_h;
    let pw: usize = w / grid_w;
    // [B, C, gh, ph, gw, pw] → [B, C, gh, gw, ph, pw] → [B, C*gh*gw, ph, pw]
    x.reshape((b, c, grid_h, ph, grid_w, pw))?
        .permute((0, 1, 2, 4, 3, 5))?
        .reshape((b, c * grid_h * grid_w, ph, pw))
}

/// Full BiRefNet model (inference only).
pub struct BiRefNet {
    bb: SwinTransformer,
    squeeze_module: BasicDecBlk,
    // Decoder blocks
    decoder_block4: BasicDecBlk,
    decoder_block3: BasicDecBlk,
    decoder_block2: BasicDecBlk,
    decoder_block1: BasicDecBlk,
    // Lateral blocks
    lateral_block4: BasicLatBlk,
    lateral_block3: BasicLatBlk,
    lateral_block2: BasicLatBlk,
    // Input patching blocks
    ipt_blk5: SimpleConvs,
    ipt_blk4: SimpleConvs,
    ipt_blk3: SimpleConvs,
    ipt_blk2: SimpleConvs,
    ipt_blk1: SimpleConvs,
    // Gradient attention
    gdt_4: GradientAttention,
    gdt_3: GradientAttention,
    gdt_2: GradientAttention,
    // Final output
    conv_out1: Conv2d,
}

impl BiRefNet {
    /// Create BiRefNet with the default Swin-V1-Large backbone configuration.
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let swin_config: SwinTransformerConfig = SwinTransformerConfig::large();
        let bb: SwinTransformer = SwinTransformer::new(&swin_config, vb.pp("bb"))?;

        // channels after mul_scl_ipt='cat': [3072, 1536, 768, 384]
        let channels: [usize; 4] = [3072, 1536, 768, 384];
        // cxt = [384, 768, 1536] (last 3 of channels reversed, cxt_num=3)
        let cxt_sum: usize = 384 + 768 + 1536; // 2688

        // Squeeze: BasicDecBlk(channels[0]+cxt_sum, channels[0])
        let squeeze_module = BasicDecBlk::new(
            channels[0] + cxt_sum,
            channels[0],
            vb.pp("squeeze_module.0"),
        )?;

        // Decoder channel computation (no pyramid neck for swin)
        let bb_neck: [usize; 4] = channels;
        let dec_out: [usize; 4] = [bb_neck[1], bb_neck[2], bb_neck[3], bb_neck[3] / 2];
        // dec_out = [1536, 768, 384, 192]

        // ipt_blk channels
        let ipt_in: [usize; 5] = [3072, 768, 192, 48, 3];
        let ipt_out: [usize; 4] = [
            channels[0] / 8,
            channels[1] / 8,
            channels[2] / 8,
            channels[3] / 8,
        ];
        // ipt_out = [384, 192, 96, 48]

        // dec_blk_in = bb_neck[i] + ipt_out[max(0, i-1)]
        let dec_in: [usize; 4] = [
            bb_neck[0] + ipt_out[0], // 3072 + 384 = 3456
            bb_neck[1] + ipt_out[0], // 1536 + 384 = 1920
            bb_neck[2] + ipt_out[1], // 768 + 192 = 960
            bb_neck[3] + ipt_out[2], // 384 + 96 = 480
        ];

        let decoder_block4 =
            BasicDecBlk::new(dec_in[0], dec_out[0], vb.pp("decoder.decoder_block4"))?;
        let decoder_block3 =
            BasicDecBlk::new(dec_in[1], dec_out[1], vb.pp("decoder.decoder_block3"))?;
        let decoder_block2 =
            BasicDecBlk::new(dec_in[2], dec_out[2], vb.pp("decoder.decoder_block2"))?;
        let decoder_block1 =
            BasicDecBlk::new(dec_in[3], dec_out[3], vb.pp("decoder.decoder_block1"))?;

        let lateral_block4 =
            BasicLatBlk::new(bb_neck[1], dec_out[0], vb.pp("decoder.lateral_block4"))?;
        let lateral_block3 =
            BasicLatBlk::new(bb_neck[2], dec_out[1], vb.pp("decoder.lateral_block3"))?;
        let lateral_block2 =
            BasicLatBlk::new(bb_neck[3], dec_out[2], vb.pp("decoder.lateral_block2"))?;

        let ic: usize = 64;
        let ipt_blk5 = SimpleConvs::new(ipt_in[0], ipt_out[0], ic, vb.pp("decoder.ipt_blk5"))?;
        let ipt_blk4 = SimpleConvs::new(ipt_in[1], ipt_out[0], ic, vb.pp("decoder.ipt_blk4"))?;
        let ipt_blk3 = SimpleConvs::new(ipt_in[2], ipt_out[1], ic, vb.pp("decoder.ipt_blk3"))?;
        let ipt_blk2 = SimpleConvs::new(ipt_in[3], ipt_out[2], ic, vb.pp("decoder.ipt_blk2"))?;
        let ipt_blk1 = SimpleConvs::new(ipt_in[4], ipt_out[3], ic, vb.pp("decoder.ipt_blk1"))?;

        let gdt_4 = GradientAttention::new(dec_out[0], "4", vb.pp("decoder"))?;
        let gdt_3 = GradientAttention::new(dec_out[1], "3", vb.pp("decoder"))?;
        let gdt_2 = GradientAttention::new(dec_out[2], "2", vb.pp("decoder"))?;

        // conv_out1: Conv2d(dec_out[3] + ipt_out[3], 1, 1)
        let conv_out1 = candle_nn::conv2d(
            dec_out[3] + ipt_out[3], // 192 + 48 = 240
            1,
            1,
            Default::default(),
            vb.pp("decoder.conv_out1.0"),
        )?;

        Ok(Self {
            bb,
            squeeze_module,
            decoder_block4,
            decoder_block3,
            decoder_block2,
            decoder_block1,
            lateral_block4,
            lateral_block3,
            lateral_block2,
            ipt_blk5,
            ipt_blk4,
            ipt_blk3,
            ipt_blk2,
            ipt_blk1,
            gdt_4,
            gdt_3,
            gdt_2,
            conv_out1,
        })
    }

    /// Run inference.
    ///
    /// Input: `[B, 3, H, W]` — normalized RGB image
    /// Output: `Vec<Tensor>` containing `[B, 1, H, W]` segmentation mask
    pub fn forward(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        let (_, _, h, w) = x.dims4()?;

        // ===== Encoder =====
        let features: Vec<Tensor> = self.bb.forward(x)?;
        // Swin output order matches BiRefNet: x1(shallowest) .. x4(deepest)
        let (x1, x2, x3, x4) = (&features[0], &features[1], &features[2], &features[3]);

        // Multi-scale input (cat mode): run backbone on half-resolution, concat
        let x_half: Tensor = x.upsample_bilinear2d(h / 2, w / 2, true)?;
        let features_half: Vec<Tensor> = self.bb.forward(&x_half)?;
        let (x1_, x2_, x3_, x4_) = (
            &features_half[0],
            &features_half[1],
            &features_half[2],
            &features_half[3],
        );

        let x1: Tensor = Tensor::cat(
            &[
                x1,
                &x1_.upsample_bilinear2d(x1.dims()[2], x1.dims()[3], true)?,
            ],
            1,
        )?;
        let x2: Tensor = Tensor::cat(
            &[
                x2,
                &x2_.upsample_bilinear2d(x2.dims()[2], x2.dims()[3], true)?,
            ],
            1,
        )?;
        let x3: Tensor = Tensor::cat(
            &[
                x3,
                &x3_.upsample_bilinear2d(x3.dims()[2], x3.dims()[3], true)?,
            ],
            1,
        )?;
        let x4: Tensor = Tensor::cat(
            &[
                x4,
                &x4_.upsample_bilinear2d(x4.dims()[2], x4.dims()[3], true)?,
            ],
            1,
        )?;

        // Context augmentation (cxt_num=3): concat x1,x2,x3 downsampled to x4's size
        let x4_h: usize = x4.dims()[2];
        let x4_w: usize = x4.dims()[3];
        let x4: Tensor = Tensor::cat(
            &[
                &x1.upsample_bilinear2d(x4_h, x4_w, true)?,
                &x2.upsample_bilinear2d(x4_h, x4_w, true)?,
                &x3.upsample_bilinear2d(x4_h, x4_w, true)?,
                &x4,
            ],
            1,
        )?;

        // ===== Squeeze =====
        let x4: Tensor = self.squeeze_module.forward(&x4)?;

        // ===== Decoder =====
        // Stage 4
        let patches: Tensor = self.make_patches(x, &x4)?;
        let ipt_feat: Tensor = self.ipt_blk5.forward(&patches.upsample_bilinear2d(
            x4.dims()[2],
            x4.dims()[3],
            true,
        )?)?;
        let x4: Tensor = Tensor::cat(&[&x4, &ipt_feat], 1)?;
        let p4: Tensor = self.decoder_block4.forward(&x4)?;
        let attn_4: Tensor = self.gdt_4.forward(&p4)?;
        let p4: Tensor = p4.broadcast_mul(&attn_4)?;

        // Stage 3
        let p4_up: Tensor = p4.upsample_bilinear2d(x3.dims()[2], x3.dims()[3], true)?;
        let p3: Tensor = (p4_up + self.lateral_block4.forward(&x3)?)?;
        let patches: Tensor = self.make_patches(x, &p3)?;
        let ipt_feat: Tensor = self.ipt_blk4.forward(&patches.upsample_bilinear2d(
            x3.dims()[2],
            x3.dims()[3],
            true,
        )?)?;
        let p3: Tensor = Tensor::cat(&[&p3, &ipt_feat], 1)?;
        let p3: Tensor = self.decoder_block3.forward(&p3)?;
        let attn_3: Tensor = self.gdt_3.forward(&p3)?;
        let p3: Tensor = p3.broadcast_mul(&attn_3)?;

        // Stage 2
        let p3_up: Tensor = p3.upsample_bilinear2d(x2.dims()[2], x2.dims()[3], true)?;
        let p2: Tensor = (p3_up + self.lateral_block3.forward(&x2)?)?;
        let patches: Tensor = self.make_patches(x, &p2)?;
        let ipt_feat: Tensor = self.ipt_blk3.forward(&patches.upsample_bilinear2d(
            x2.dims()[2],
            x2.dims()[3],
            true,
        )?)?;
        let p2: Tensor = Tensor::cat(&[&p2, &ipt_feat], 1)?;
        let p2: Tensor = self.decoder_block2.forward(&p2)?;
        let attn_2: Tensor = self.gdt_2.forward(&p2)?;
        let p2: Tensor = p2.broadcast_mul(&attn_2)?;

        // Stage 1
        let p2_up: Tensor = p2.upsample_bilinear2d(x1.dims()[2], x1.dims()[3], true)?;
        let p1: Tensor = (p2_up + self.lateral_block2.forward(&x1)?)?;
        let patches: Tensor = self.make_patches(x, &p1)?;
        let ipt_feat: Tensor = self.ipt_blk2.forward(&patches.upsample_bilinear2d(
            x1.dims()[2],
            x1.dims()[3],
            true,
        )?)?;
        let p1: Tensor = Tensor::cat(&[&p1, &ipt_feat], 1)?;
        let p1: Tensor = self.decoder_block1.forward(&p1)?;
        let p1: Tensor = p1.upsample_bilinear2d(h, w, true)?;

        // Final stage with full-res patching
        let patches: Tensor = self.make_patches(x, &p1)?;
        let ipt_feat: Tensor = self
            .ipt_blk1
            .forward(&patches.upsample_bilinear2d(h, w, true)?)?;
        let p1: Tensor = Tensor::cat(&[&p1, &ipt_feat], 1)?;
        let p1_out: Tensor = self.conv_out1.forward(&p1)?;

        Ok(vec![p1_out])
    }

    /// Split input image into patches relative to a reference feature map size.
    fn make_patches(&self, x: &Tensor, reference: &Tensor) -> Result<Tensor> {
        let grid_h: usize = x.dims()[2] / reference.dims()[2];
        let grid_w: usize = x.dims()[3] / reference.dims()[3];
        image2patches(x, grid_h, grid_w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_image2patches() -> Result<()> {
        let device = &Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (1, 3, 8, 8), device)?;
        let patches = image2patches(&x, 2, 2)?;
        // [1, 3, 8, 8] → [1, 3*2*2, 4, 4] = [1, 12, 4, 4]
        assert_eq!(patches.dims(), &[1, 12, 4, 4]);
        Ok(())
    }

    #[test]
    fn test_image2patches_asymmetric() -> Result<()> {
        let device = &Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (1, 3, 16, 8), device)?;
        let patches = image2patches(&x, 4, 2)?;
        // [1, 3, 16, 8] → [1, 3*4*2, 4, 4] = [1, 24, 4, 4]
        assert_eq!(patches.dims(), &[1, 24, 4, 4]);
        Ok(())
    }
}
