import torch.nn as nn
from lib.models.lightfc import MobileNetV2, pwcorr_se_scf_sc_iab_sc_concat, repn33_se_center_concat, pwcorr_scf_sc_iab_sc_concat, repn33_center_concat_geom
from lib.utils.load import load_pretrain


class LightFC(nn.Module):
    def __init__(self, cfg, env_num=0, training=False, ):
        super(LightFC, self).__init__()

        if cfg.MODEL.BACKBONE.TYPE == 'MobileNetV2':
            self.backbone = MobileNetV2()
        self.is_train_mode = training
        if self.is_train_mode:
            load_pretrain(self.backbone, env_num=env_num, training=training, cfg=cfg, mode=cfg.MODEL.BACKBONE.LOAD_MODE)

        self.fusion = pwcorr_scf_sc_iab_sc_concat(num_kernel=cfg.MODEL.FUSION.PARAMS.num_kernel,
                                                     adj_channel=cfg.MODEL.FUSION.PARAMS.adj_channel
                                                     )

        self.head = repn33_center_concat_geom(inplanes=cfg.MODEL.HEAD.PARAMS.inplanes,
                                            channel=cfg.MODEL.HEAD.PARAMS.channel,
                                            feat_sz=cfg.MODEL.HEAD.PARAMS.feat_sz,
                                            stride=cfg.MODEL.HEAD.PARAMS.stride,
                                            freeze_bn=cfg.MODEL.HEAD.PARAMS.freeze_bn,
                                            )

    def forward(self, z, x):
        if self.is_train_mode:
            z = self.backbone(z)
            x = self.backbone(x)

            opt = self.fusion(z, x)

            out = self.head(opt)
        else:
            return self.forward_tracking(z, x)
        return out

    #
    def forward_backbone(self, z):
        z = self.backbone(z)
        return z

    def forward_tracking(self, z_feat, x):
        x = self.backbone(x)
        opt = self.fusion(z_feat, x)
        out = self.head(opt)
        return out

    # ------------------------------------------------------------------
    # DPU export split (pairs with forward_dpu in ecm.py's
    # pwcorr_scf_sc_iab_sc_concat, and forward_dpu in erh.py's
    # repn33_center_concat_geom).
    #
    # Deployment call sequence, per frame:
    #   1. x_feat   = forward_search_backbone(x)                         [DPU]
    #   2. corr     = pixel_wise_corr_fpga(z_feat, x_feat)               [C++, host]
    #   3. raw_ctr, raw_offset, raw_size = forward_fusion_head_dpu(corr, x_feat)  [DPU]
    #   4. box      = decode(raw_ctr, raw_offset, raw_size)              [C++, host]
    #
    # z_feat comes from a one-time forward_backbone(z) call at init,
    # exactly as forward_tracking already assumes today.
    # ------------------------------------------------------------------

    def forward_search_backbone(self, x):
        """
        Export target: backbone on the search-region input only. Pure
        conv/BN traces, quantizes, and compiles to DPU cleanly. Uses the
        same weights as forward_backbone(z).
        """
        return self.backbone(x)

    def forward_fusion_head_dpu(self, corr, x_feat):
        """
        Export target: fusion (post-correlation) + head, stopping right
        before the head's sigmoid/clamp/decode step. `corr` MUST already
        be pixel_wise_corr_fpga(z_feat, x_feat), computed on the host -
        this method never calls pw_corr.

        Returns (raw_ctr, raw_offset, raw_size), all BEFORE sigmoid.
        Decode (cal_bbox) runs separately on the host.
        """
        opt = self.fusion.forward_dpu(corr, x_feat)
        return self.head.forward_dpu(opt)
