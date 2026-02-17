import logging
import math

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


class DDColorWrapper:
    """Wrapper for DDColor colorization model — real inference."""

    MODEL_SIZE_MAP = {
        "paper_tiny": "tiny",
        "modelscope": "large",
        "artistic": "large",
    }

    def __init__(
        self, model_path: str, device: str, variant: str = "paper_tiny"
    ) -> None:
        from ddcolor import DDColor, ColorizationPipeline, build_ddcolor_model

        model_size = self.MODEL_SIZE_MAP.get(variant, "tiny")
        torch_device = torch.device(device)

        self.model = build_ddcolor_model(
            DDColor,
            model_path=model_path,
            input_size=512,
            model_size=model_size,
            device=torch_device,
        )
        self.pipeline = ColorizationPipeline(
            self.model, input_size=512, device=torch_device
        )
        self.device = device
        logger.info(
            "DDColorWrapper loaded — %s (size=%s) on %s", model_path, model_size, device
        )

    def predict(self, image: np.ndarray, **kwargs) -> np.ndarray:
        return self.pipeline.process(image)


class RealESRGANWrapper:
    """Wrapper for Real-ESRGAN upscaling model — real inference."""

    def __init__(self, model_path: str, device: str, variant: str = "x4plus") -> None:
        import re
        from models.archs.rrdbnet_arch import RRDBNet

        self.device = device

        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]

        # Infer architecture from checkpoint so any RRDBNet weights work.
        num_feat = state_dict["conv_first.weight"].shape[0]
        num_in_ch = state_dict["conv_first.weight"].shape[1]
        num_out_ch = state_dict["conv_last.weight"].shape[0]
        num_grow_ch = state_dict["body.0.rdb1.conv1.weight"].shape[0]
        num_block = 1 + max(
            int(m.group(1))
            for k in state_dict
            for m in [re.match(r"body\.(\d+)\.", k)]
            if m
        )
        # conv_first input channels reveal the scale:
        # 3 → scale 4 (raw input), 12 → scale 2 (pixel_unshuffle ×2)
        self.scale = {3: 4, 12: 2, 48: 1}.get(num_in_ch, 4)

        model = RRDBNet(
            num_in_ch=num_out_ch,
            num_out_ch=num_out_ch,
            num_feat=num_feat,
            num_block=num_block,
            num_grow_ch=num_grow_ch,
            scale=self.scale,
        )

        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.to(device)
        self.model = model
        logger.info(
            "RealESRGANWrapper loaded — %s (scale=%d) on %s",
            model_path,
            self.scale,
            device,
        )

    def predict(self, image: np.ndarray, **kwargs) -> np.ndarray:
        tile_size = kwargs.get("tile_size", 0)

        # BGR uint8 -> RGB float32 [0, 1]
        img = image[:, :, ::-1].astype(np.float32) / 255.0
        img_t = (
            torch.from_numpy(img.copy()).permute(2, 0, 1).unsqueeze(0).to(self.device)
        )

        with torch.no_grad():
            if tile_size > 0:
                output = self._tile_process(img_t, tile_size)
            else:
                output = self.model(img_t)

        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        # RGB -> BGR
        return output[:, :, ::-1].copy()

    def _tile_process(self, img, tile_size=256, tile_pad=10):
        batch, channel, height, width = img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output = img.new_zeros(batch, channel, output_height, output_width)

        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        for y in range(tiles_y):
            for x in range(tiles_x):
                ofs_x = x * tile_size
                ofs_y = y * tile_size

                input_start_x = max(ofs_x - tile_pad, 0)
                input_end_x = min(ofs_x + tile_size + tile_pad, width)
                input_start_y = max(ofs_y - tile_pad, 0)
                input_end_y = min(ofs_y + tile_size + tile_pad, height)

                input_tile = img[
                    :, :, input_start_y:input_end_y, input_start_x:input_end_x
                ]
                output_tile = self.model(input_tile)

                # Remove padding from output tile
                out_start_x = (ofs_x - input_start_x) * self.scale
                out_end_x = (
                    output_tile.shape[3]
                    - (input_end_x - min(ofs_x + tile_size, width)) * self.scale
                )
                out_start_y = (ofs_y - input_start_y) * self.scale
                out_end_y = (
                    output_tile.shape[2]
                    - (input_end_y - min(ofs_y + tile_size, height)) * self.scale
                )

                dest_start_x = ofs_x * self.scale
                dest_end_x = min(
                    (ofs_x + tile_size) * self.scale,
                    output_width,
                )
                dest_start_y = ofs_y * self.scale
                dest_end_y = min(
                    (ofs_y + tile_size) * self.scale,
                    output_height,
                )

                output[:, :, dest_start_y:dest_end_y, dest_start_x:dest_end_x] = (
                    output_tile[:, :, out_start_y:out_end_y, out_start_x:out_end_x]
                )
        return output


class NAFNetWrapper:
    """Wrapper for NAFNet restoration model — real inference."""

    def __init__(self, model_path: str, device: str, variant: str = "denoise") -> None:
        import re
        from models.archs.nafnet_arch import NAFNet

        self.device = device

        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        if "params" in state_dict:
            state_dict = state_dict["params"]

        # Infer architecture from checkpoint keys so we don't need
        # hard-coded block counts per variant.
        enc_blk_nums = self._count_blocks(state_dict, r"encoders\.(\d+)\.(\d+)\.")
        dec_blk_nums = self._count_blocks(state_dict, r"decoders\.(\d+)\.(\d+)\.")
        middle_blk_num = 1 + max(
            (
                int(m.group(1))
                for k in state_dict
                for m in [re.match(r"middle_blks\.(\d+)\.", k)]
                if m
            ),
            default=0,
        )
        width = state_dict["intro.weight"].shape[0]

        model = NAFNet(
            img_channel=3,
            width=width,
            middle_blk_num=middle_blk_num,
            enc_blk_nums=enc_blk_nums,
            dec_blk_nums=dec_blk_nums,
        )

        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.to(device)
        self.model = model
        logger.info(
            "NAFNetWrapper loaded — %s (variant=%s) on %s",
            model_path,
            variant,
            device,
        )

    @staticmethod
    def _count_blocks(state_dict, pattern):
        """Count blocks per stage from checkpoint key names."""
        import re

        stages: dict[int, int] = {}
        for k in state_dict:
            m = re.match(pattern, k)
            if m:
                stage, idx = int(m.group(1)), int(m.group(2))
                stages[stage] = max(stages.get(stage, 0), idx + 1)
        return [stages[i] for i in range(len(stages))]

    def predict(self, image: np.ndarray, **kwargs) -> np.ndarray:
        # BGR uint8 -> RGB float32 [0, 1]
        img = image[:, :, ::-1].astype(np.float32) / 255.0
        img_t = (
            torch.from_numpy(img.copy()).permute(2, 0, 1).unsqueeze(0).to(self.device)
        )

        with torch.no_grad():
            output = self.model(img_t)

        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        # RGB -> BGR
        return output[:, :, ::-1].copy()


class CodeFormerWrapper:
    """Wrapper for CodeFormer face restoration model — real inference."""

    def __init__(self, model_path: str, device: str, variant: str = "v0.1") -> None:
        from models.archs.codeformer_arch import CodeFormer

        self.device = device

        model = CodeFormer(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        )

        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]

        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.to(device)
        self.model = model

        from facexlib.utils.face_restoration_helper import FaceRestoreHelper

        self.face_helper = FaceRestoreHelper(
            upscale_factor=2,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=device,
        )
        logger.info(
            "CodeFormerWrapper loaded — %s on %s",
            model_path,
            device,
        )

    def predict(self, image: np.ndarray, **kwargs) -> np.ndarray:
        fidelity = kwargs.get("fidelity", 0.5)
        upscale = kwargs.get("upscale", 2)

        self.face_helper.clean_all()
        self.face_helper.upscale_factor = upscale

        # facexlib expects BGR uint8 numpy array
        self.face_helper.read_image(image)
        self.face_helper.get_face_landmarks_5(
            only_center_face=False,
            resize=640,
            eye_dist_threshold=5,
        )
        self.face_helper.align_warp_face()

        if len(self.face_helper.cropped_faces) == 0:
            logger.warning("No faces detected, returning bicubic upscale")
            h, w = image.shape[:2]
            return cv2.resize(
                image,
                (w * upscale, h * upscale),
                interpolation=cv2.INTER_CUBIC,
            )

        for cropped_face in self.face_helper.cropped_faces:
            # BGR uint8 -> RGB float32 [-1, 1]
            face = cropped_face[:, :, ::-1].astype(np.float32) / 255.0
            face_t = torch.from_numpy(face.copy()).permute(2, 0, 1).unsqueeze(0)
            face_t = (face_t - 0.5) / 0.5  # normalize to [-1, 1]
            face_t = face_t.to(self.device)

            with torch.no_grad():
                output, _, _ = self.model(face_t, w=fidelity, adain=True)

            # [-1, 1] -> [0, 1] -> uint8, RGB -> BGR
            restored = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            restored = np.clip((restored + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
            restored = restored[:, :, ::-1].copy()  # RGB -> BGR

            # Preserve original color: keep luminance from restored,
            # chrominance from original to prevent color artifacts
            # (e.g. blue tints on B&W/sepia photos).
            original_ycrcb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2YCrCb)
            restored_ycrcb = cv2.cvtColor(restored, cv2.COLOR_BGR2YCrCb)
            restored_ycrcb[:, :, 1:] = original_ycrcb[:, :, 1:]
            restored = cv2.cvtColor(restored_ycrcb, cv2.COLOR_YCrCb2BGR)

            self.face_helper.add_restored_face(restored)

        self.face_helper.get_inverse_affine(None)
        result = self.face_helper.paste_faces_to_input_image()

        return result
