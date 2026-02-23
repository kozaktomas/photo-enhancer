import logging
import math
import os
from typing import ClassVar

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


class DDColorWrapper:
    """Wrapper for DDColor colorization model — real inference."""

    MODEL_SIZE_MAP: ClassVar[dict[str, str]] = {
        "paper_tiny": "tiny",
        "modelscope": "large",
        "artistic": "large",
    }

    def __init__(self, model_path: str, device: str, variant: str = "paper_tiny") -> None:
        """Load DDColor model and build the colorization pipeline.

        Args:
            model_path: Path to the DDColor checkpoint file.
            device: Compute device string (``"cuda"``, ``"mps"``, or ``"cpu"``).
            variant: Model variant name used to select model size.
        """
        from ddcolor import ColorizationPipeline, DDColor, build_ddcolor_model

        model_size = self.MODEL_SIZE_MAP.get(variant, "tiny")
        torch_device = torch.device(device)

        self.model = build_ddcolor_model(
            DDColor,
            model_path=model_path,
            input_size=512,
            model_size=model_size,
            device=torch_device,
        )
        self.pipeline = ColorizationPipeline(self.model, input_size=512, device=torch_device)
        self.device = device
        logger.info("DDColorWrapper loaded — %s (size=%s) on %s", model_path, model_size, device)

    def predict(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Colorize a grayscale/B&W image.

        Args:
            image: BGR uint8 numpy array.
            **kwargs: Optional ``render_factor`` (unused by DDColor pipeline).

        Returns:
            Colorized BGR uint8 numpy array.
        """
        return self.pipeline.process(image)


class RealESRGANWrapper:
    """Wrapper for Real-ESRGAN upscaling model — real inference."""

    def __init__(self, model_path: str, device: str, variant: str = "x4plus") -> None:
        """Load a Real-ESRGAN RRDBNet checkpoint, inferring architecture from keys.

        Args:
            model_path: Path to the Real-ESRGAN checkpoint file.
            device: Compute device string.
            variant: Model variant name (informational, architecture is inferred).
        """
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
            int(m.group(1)) for k in state_dict for m in [re.match(r"body\.(\d+)\.", k)] if m
        )
        # conv_first input channels reveal the scale:
        # 3 -> scale 4 (raw input), 12 -> scale 2 (pixel_unshuffle x2)
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
        """Upscale an image using the loaded Real-ESRGAN model.

        Args:
            image: BGR uint8 numpy array.
            **kwargs: Optional ``tile_size`` (int) for tiled processing,
                ``scale`` (int, unused — scale is determined by the model).

        Returns:
            Upscaled BGR uint8 numpy array.
        """
        tile_size = kwargs.get("tile_size", 0)

        # BGR uint8 -> RGB float32 [0, 1]
        img = image[:, :, ::-1].astype(np.float32) / 255.0
        img_t = torch.from_numpy(img.copy()).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self._tile_process(img_t, tile_size) if tile_size > 0 else self.model(img_t)

        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        # RGB -> BGR
        return output[:, :, ::-1].copy()

    def _tile_process(self, img, tile_size=256, tile_pad=10):
        """Process an image tensor in tiles to limit VRAM usage.

        Splits the input into a grid of overlapping tiles, runs each through the
        model, strips the overlap padding, and assembles the output.

        Args:
            img: Input tensor of shape ``(B, C, H, W)``.
            tile_size: Tile dimension in pixels (default 256).
            tile_pad: Overlap padding in pixels (default 10).

        Returns:
            Output tensor of shape ``(B, C, H*scale, W*scale)``.
        """
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

                input_tile = img[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
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

                output[:, :, dest_start_y:dest_end_y, dest_start_x:dest_end_x] = output_tile[
                    :, :, out_start_y:out_end_y, out_start_x:out_end_x
                ]
        return output


class NAFNetWrapper:
    """Wrapper for NAFNet restoration model — real inference."""

    def __init__(self, model_path: str, device: str, variant: str = "denoise") -> None:
        """Load a NAFNet checkpoint, inferring architecture from keys.

        Args:
            model_path: Path to the NAFNet checkpoint file.
            device: Compute device string.
            variant: Model variant name (informational, architecture is inferred).
        """
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
        """Count blocks per stage from checkpoint key names.

        Args:
            state_dict: Model state dict mapping key names to tensors.
            pattern: Regex with two capture groups: ``(stage_idx, block_idx)``.

        Returns:
            List of block counts, one per stage, in order.
        """
        import re

        stages: dict[int, int] = {}
        for k in state_dict:
            m = re.match(pattern, k)
            if m:
                stage, idx = int(m.group(1)), int(m.group(2))
                stages[stage] = max(stages.get(stage, 0), idx + 1)
        return [stages[i] for i in range(len(stages))]

    def predict(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Remove noise or blur from an image using the loaded NAFNet model.

        Args:
            image: BGR uint8 numpy array.
            **kwargs: Optional ``tile_size`` (int, unused by NAFNet wrapper).

        Returns:
            Restored BGR uint8 numpy array.
        """
        # BGR uint8 -> RGB float32 [0, 1]
        img = image[:, :, ::-1].astype(np.float32) / 255.0
        img_t = torch.from_numpy(img.copy()).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_t)

        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        # RGB -> BGR
        return output[:, :, ::-1].copy()


class CodeFormerWrapper:
    """Wrapper for CodeFormer face restoration model — real inference."""

    def __init__(self, model_path: str, device: str, variant: str = "v0.1") -> None:
        """Load CodeFormer checkpoint and initialize the face detection helper.

        Args:
            model_path: Path to the CodeFormer checkpoint file.
            device: Compute device string.
            variant: Model variant name (informational).
        """
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
        """Detect and restore faces in an image.

        Detects all faces, restores each via the CodeFormer transformer, and
        pastes them back into the (optionally upscaled) original. If no faces
        are detected, returns a bicubic upscale.

        Args:
            image: BGR uint8 numpy array.
            **kwargs: Optional ``fidelity`` (float, 0-1) and ``upscale`` (int, 1-4).

        Returns:
            Face-restored BGR uint8 numpy array.
        """
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


class LaMaWrapper:
    """Wrapper for LaMa inpainting model — real inference.

    Uses a TorchScript JIT model to fill masked regions in an image.
    """

    def __init__(self, model_path: str, device: str, variant: str = "big") -> None:
        """Load a LaMa TorchScript checkpoint.

        Args:
            model_path: Path to the ``.pt`` TorchScript model file.
            device: Compute device string (``"cuda"``, ``"mps"``, or ``"cpu"``).
            variant: Model variant name (informational).
        """
        self.device = device
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        logger.info("LaMaWrapper loaded — %s on %s", model_path, device)

    def predict(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Inpaint masked regions of an image.

        Args:
            image: BGR uint8 numpy array.
            **kwargs: Required ``mask`` (grayscale uint8 numpy array, 255 = inpaint).

        Returns:
            Inpainted BGR uint8 numpy array.

        Raises:
            ValueError: If no mask is provided or mask dimensions don't match.
        """
        mask = kwargs.get("mask")
        if mask is None:
            raise ValueError("Inpainting requires a mask image")

        h, w = image.shape[:2]
        mh, mw = mask.shape[:2]
        if (mh, mw) != (h, w):
            raise ValueError(f"Mask dimensions ({mw}x{mh}) do not match image dimensions ({w}x{h})")

        # Ensure mask is single-channel
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # BGR -> RGB, float32 [0,1]
        img_rgb = image[:, :, ::-1].astype(np.float32) / 255.0
        mask_f = mask.astype(np.float32) / 255.0

        # Pad to multiple of 8
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8

        img_t = torch.from_numpy(img_rgb.copy()).permute(2, 0, 1).unsqueeze(0)
        mask_t = torch.from_numpy(mask_f.copy()).unsqueeze(0).unsqueeze(0)

        if pad_h > 0 or pad_w > 0:
            img_t = torch.nn.functional.pad(img_t, (0, pad_w, 0, pad_h), mode="reflect")
            mask_t = torch.nn.functional.pad(mask_t, (0, pad_w, 0, pad_h), mode="reflect")

        # Binarize mask for model input
        mask_t = (mask_t > 0.5).float()

        img_t = img_t.to(self.device)
        mask_t = mask_t.to(self.device)

        with torch.no_grad():
            output = self.model(img_t, mask_t)

        # Crop padding and convert back
        output = output[:, :, :h, :w]
        result = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)

        # RGB -> BGR
        result = result[:, :, ::-1].copy()

        # Blend: keep original pixels in non-masked areas
        mask_3ch = (mask[:, :, np.newaxis] > 127).astype(np.float32)
        result = (result * mask_3ch + image * (1 - mask_3ch)).astype(np.uint8)

        return result


class OldPhotoRestoreWrapper:
    """Wrapper for "Bringing Old Photos Back to Life" restoration pipeline.

    Performs automatic scratch detection, global restoration via VAE + mapping
    network, and optional face enhancement via SPADE generator. Uses dlib for
    face detection/alignment instead of facexlib.
    """

    FACE_TEMPLATE = np.float32(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ]
    )

    def __init__(self, model_path: str, device: str, variant: str = "v1") -> None:
        """Load all sub-networks for old photo restoration.

        Args:
            model_path: Path to the directory containing all weight files.
            device: Compute device string (``"cuda"``, ``"mps"``, or ``"cpu"``).
            variant: Model variant name (informational).
        """
        import dlib

        from models.archs.old_photo_detect_arch import UNet
        from models.archs.old_photo_face_arch import SPADEGenerator
        from models.archs.old_photo_global_arch import (
            GlobalGenerator_DCDCv2,
            Mapping_Model_with_mask_2,
        )

        self.device = device

        # Scratch detection UNet
        self.scratch_net = UNet(in_channels=1, out_channels=1, depth=4, conv_num=2, wf=6)
        scratch_sd = torch.load(
            os.path.join(model_path, "scratch_detection.pt"),
            map_location=device,
            weights_only=False,
        )
        if "model_state" in scratch_sd:
            scratch_sd = scratch_sd["model_state"]
        self.scratch_net.load_state_dict(scratch_sd)
        self.scratch_net.eval().to(device)

        # VAE A encoder (quality)
        self.vae_a = GlobalGenerator_DCDCv2(
            input_nc=3, output_nc=3, ngf=64, k_size=4, n_downsampling=3
        )
        vae_a_sd = torch.load(
            os.path.join(model_path, "vae_a_encoder.pth"),
            map_location=device,
            weights_only=False,
        )
        self.vae_a.load_state_dict(vae_a_sd)
        self.vae_a.eval().to(device)

        # VAE B decoder (scratch)
        self.vae_b = GlobalGenerator_DCDCv2(
            input_nc=3, output_nc=3, ngf=64, k_size=4, n_downsampling=3
        )
        vae_b_sd = torch.load(
            os.path.join(model_path, "vae_b_decoder.pth"),
            map_location=device,
            weights_only=False,
        )
        self.vae_b.load_state_dict(vae_b_sd)
        self.vae_b.eval().to(device)

        # Mapping network
        self.mapping_net = Mapping_Model_with_mask_2(nc=64, mc=512)
        mapping_sd = torch.load(
            os.path.join(model_path, "mapping_net.pth"),
            map_location=device,
            weights_only=False,
        )
        self.mapping_net.load_state_dict(mapping_sd)
        self.mapping_net.eval().to(device)

        # Face enhancement SPADE generator
        self.face_gen = SPADEGenerator(input_nc=3, output_nc=3, ngf=64, semantic_nc=3)
        face_sd = torch.load(
            os.path.join(model_path, "face_enhance_gen.pth"),
            map_location=device,
            weights_only=False,
        )
        self.face_gen.load_state_dict(face_sd)
        self.face_gen.eval().to(device)

        # dlib face detector and landmark predictor
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(
            os.path.join(model_path, "shape_predictor_68_face_landmarks.dat")
        )

        logger.info(
            "OldPhotoRestoreWrapper loaded — %s (variant=%s) on %s",
            model_path,
            variant,
            device,
        )

    def predict(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Restore an old/damaged photo.

        Args:
            image: BGR uint8 numpy array.
            **kwargs: Optional parameters:
                - ``with_scratch`` (bool, default True): detect and repair scratches.
                - ``with_face`` (bool, default True): enhance detected faces.
                - ``scratch_threshold`` (float, default 0.4): scratch detection threshold.

        Returns:
            Restored BGR uint8 numpy array.
        """
        with_scratch = kwargs.get("with_scratch", True)
        with_face = kwargs.get("with_face", True)
        scratch_threshold = kwargs.get("scratch_threshold", 0.4)

        h, w = image.shape[:2]

        # Step 1: Detect scratches
        if with_scratch:
            scratch_mask = self._detect_scratches(image, threshold=scratch_threshold)
        else:
            scratch_mask = np.zeros((h, w), dtype=np.uint8)

        # Step 2: Global restoration
        result = self._global_restore(image, scratch_mask)

        # Step 3: Face enhancement (detect on original for sharper landmarks)
        if with_face:
            result = self._enhance_faces(result, detect_image=image)

        return result

    def _detect_scratches(self, image: np.ndarray, threshold: float = 0.4) -> np.ndarray:
        """Detect scratches in the image using UNet.

        Args:
            image: BGR uint8 numpy array.
            threshold: Sigmoid probability below which pixels are zeroed out.

        Returns:
            Soft scratch mask (uint8, 0-255) at the original image resolution.
            Values represent scratch confidence: 0 = clean, 255 = definite scratch.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize to fixed size for the UNet
        resized = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
        inp = resized.astype(np.float32) / 255.0
        inp_t = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.scratch_net(inp_t)
            out = torch.sigmoid(out)

        prob = out.squeeze().cpu().numpy()
        # Zero out below threshold, keep soft probabilities above
        prob[prob < threshold] = 0.0

        # Dilate to cover scratch edges (acts as max filter on soft values)
        mask = (prob * 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Resize back to original resolution with smooth interpolation
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        return mask

    def _global_restore(self, image: np.ndarray, scratch_mask: np.ndarray) -> np.ndarray:
        """Globally restore the image using VAE encoder → mapping → decoder.

        Args:
            image: BGR uint8 numpy array.
            scratch_mask: Soft mask (uint8, 0-255) at image resolution.

        Returns:
            Restored BGR uint8 numpy array at original resolution.
        """
        h, w = image.shape[:2]

        # Resize to 256x256 for the VAE — binarize for the mapping network
        # (trained with binary masks) while keeping the soft version for blending.
        img_256 = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        mask_256 = cv2.resize(scratch_mask, (256, 256), interpolation=cv2.INTER_LINEAR)
        mask_256_bin = ((mask_256 > 127).astype(np.uint8)) * 255

        # BGR → RGB, normalize to [-1, 1]
        img_rgb = img_256[:, :, ::-1].astype(np.float32) / 127.5 - 1.0
        img_t = torch.from_numpy(img_rgb.copy()).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Binary mask to [0, 1] tensor for mapping network
        mask_t = torch.from_numpy(mask_256_bin.astype(np.float32) / 255.0)
        mask_t = mask_t.unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Encode with VAE_A
            features = self.vae_a(img_t, flow="enc")

            # Map features with scratch mask awareness
            mapped = self.mapping_net(features, mask_t)

            # Decode with VAE_B
            output = self.vae_b(mapped, flow="dec")

        # [-1, 1] → [0, 255], RGB → BGR
        result = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        result = np.clip((result + 1.0) * 127.5, 0, 255).astype(np.uint8)
        result = result[:, :, ::-1].copy()

        # Resize back to original
        result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LANCZOS4)

        # Blend with original to preserve detail in non-scratched areas.
        # Scratched pixels (mask=255) get 100% restored result; clean pixels
        # keep most of the original luminance (sharpness) while adopting the
        # restored chrominance (color correction from the VAE).
        blend_mask = scratch_mask.astype(np.float32) / 255.0
        # Feather the scratch mask for smooth transitions
        blur_k = max(3, min(w, h) // 32) | 1  # ensure odd
        blend_mask = cv2.GaussianBlur(blend_mask, (blur_k, blur_k), 0)

        orig_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        rest_ycrcb = cv2.cvtColor(result, cv2.COLOR_BGR2YCrCb).astype(np.float32)

        alpha = blend_mask[:, :, np.newaxis]
        # Luminance: smooth blend — clean(alpha=0) keeps 70% original / 30% restored,
        # scratched(alpha=1) gets 100% restored, linear in between.
        lum = (1 - alpha) * (0.7 * orig_ycrcb[:, :, 0:1] + 0.3 * rest_ycrcb[:, :, 0:1]) + (
            alpha * rest_ycrcb[:, :, 0:1]
        )
        # Chrominance: clean(alpha=0) keeps 30% original / 70% restored,
        # scratched(alpha=1) gets 100% restored.
        chrom = (1 - alpha) * (0.3 * orig_ycrcb[:, :, 1:3] + 0.7 * rest_ycrcb[:, :, 1:3]) + (
            alpha * rest_ycrcb[:, :, 1:3]
        )

        merged = np.concatenate([lum, chrom], axis=2)
        merged = np.clip(merged, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
        return result

    def _enhance_faces(
        self, image: np.ndarray, detect_image: np.ndarray | None = None
    ) -> np.ndarray:
        """Detect and enhance faces using dlib + SPADE generator.

        Args:
            image: BGR uint8 numpy array (restored image for face cropping).
            detect_image: BGR uint8 numpy array to run face detection on.
                If ``None``, detection runs on *image*.

        Returns:
            Image with enhanced faces blended back, BGR uint8.
        """
        det_img = detect_image if detect_image is not None else image
        gray = cv2.cvtColor(det_img, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray, 1)

        if len(faces) == 0:
            return image

        result = image.copy()
        for face in faces:
            landmarks = self.landmark_predictor(gray, face)

            # Extract 5-point landmarks (eye centers, nose, mouth corners)
            left_eye = np.mean(
                [[landmarks.part(i).x, landmarks.part(i).y] for i in range(36, 42)],
                axis=0,
            )
            right_eye = np.mean(
                [[landmarks.part(i).x, landmarks.part(i).y] for i in range(42, 48)],
                axis=0,
            )
            src_pts = np.float32(
                [
                    left_eye,  # left eye center
                    right_eye,  # right eye center
                    [landmarks.part(30).x, landmarks.part(30).y],  # nose tip
                    [landmarks.part(48).x, landmarks.part(48).y],  # left mouth
                    [landmarks.part(54).x, landmarks.part(54).y],  # right mouth
                ]
            )

            # Target face size for the SPADE generator (256x256)
            face_size = 256
            dst_pts = self.FACE_TEMPLATE * (face_size / 112.0)

            # Compute affine transform
            tfm, _ = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.LMEDS)
            if tfm is None:
                continue

            # Warp face to aligned position
            aligned = cv2.warpAffine(image, tfm, (face_size, face_size), flags=cv2.INTER_LINEAR)

            # Run SPADE generator
            # BGR → RGB, normalize to [-1, 1]
            face_rgb = aligned[:, :, ::-1].astype(np.float32) / 127.5 - 1.0
            face_t = torch.from_numpy(face_rgb.copy()).permute(2, 0, 1).unsqueeze(0).to(self.device)

            with torch.no_grad():
                enhanced = self.face_gen(face_t)

            # [-1, 1] → [0, 255], RGB → BGR
            enhanced = enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy()
            enhanced = np.clip((enhanced + 1.0) * 127.5, 0, 255).astype(np.uint8)
            enhanced = enhanced[:, :, ::-1].copy()

            # Inverse warp back
            tfm_inv = cv2.invertAffineTransform(tfm)
            h, w = image.shape[:2]
            warped_back = cv2.warpAffine(enhanced, tfm_inv, (w, h), flags=cv2.INTER_LINEAR)

            # Create mask for blending with proportional feathering
            face_mask = np.ones((face_size, face_size), dtype=np.uint8) * 255
            erode_size = max(1, face_size // 16)
            erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
            face_mask = cv2.erode(face_mask, erode_kernel, iterations=1)
            blur_size = max(1, face_size // 8) | 1  # ensure odd
            face_mask = cv2.GaussianBlur(face_mask, (blur_size, blur_size), 0)
            warped_mask = cv2.warpAffine(face_mask, tfm_inv, (w, h), flags=cv2.INTER_LINEAR)
            warped_mask = warped_mask.astype(np.float32) / 255.0

            # Blend enhanced face into result
            mask_3ch = warped_mask[:, :, np.newaxis]
            result = (result * (1 - mask_3ch) + warped_back * mask_3ch).astype(np.uint8)

        return result
