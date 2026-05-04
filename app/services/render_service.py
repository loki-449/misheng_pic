import os
import time
import uuid
from datetime import UTC, datetime
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from app.services.font_service import load_title_fonts


def _open_rgba(path: str) -> Image.Image | None:
    try:
        with Image.open(path) as im:
            return im.convert("RGBA")
    except OSError:
        return None


def _resize_contain_rgba(
    im: Image.Image | None,
    w: int,
    h: int,
    pad: tuple[int, int, int] = (252, 251, 248),
) -> Image.Image | None:
    """等比放入画框，不足区域用浅色底（不裁切商品主体）。"""
    if im is None or w <= 0 or h <= 0:
        return None
    im = im.copy()
    im.thumbnail((max(1, w - 4), max(1, h - 4)), Image.Resampling.LANCZOS)
    out = Image.new("RGBA", (w, h), (*pad, 255))
    ox = (w - im.width) // 2
    oy = (h - im.height) // 2
    if im.mode == "RGBA":
        out.paste(im, (ox, oy), im)
    else:
        out.paste(im.convert("RGBA"), (ox, oy))
    return out


def _resize_cover(im: Image.Image | None, w: int, h: int) -> Image.Image | None:
    if im is None or w <= 0 or h <= 0:
        return None
    src_w, src_h = im.size
    if src_w <= 0 or src_h <= 0:
        return None
    scale = max(w / float(src_w), h / float(src_h))
    nw = max(1, int(round(src_w * scale)))
    nh = max(1, int(round(src_h * scale)))
    resized = im.resize((nw, nh), Image.Resampling.LANCZOS)
    left = max(0, (nw - w) // 2)
    top = max(0, (nh - h) // 2)
    return resized.crop((left, top, left + w, top + h))


def _grid_shape(k: int, distribution: str) -> tuple[int, int]:
    d = (distribution or "").strip()
    if d == "grid_1x2":
        return (2, 1) if k >= 2 else (1, 1)
    if d == "grid_1x3":
        return (3, 1) if k >= 3 else (k, 1)
    if d == "grid_2x2":
        return (2, 2)
    if d == "grid_2x3":
        return (3, 2)
    cols = max(1, int(round(k**0.5)))
    rows = max(1, (k + cols - 1) // cols)
    return cols, rows


class PosterRenderer:
    """
    海报渲染：
    - MVP：素材背景合成或程序化渐变背景、排版模板、多版本变体
    - 兼容旧版：无 _mvp_layout 时沿用居中纯色逻辑
    """

    def __init__(self) -> None:
        self.output_dir = "./app/static/results"
        self.upload_dir = "./app/static/uploads"
        self.cleanup_max_files = int(os.getenv("STORAGE_MAX_FILES", "500"))
        self.cleanup_max_age_days = int(os.getenv("STORAGE_MAX_AGE_DAYS", "7"))
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.upload_dir, exist_ok=True)

    def cleanup_directory(self, directory: str) -> dict[str, int]:
        """
        清理目录策略：
        1) 删除超过保留天数的文件
        2) 若仍超出最大文件数，继续删除最旧文件
        """
        deleted_by_age = 0
        deleted_by_count = 0
        now = time.time()
        max_age_seconds = max(1, self.cleanup_max_age_days) * 24 * 3600
        max_files = max(1, self.cleanup_max_files)

        if not os.path.exists(directory):
            return {"deleted_by_age": 0, "deleted_by_count": 0}

        files: list[tuple[str, float]] = []
        for entry in os.scandir(directory):
            if entry.is_file():
                try:
                    mtime = entry.stat().st_mtime
                    files.append((entry.path, mtime))
                except OSError:
                    continue

        for path, mtime in files:
            if now - mtime > max_age_seconds:
                try:
                    os.remove(path)
                    deleted_by_age += 1
                except OSError:
                    pass

        refreshed: list[tuple[str, float]] = []
        for entry in os.scandir(directory):
            if entry.is_file():
                try:
                    refreshed.append((entry.path, entry.stat().st_mtime))
                except OSError:
                    continue
        refreshed.sort(key=lambda x: x[1])
        overflow = len(refreshed) - max_files
        if overflow > 0:
            for path, _ in refreshed[:overflow]:
                try:
                    os.remove(path)
                    deleted_by_count += 1
                except OSError:
                    pass

        return {"deleted_by_age": deleted_by_age, "deleted_by_count": deleted_by_count}

    def run_storage_cleanup(self) -> dict[str, dict[str, int]]:
        return {
            "results": self.cleanup_directory(self.output_dir),
            "uploads": self.cleanup_directory(self.upload_dir),
        }

    @staticmethod
    def _resolve_background_file(rel_path: str) -> str | None:
        if not rel_path:
            return None
        candidates = [rel_path, os.path.join(".", rel_path)]
        for p in candidates:
            if os.path.isfile(p):
                return p
        return None

    def _load_or_procedural_background(
        self,
        size: int,
        variant_index: int,
        candidates: list[dict[str, Any]],
        image_stats: dict[str, Any] | None,
    ) -> tuple[Image.Image, bool]:
        """
        返回 (RGB 背景图, 是否使用了素材文件)。
        """
        n = len(candidates)
        order = [((variant_index + i) % n) for i in range(n)] if n else []
        for idx in order:
            item = candidates[idx]
            resolved = self._resolve_background_file(str(item.get("path", "")))
            if resolved:
                try:
                    with Image.open(resolved) as bg_img:
                        bg = bg_img.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
                    darken = Image.new("RGB", (size, size), (20, 20, 24))
                    bg = Image.blend(bg, darken, 0.12)
                    return bg, True
                except OSError:
                    continue

        dom = (image_stats or {}).get("dominant_rgb", [230, 224, 216])
        r, g, b = float(dom[0]), float(dom[1]), float(dom[2])
        v = float(variant_index)
        top = np.array([r + v * 14.0, g + v * 9.0, b - v * 8.0], dtype=np.float32)
        bot = np.array(
            [max(0.0, r - 58.0 - v * 6.0), max(0.0, g - 52.0 - v * 5.0), max(0.0, b - 48.0)],
            dtype=np.float32,
        )
        ys = np.linspace(0.0, 1.0, size, dtype=np.float32)[:, None]
        interp = top * (1.0 - ys) + bot * ys
        grid = np.repeat(interp[:, None, :], size, axis=1).astype(np.float32)
        cy, cx = size // 2, size // 2
        yy, xx = np.ogrid[0:size, 0:size]
        dist = np.sqrt((yy.astype(np.float32) - cy) ** 2 + (xx.astype(np.float32) - cx) ** 2) / float(
            max(size * 0.72, 1)
        )
        vignette = np.clip(1.0 - dist * 0.38, 0.52, 1.0)[:, :, np.newaxis]
        grid = np.clip(grid * vignette, 0.0, 255.0)
        rng = np.random.default_rng(17 + int(variant_index) * 997)
        grain = rng.normal(0.0, 2.2, grid.shape).astype(np.float32)
        grid = np.clip(grid + grain, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(grid, mode="RGB"), False

    @staticmethod
    def _max_cells_for_distribution(distribution: str, n_paths: int) -> int:
        d = (distribution or "").strip()
        if d == "grid_1x2":
            return min(2, n_paths)
        if d == "grid_1x3":
            return min(3, n_paths)
        if d == "grid_2x2":
            return min(4, n_paths)
        if d == "grid_2x3":
            return min(6, n_paths)
        return n_paths

    def _build_product_collage_rgba(
        self,
        paths: list[str],
        out_w: int,
        out_h: int,
        stitch_mode: str,
        distribution: str,
    ) -> Image.Image:
        """
        在固定 out_w x out_h 画布上拼接多张商品图（透明底）。
        """
        ow = max(1, int(out_w))
        oh = max(1, int(out_h))
        canvas = Image.new("RGBA", (ow, oh), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        valid = [p for p in paths if p and os.path.exists(p)]
        if not valid:
            draw.rectangle([(0, 0), (ow - 1, oh - 1)], outline=(180, 180, 190, 255), width=max(1, ow // 256))
            return canvas

        n = len(valid)
        mode = stitch_mode or "single_main_aux"
        dist = distribution or "grid_auto_square"

        def paste_in_cell(idx: int, cell: tuple[int, int, int, int]) -> None:
            if idx >= len(valid):
                return
            cx, cy, cw, ch = cell
            im = _open_rgba(valid[idx])
            if im is None:
                return
            im.thumbnail((max(1, cw - 4), max(1, ch - 4)))
            px = cx + (cw - im.width) // 2
            py = cy + (ch - im.height) // 2
            canvas.paste(im, (px, py), im)

        if n == 1:
            fit = (os.getenv("MVP_PRODUCT_IMAGE_FIT") or "contain").strip().lower()
            if fit == "cover":
                m0 = _resize_cover(_open_rgba(valid[0]), ow, oh)
            else:
                m0 = _resize_contain_rgba(_open_rgba(valid[0]), ow, oh)
            if m0:
                canvas.paste(m0, (0, 0), m0)
            return canvas

        if mode == "single_main_aux":
            if dist == "single_main_center_aux_right_stack":
                split = int(ow * 0.72)
                fit = (os.getenv("MVP_PRODUCT_IMAGE_FIT") or "contain").strip().lower()
                main = (
                    _resize_cover(_open_rgba(valid[0]), max(40, split - 10), oh)
                    if fit == "cover"
                    else _resize_contain_rgba(_open_rgba(valid[0]), max(40, split - 10), oh)
                )
                if main:
                    canvas.paste(main, (max(0, (split - main.width) // 2), max(0, (oh - main.height) // 2)), main)
                aux_x = split + 6
                aux_w = max(40, ow - aux_x)
                aux_count = n - 1
                each_h = max(48, (oh - (aux_count - 1) * 6) // max(aux_count, 1))
                y0 = 0
                for j in range(aux_count):
                    aux = _open_rgba(valid[1 + j])
                    if not aux:
                        continue
                    aux.thumbnail((aux_w - 6, each_h - 6))
                    canvas.paste(
                        aux,
                        (aux_x + max(0, (aux_w - aux.width) // 2), y0 + max(0, (each_h - aux.height) // 2)),
                        aux,
                    )
                    y0 += each_h + 6
                return canvas

            # single_main_bottom_aux_top_strip（默认）：上方副图带，下方主图
            strip_h = max(int(oh * 0.22), 72)
            main_h = oh - strip_h - 6
            main_h = max(80, main_h)
            aux_n = n - 1
            if aux_n > 0:
                cell_w = max(40, (ow - (aux_n - 1) * 6) // aux_n)
                x0 = 0
                for j in range(aux_n):
                    aux = _open_rgba(valid[1 + j])
                    if not aux:
                        continue
                    aux.thumbnail((cell_w - 6, strip_h - 6))
                    canvas.paste(aux, (x0 + (cell_w - aux.width) // 2, (strip_h - aux.height) // 2), aux)
                    x0 += cell_w + 6
            fit = (os.getenv("MVP_PRODUCT_IMAGE_FIT") or "contain").strip().lower()
            main = (
                _resize_cover(_open_rgba(valid[0]), ow, main_h)
                if fit == "cover"
                else _resize_contain_rgba(_open_rgba(valid[0]), ow, main_h)
            )
            if main:
                y_main = oh - main_h
                canvas.paste(main, (0, y_main), main)
            return canvas

        # multi_equal_grid
        cap = self._max_cells_for_distribution(dist, n)
        use = valid[:cap]
        k = len(use)
        cols, rows = _grid_shape(k, dist)
        cw = ow // cols
        ch = oh // rows
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= k:
                    return canvas
                cell = (c * cw, r * ch, cw, ch)
                paste_in_cell(idx, cell)
                idx += 1
        return canvas

    @staticmethod
    def _draw_text_with_soft_shadow(
        draw: ImageDraw.ImageDraw,
        xy: tuple[int, int],
        text: str,
        fill: tuple[int, int, int],
        font: Any,
        shadow: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        x, y = xy
        for ox, oy in ((1, 1), (1, 0), (0, 1), (-1, -1)):
            draw.text((x + ox, y + oy), text, fill=shadow, font=font)
        draw.text((x, y), text, fill=fill, font=font)

    def _draw_titles(
        self,
        draw: ImageDraw.ImageDraw,
        size: int,
        product_name: str,
        tagline: str,
        margin: int,
        title_anchor: str,
    ) -> None:
        font_title, font_sub = load_title_fonts(size)
        fill_title = (26, 28, 34)
        fill_sub = (88, 90, 96)
        gap = int(size * 0.048)

        if title_anchor == "top_center":
            bbox = draw.textbbox((0, 0), product_name, font=font_title)
            tw = bbox[2] - bbox[0]
            x = max(margin, (size - tw) // 2)
            self._draw_text_with_soft_shadow(draw, (x, margin), product_name, fill_title, font_title)
            bbox2 = draw.textbbox((0, 0), tagline, font=font_sub)
            tw2 = bbox2[2] - bbox2[0]
            x2 = max(margin, (size - tw2) // 2)
            self._draw_text_with_soft_shadow(draw, (x2, margin + gap), tagline, fill_sub, font_sub)
        else:
            self._draw_text_with_soft_shadow(draw, (margin, margin), product_name, fill_title, font_title)
            self._draw_text_with_soft_shadow(draw, (margin, margin + gap), tagline, fill_sub, font_sub)

    def render_mvp(
        self,
        plan: dict[str, Any],
        product_name: str,
        tagline: str,
        request_id: str,
        product_image_path: str | None,
        variant_index: int,
        image_stats: dict[str, Any] | None = None,
        background_raster_bytes: bytes | None = None,
        product_image_paths: list[str] | None = None,
        promo_image_bytes: bytes | None = None,
        promo_copy: str = "",
    ) -> dict[str, Any]:
        start = time.perf_counter()
        layout = plan.get("layout", {})
        background = plan.get("background", {})
        mvp = plan.get("_mvp_layout") or {}
        size = int(layout.get("canvas_size", 2048) or 2048)
        size = max(512, min(4096, size))

        candidates = list(background.get("candidates", []) or [])
        used_asset = False
        if background_raster_bytes:
            try:
                with Image.open(BytesIO(background_raster_bytes)) as bg_img:
                    base = bg_img.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
                soften = Image.new("RGB", (size, size), (14, 14, 18))
                base = Image.blend(base, soften, 0.07)
                used_asset = True
            except OSError:
                base, used_asset = self._load_or_procedural_background(
                    size=size,
                    variant_index=variant_index,
                    candidates=candidates,
                    image_stats=image_stats,
                )
        else:
            base, used_asset = self._load_or_procedural_background(
                size=size,
                variant_index=variant_index,
                candidates=candidates,
                image_stats=image_stats,
            )
        image = base.copy()
        draw = ImageDraw.Draw(image)

        margin = int(size * (int(layout.get("safe_margin_percent", 8) or 8) / 100.0))
        title_anchor = str(mvp.get("title_anchor", "top_left"))
        self._draw_titles(draw, size, product_name, tagline, margin, title_anchor)

        box = mvp.get("product_box") or [0.22, 0.30, 0.56, 0.56]
        fx, fy, fw, fh = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        box_w = int(size * fw)
        box_h = int(size * fh)
        box_x = int(size * fx)
        box_y = int(size * fy)

        inset_r = float(os.getenv("MVP_PRODUCT_BOX_INSET_RATIO", "0.03"))
        inset_r = max(0.0, min(0.12, inset_r))
        box_w_i = max(64, int(box_w * (1 - 2 * inset_r)))
        box_h_i = max(64, int(box_h * (1 - 2 * inset_r)))
        box_x_i = box_x + (box_w - box_w_i) // 2
        box_y_i = box_y + (box_h - box_h_i) // 2

        paths: list[str] = []
        if product_image_paths:
            paths = [p for p in product_image_paths if p]
        elif product_image_path:
            paths = [product_image_path]

        stitch = plan.get("_mvp_stitch") or {}
        stitch_mode = str(stitch.get("stitch_mode", "single_main_aux"))
        distribution = str(stitch.get("distribution", "single_main_bottom_aux_top_strip"))

        promo_band_ratio = 0.0
        pc = (promo_copy or "").strip()
        if pc:
            promo_band_ratio = float(os.getenv("MVP_PROMO_BAND_RATIO", "0.2"))
        promo_band_ratio = max(0.0, min(0.45, promo_band_ratio))
        promo_h = int(round(box_h_i * promo_band_ratio)) if promo_band_ratio > 0 else 0
        prod_h = max(80, box_h_i - promo_h)

        radius = max(10, min(size // 55, 40))
        panel = Image.new("RGBA", (box_w_i, box_h_i), (0, 0, 0, 0))
        pdraw = ImageDraw.Draw(panel)
        pdraw.rounded_rectangle(
            [0, 0, box_w_i - 1, box_h_i - 1],
            radius=radius,
            fill=(255, 255, 255, 242),
            outline=(218, 214, 206),
            width=max(1, size // 640),
        )
        image.paste(panel, (box_x_i, box_y_i), panel)

        collage = self._build_product_collage_rgba(
            paths=paths,
            out_w=max(1, box_w_i),
            out_h=max(1, prod_h),
            stitch_mode=stitch_mode,
            distribution=distribution,
        )
        pasted = bool(paths) and any(os.path.exists(p) for p in paths)
        image.paste(collage, (box_x_i, box_y_i), collage)

        if promo_h > 0:
            band_top = box_y_i + prod_h
            bg_band = (248, 247, 244, 255)
            overlay = Image.new("RGBA", (box_w_i, promo_h), bg_band)
            image.paste(overlay, (box_x_i, band_top), overlay)
            if promo_image_bytes:
                try:
                    with Image.open(BytesIO(promo_image_bytes)) as pr:
                        pr_rgba = pr.convert("RGBA").resize(
                            (max(1, box_w_i - 8), max(1, promo_h - 8)), Image.Resampling.LANCZOS
                        )
                    px = box_x_i + (box_w_i - pr_rgba.width) // 2
                    py = band_top + (promo_h - pr_rgba.height) // 2
                    image.paste(pr_rgba, (px, py), pr_rgba)
                except OSError:
                    draw.rectangle(
                        [(box_x_i + 2, band_top + 2), (box_x_i + box_w_i - 2, band_top + promo_h - 2)],
                        outline=(190, 190, 198),
                        width=max(1, size // 512),
                    )
            else:
                draw.rectangle(
                    [(box_x_i + 2, band_top + 2), (box_x_i + box_w_i - 2, band_top + promo_h - 2)],
                    outline=(200, 200, 208),
                    width=max(1, size // 512),
                )

        if not pasted:
            draw.rectangle(
                [(box_x, box_y), (box_x + box_w, box_y + box_h)],
                outline=(160, 160, 168),
                width=max(2, size // 512),
            )

        ts = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        short_id = request_id.replace("-", "")[:8] if request_id else uuid.uuid4().hex[:8]
        filename = f"poster_mvp_{ts}_{short_id}_v{variant_index}.png"
        save_path = os.path.join(self.output_dir, filename)
        image.save(save_path, format="PNG")
        thumb_name = f"poster_mvp_{ts}_{short_id}_v{variant_index}_thumb.png"
        thumb_path = os.path.join(self.output_dir, thumb_name)
        thumb = image.copy()
        thumb.thumbnail((512, 512))
        thumb.save(thumb_path, format="PNG")
        render_ms = int((time.perf_counter() - start) * 1000)
        self.run_storage_cleanup()

        return {
            "image_path": save_path.replace("\\", "/"),
            "thumbnail_path": thumb_path.replace("\\", "/"),
            "width": size,
            "height": size,
            "background_hit": used_asset,
            "product_pasted": pasted,
            "render_ms": render_ms,
            "variant_index": variant_index,
        }

    def render(
        self,
        plan: dict[str, Any],
        product_name: str,
        tagline: str,
        request_id: str,
        product_image_path: str | None = None,
    ) -> dict[str, Any]:
        """兼容旧接口：无 MVP 排版信息时使用简化居中布局。"""
        if plan.get("_mvp_layout"):
            p_list = [product_image_path] if product_image_path else []
            return self.render_mvp(
                plan=plan,
                product_name=product_name,
                tagline=tagline,
                request_id=request_id,
                product_image_path=product_image_path,
                variant_index=0,
                image_stats=None,
                product_image_paths=p_list,
                promo_copy="",
            )

        start = time.perf_counter()
        layout = plan.get("layout", {})
        background = plan.get("background", {})
        size = int(layout.get("canvas_size", 2048) or 2048)
        size = max(512, min(4096, size))

        candidates = background.get("candidates", [])
        bg_hit = bool(candidates)

        image = Image.new("RGB", (size, size), color=(244, 241, 236))
        draw = ImageDraw.Draw(image)

        margin = int(size * 0.08)
        f1, f2 = load_title_fonts(size)
        draw.text((margin, margin), product_name, fill=(35, 35, 35), font=f1)
        draw.text((margin, margin + int(size * 0.08)), tagline, fill=(70, 70, 70), font=f2)

        box_w = int(size * 0.56)
        box_h = int(size * 0.56)
        box_x = (size - box_w) // 2
        box_y = int(size * 0.32)
        pasted = False
        if product_image_path:
            try:
                if os.path.exists(product_image_path):
                    with Image.open(product_image_path) as prod_img:
                        prod_img = prod_img.convert("RGBA")
                        prod_img.thumbnail((box_w, box_h))
                        paste_x = box_x + (box_w - prod_img.width) // 2
                        paste_y = box_y + (box_h - prod_img.height) // 2
                        image.paste(prod_img, (paste_x, paste_y), prod_img)
                        pasted = True
            except OSError:
                pasted = False
        if not pasted:
            draw.rectangle(
                [(box_x, box_y), (box_x + box_w, box_y + box_h)],
                outline=(155, 155, 155),
                width=max(2, size // 512),
            )

        ts = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        short_id = request_id.replace("-", "")[:8] if request_id else uuid.uuid4().hex[:8]
        filename = f"poster_{ts}_{short_id}.png"
        save_path = os.path.join(self.output_dir, filename)
        image.save(save_path, format="PNG")
        thumb_name = f"poster_{ts}_{short_id}_thumb.png"
        thumb_path = os.path.join(self.output_dir, thumb_name)
        thumb = image.copy()
        thumb.thumbnail((512, 512))
        thumb.save(thumb_path, format="PNG")
        render_ms = int((time.perf_counter() - start) * 1000)
        self.run_storage_cleanup()

        return {
            "image_path": save_path.replace("\\", "/"),
            "thumbnail_path": thumb_path.replace("\\", "/"),
            "width": size,
            "height": size,
            "background_hit": bg_hit,
            "product_pasted": pasted,
            "render_ms": render_ms,
        }
