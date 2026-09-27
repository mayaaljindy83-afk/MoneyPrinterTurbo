"""Ready-made subtitle looks, so users do not have to tune every colour.

Sizes are defined for a 1080p canvas; ``video.generate_video`` scales them
for 720p output. Every preset uses the bundled Tajawal font, which covers
both Arabic and Latin text.
"""

from __future__ import annotations

SUBTITLE_STYLES: dict[str, dict] = {
    # White text with a black outline: readable on any footage.
    "classic": {
        "font_name": "Tajawal-Bold.ttf",
        "font_size": 64,
        "text_fore_color": "#FFFFFF",
        "stroke_color": "#000000",
        "stroke_width": 3,
        "text_background_color": False,
        "rounded_subtitle_background": False,
        "subtitle_position": "bottom",
    },
    # Semi-transparent rounded box, like news and documentary channels.
    "boxed": {
        "font_name": "Tajawal-Bold.ttf",
        "font_size": 58,
        "text_fore_color": "#FFFFFF",
        "stroke_color": "#000000",
        "stroke_width": 0,
        "text_background_color": "#000000",
        "rounded_subtitle_background": True,
        "subtitle_position": "bottom",
    },
    # Yellow text with a heavy outline, popular on YouTube.
    "youtube_yellow": {
        "font_name": "Tajawal-ExtraBold.ttf",
        "font_size": 66,
        "text_fore_color": "#FFD60A",
        "stroke_color": "#000000",
        "stroke_width": 4,
        "text_background_color": False,
        "rounded_subtitle_background": False,
        "subtitle_position": "bottom",
    },
    # Big centred words for TikTok / Reels.
    "shorts_bold": {
        "font_name": "Tajawal-ExtraBold.ttf",
        "font_size": 84,
        "text_fore_color": "#FFFFFF",
        "stroke_color": "#000000",
        "stroke_width": 5,
        "text_background_color": False,
        "rounded_subtitle_background": False,
        "subtitle_position": "two_thirds_bottom",
        "subtitle_animation": "pop_spring",
    },
    # Small and quiet, for calm or cinematic videos.
    "minimal": {
        "font_name": "Tajawal-Bold.ttf",
        "font_size": 50,
        "text_fore_color": "#F5F5F5",
        "stroke_color": "#000000",
        "stroke_width": 2,
        "text_background_color": False,
        "rounded_subtitle_background": False,
        "subtitle_position": "bottom",
    },
}


def style_names() -> list[str]:
    return list(SUBTITLE_STYLES)


def apply_subtitle_style(params) -> bool:
    """Copy the preset named by ``params.subtitle_style`` onto ``params``.

    Returns False (and leaves ``params`` untouched) for an empty or unknown
    style, so custom settings keep working.
    """
    style = SUBTITLE_STYLES.get(str(getattr(params, "subtitle_style", "") or "").strip())
    if not style:
        return False
    for key, value in style.items():
        setattr(params, key, value)
    return True
