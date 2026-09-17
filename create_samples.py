"""
create_samples.py
Generates sample reference and candidate images for testing the AI Prompt Competition scoring tool.
"""
import os
from PIL import Image, ImageDraw

def create_sample_images(output_dir="sample_images"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Target Reference Image: Sunset over calm ocean and mountains
    ref_img = Image.new("RGB", (512, 512), color=(255, 140, 60))
    ref_draw = ImageDraw.Draw(ref_img)
    # Sun
    ref_draw.ellipse([200, 120, 312, 232], fill=(255, 230, 120))
    # Mountains
    ref_draw.polygon([(0, 340), (160, 200), (320, 340)], fill=(90, 45, 95))
    ref_draw.polygon([(220, 340), (380, 180), (512, 340)], fill=(70, 35, 80))
    # Ocean
    ref_draw.rectangle([0, 340, 512, 512], fill=(20, 50, 100))
    # Ocean reflections
    for y in range(360, 500, 20):
        ref_draw.line([(180, y), (330, y)], fill=(240, 160, 80), width=3)
    ref_path = os.path.join(output_dir, "reference_sunset.png")
    ref_img.save(ref_path)

    # 2. Candidate A (High Similarity Match): Similar sunset and mountains
    match_img = Image.new("RGB", (512, 512), color=(250, 120, 50))
    match_draw = ImageDraw.Draw(match_img)
    # Sun slightly shifted
    match_draw.ellipse([220, 130, 330, 240], fill=(255, 240, 140))
    # Mountains
    match_draw.polygon([(0, 350), (180, 210), (340, 350)], fill=(85, 40, 90))
    match_draw.polygon([(200, 350), (360, 190), (512, 350)], fill=(65, 30, 75))
    # Ocean
    match_draw.rectangle([0, 350, 512, 512], fill=(25, 55, 110))
    for y in range(370, 500, 25):
        match_draw.line([(190, y), (340, y)], fill=(230, 150, 70), width=4)
    match_path = os.path.join(output_dir, "candidate_sunset_match.png")
    match_img.save(match_path)

    # 3. Candidate B (Low Similarity Divergent): Cyberpunk neon green grid
    diff_img = Image.new("RGB", (512, 512), color=(10, 15, 25))
    diff_draw = ImageDraw.Draw(diff_img)
    # Neon green grid
    for x in range(0, 512, 40):
        diff_draw.line([(x, 0), (x, 512)], fill=(0, 255, 130), width=1)
    for y in range(0, 512, 40):
        diff_draw.line([(0, y), (512, y)], fill=(0, 180, 255), width=1)
    # Center glowing polygon
    diff_draw.polygon([(256, 120), (380, 320), (132, 320)], outline=(0, 255, 150), width=4)
    diff_path = os.path.join(output_dir, "candidate_cyberpunk_divergent.png")
    diff_img.save(diff_path)

    print(f"Sample images created in '{output_dir}':")
    print(f" - {ref_path}")
    print(f" - {match_path}")
    print(f" - {diff_path}")

if __name__ == "__main__":
    create_sample_images()
