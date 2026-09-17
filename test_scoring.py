"""
test_scoring.py - Automated verification for CLIP scoring and formula calculations.
"""
import os
import sys
from PIL import Image

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def run_tests():
    print("[1/5] Importing app module and dependencies...")
    from app import compute_clip_similarity, calculate_score, DEVICE
    print(f"      Running on device: {DEVICE}")

    # Generate sample images if not present
    from create_samples import create_sample_images
    sample_dir = "sample_images"
    create_sample_images(sample_dir)

    ref_path = os.path.join(sample_dir, "reference_sunset.png")
    match_path = os.path.join(sample_dir, "candidate_sunset_match.png")
    diff_path = os.path.join(sample_dir, "candidate_cyberpunk_divergent.png")

    ref_img = Image.open(ref_path)
    match_img = Image.open(match_path)
    diff_img = Image.open(diff_path)

    print("[2/5] Testing self-similarity (identical image)...")
    self_sim = compute_clip_similarity(ref_img, ref_img)
    print(f"      Self similarity: {self_sim * 100:.2f}%")
    assert self_sim > 0.99, f"Expected self-similarity ~1.0, got {self_sim}"

    print("[3/5] Testing visual similarity comparison...")
    sim_match = compute_clip_similarity(ref_img, match_img)
    sim_diff = compute_clip_similarity(ref_img, diff_img)
    print(f"      Match candidate similarity: {sim_match * 100:.2f}%")
    print(f"      Divergent candidate similarity: {sim_diff * 100:.2f}%")
    assert sim_match > sim_diff, f"Similar image ({sim_match}) should score higher than divergent image ({sim_diff})"

    print("[4/5] Testing complete scoring formula...")
    # Formula: Final Score = (Similarity %) - (Lambda * Tokens)
    tokens = 20
    lam = 0.3
    sim_pct, deduction, final_score = calculate_score(ref_img, match_img, tokens, lam)
    
    expected_deduction = round(lam * tokens, 2)
    expected_final = round(sim_pct - expected_deduction, 2)
    
    print(f"      Sim %: {sim_pct}%, Deduction: {deduction}, Final Score: {final_score}")
    assert deduction == expected_deduction, f"Expected deduction {expected_deduction}, got {deduction}"
    assert final_score == expected_final, f"Expected final score {expected_final}, got {final_score}"

    print("[5/5] Testing default lambda (0.3) with 0 tokens...")
    sim_pct_0, deduction_0, final_score_0 = calculate_score(ref_img, match_img, 0, 0.3)
    assert deduction_0 == 0.0
    assert final_score_0 == sim_pct_0

    print("\n[PASSED] All scoring tests passed successfully!")

if __name__ == "__main__":
    run_tests()
