"""
Simple direct test runner
"""
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_feature_extraction import test_keystroke_feature_extraction, test_mouse_curvature_extraction
from tests.test_bot_detection import test_synthetic_untrusted_dom_event, test_zero_variance_scripted_typing
from tests.test_classification import test_genuine_vs_impostor_classification

def run_all():
    print("Running test_keystroke_feature_extraction...")
    test_keystroke_feature_extraction()
    print("  -> PASSED")

    print("Running test_mouse_curvature_extraction...")
    test_mouse_curvature_extraction()
    print("  -> PASSED")

    print("Running test_synthetic_untrusted_dom_event...")
    test_synthetic_untrusted_dom_event()
    print("  -> PASSED")

    print("Running test_zero_variance_scripted_typing...")
    test_zero_variance_scripted_typing()
    print("  -> PASSED")

    print("Running test_genuine_vs_impostor_classification...")
    test_genuine_vs_impostor_classification()
    print("  -> PASSED")

    print("\nALL 5 CORE BIOMETRIC TESTS PASSED PERFECTLY!")

if __name__ == "__main__":
    run_all()

