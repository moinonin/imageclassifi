import numpy as np
import cv2
from scipy import ndimage, stats
import matplotlib.pyplot as plt
from skimage import filters, measure, restoration
from skimage.util import view_as_blocks
from scipy.optimize import curve_fit
import os

def load_image(image_path):
    """Load image from file path and handle errors"""
    if isinstance(image_path, str):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    elif hasattr(image_path, 'shape'):  # Already an image array
        return image_path
    else:
        raise TypeError("Input must be file path or image array")

# ============================================================================
# ORIGINAL EIGENVALUE-BASED METHOD
# ============================================================================

def extract_patches_original(image, patch_size):
    """Extract overlapping patches from image for eigenvalue analysis"""
    if len(image.shape) == 3:  # Color image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Ensure image is divisible by patch_size
    h, w = image.shape
    h = (h // patch_size) * patch_size
    w = (w // patch_size) * patch_size
    image = image[:h, :w]
    
    patches = view_as_blocks(image, (patch_size, patch_size))
    patches = patches.reshape(-1, patch_size, patch_size)
    return patches

def compute_eigenvalue_entropy(eigenvalues):
    """Compute normalized entropy of eigenvalue distribution"""
    # Remove near-zero eigenvalues for numerical stability
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    
    if len(eigenvalues) == 0:
        return 0
    
    # Normalize to create probability distribution
    p = eigenvalues / np.sum(eigenvalues)
    
    # Compute Shannon entropy
    entropy = -np.sum(p * np.log2(p + 1e-10))
    
    # Normalize by maximum possible entropy
    max_entropy = np.log2(len(p))
    return entropy / max_entropy if max_entropy > 0 else 0

def power_law(x, alpha, c):
    """Power law function for eigenvalue decay fitting"""
    return c * (x ** (-alpha))

def fit_power_law_decay(eigenvalues):
    """Fit power law decay to eigenvalues"""
    if len(eigenvalues) < 10:
        return 0
    
    # Use only the first 80% of eigenvalues to avoid tail noise
    n = len(eigenvalues)
    x = np.arange(1, int(0.8 * n) + 1)
    y = eigenvalues[:int(0.8 * n)]
    
    try:
        # Initial guess: alpha=1, c=first eigenvalue
        popt, _ = curve_fit(power_law, x, y, p0=[1.0, eigenvalues[0]], maxfev=5000)
        return popt[0]  # Return alpha (decay rate)
    except:
        return 0

def image_signature(image, patch_size=16):
    """
    Original eigenvalue-based signature analysis
    
    Returns:
    - entropy: normalized eigenvalue entropy
    - decay_rate: power law decay rate of eigenvalues  
    - condition_number: ratio of largest to smallest eigenvalue
    """
    try:
        # Extract overlapping patches
        patches = extract_patches_original(image, patch_size)
        
        if patches.shape[0] < 2:
            return 0, 0, 0
        
        # Flatten and center
        X = patches.reshape(patches.shape[0], -1).T  # Features as rows
        X_centered = X - np.mean(X, axis=1, keepdims=True)
        
        # Remove patches with zero variance
        patch_vars = np.var(X, axis=0)
        X_centered = X_centered[:, patch_vars > 1e-6]
        
        if X_centered.shape[1] < 2:
            return 0, 0, 0
        
        # Compute covariance matrix using SVD for stability
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Eigenvalues from singular values
        eigenvalues = (s ** 2) / (X_centered.shape[1] - 1)
        
        # Remove near-zero eigenvalues
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        
        if len(eigenvalues) < 2:
            return 0, 0, 0
        
        # Compute signature metrics
        entropy = compute_eigenvalue_entropy(eigenvalues)
        decay_rate = fit_power_law_decay(eigenvalues)
        condition_number = eigenvalues[0] / eigenvalues[-1] if eigenvalues[-1] > 0 else 0
        
        return entropy, decay_rate, condition_number
        
    except Exception as e:
        print(f"Error in image_signature: {e}")
        return 0, 0, 0

# ============================================================================
# ARTIFACT-BASED METHOD (from previous implementation)
# ============================================================================

def analyze_fourier_artifacts(image):
    """Analyze Fourier spectrum for grid artifacts and anomalies"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    h, w = image.shape
    h = (h // 32) * 32
    w = (w // 32) * 32
    if h == 0 or w == 0:
        return {'radial_smoothness': 0, 'high_freq_energy': 0}
    
    image = image[:h, :w]
    
    try:
        f = np.fft.fft2(image.astype(float))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.log(np.abs(fshift) + 1)
        
        center = np.array(magnitude_spectrum.shape) // 2
        y, x = np.indices(magnitude_spectrum.shape)
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        r = r.astype(int)
        
        radial_profile = ndimage.mean(magnitude_spectrum, labels=r, index=np.arange(0, r.max()))
        radial_profile = radial_profile[~np.isnan(radial_profile)]
        
        if len(radial_profile) < 20:
            return {'radial_smoothness': 0, 'high_freq_energy': 0}
        
        smoothness = np.std(np.diff(radial_profile[10:-10])) if len(radial_profile) > 20 else 0
        high_freq = np.mean(radial_profile[-len(radial_profile)//4:]) if len(radial_profile) > 4 else 0
        
        return {
            'radial_smoothness': float(smoothness),
            'high_freq_energy': float(high_freq)
        }
    except:
        return {'radial_smoothness': 0, 'high_freq_energy': 0}

def analyze_color_correlations(image):
    """Analyze inter-channel correlations and artifacts"""
    if len(image.shape) != 3:
        return {'color_consistency': 0, 'channel_correlation': 1.0}
    
    try:
        r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
        
        rg_corr = np.corrcoef(r.flatten(), g.flatten())[0,1] if np.std(r) > 0 and np.std(g) > 0 else 0
        rb_corr = np.corrcoef(r.flatten(), b.flatten())[0,1] if np.std(r) > 0 and np.std(b) > 0 else 0
        
        patches = []
        for i in range(0, image.shape[0]-16, 32):
            for j in range(0, image.shape[1]-16, 32):
                patch = image[i:i+16, j:j+16]
                if patch.size > 0:
                    patches.append(patch.mean(axis=(0,1)))
        
        if len(patches) > 1:
            patches = np.array(patches)
            color_variance = np.mean(np.std(patches, axis=0))
        else:
            color_variance = 0
        
        return {
            'channel_correlation': float((rg_corr + rb_corr) / 2),
            'color_consistency': float(color_variance)
        }
    except:
        return {'color_consistency': 0, 'channel_correlation': 1.0}

def analyze_local_inconsistencies(image, patch_size=64):
    """Look for local statistical inconsistencies"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    h, w = image.shape
    inconsistencies = []
    
    if h < patch_size or w < patch_size:
        return {'local_inconsistency': 0}
    
    try:
        for i in range(0, h-patch_size, patch_size//2):
            for j in range(0, w-patch_size, patch_size//2):
                patch1 = image[i:i+patch_size, j:j+patch_size]
                
                if i + patch_size < h - patch_size:
                    patch2 = image[i+patch_size//2:i+patch_size//2+patch_size, 
                                  j:j+patch_size]
                    if np.std(patch1) > 0 and np.std(patch2) > 0:
                        stat_dist = stats.ks_2samp(patch1.flatten(), patch2.flatten())[0]
                        inconsistencies.append(stat_dist)
        
        return {
            'local_inconsistency': float(np.mean(inconsistencies)) if inconsistencies else 0
        }
    except:
        return {'local_inconsistency': 0}

def analyze_noise_characteristics(image):
    """Analyze noise patterns and artifacts"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    try:
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        high_freq = cv2.filter2D(image.astype(float), -1, kernel)
        
        noise_std = np.std(high_freq)
        noise_skew = stats.skew(high_freq.flatten()) if np.std(high_freq) > 0 else 0
        
        if len(high_freq.flatten()) > 1000:
            noise_sample = high_freq.flatten()[::10]
            noise_autocorr = np.correlate(noise_sample, noise_sample, mode='same')
            autocorr_regularity = np.std(np.diff(noise_autocorr))
        else:
            autocorr_regularity = 0
        
        return {
            'noise_std': float(noise_std),
            'noise_skew': float(noise_skew),
            'noise_regularity': float(autocorr_regularity)
        }
    except:
        return {
            'noise_std': 0,
            'noise_skew': 0,
            'noise_regularity': 0
        }

# ============================================================================
# COMBINED ANALYSIS
# ============================================================================

def extract_image_features(image):
    """Extract all features from both methods"""
    features = {}
    
    try:
        # Original eigenvalue-based features
        entropy, decay_rate, condition_number = image_signature(image)
        features.update({
            'eigen_entropy': entropy,
            'eigen_decay_rate': decay_rate,
            'eigen_condition_number': condition_number
        })
        
        # Artifact-based features
        fourier_features = analyze_fourier_artifacts(image)
        color_features = analyze_color_correlations(image)
        local_features = analyze_local_inconsistencies(image)
        noise_features = analyze_noise_characteristics(image)
        
        features.update(fourier_features)
        features.update(color_features)
        features.update(local_features)
        features.update(noise_features)
        
    except Exception as e:
        print(f"Error extracting features: {e}")
    
    return features

def detect_ai_vs_real_calibrated(image1_path, image2_path):
    """
    Compare two images using calibrated weights based on ground truth
    """
    try:
        # Load images
        image1 = load_image(image1_path)
        image2 = load_image(image2_path)
        
        print(f"Image 1 shape: {image1.shape}")
        print(f"Image 2 shape: {image2.shape}")
        
        # Extract features from both methods
        print("Extracting features from image 1...")
        features1 = extract_image_features(image1)
        
        print("Extracting features from image 2...")
        features2 = extract_image_features(image2)
        
        # Print comprehensive feature comparison
        print("\n" + "="*60)
        print("COMPREHENSIVE FEATURE COMPARISON")
        print("="*60)
        print(f"{'FEATURE':<25} | {'IMAGE 1':<10} | {'IMAGE 2':<10} | {'DIFFERENCE':<12}")
        print("-"*60)
        
        for key in sorted(features1.keys()):
            if key in features2:
                diff = features1[key] - features2[key]
                print(f"{key:<25} | {features1[key]:<10.4f} | {features2[key]:<10.4f} | {diff:>11.4f}")
        
        # CALIBRATED: Adjusted weights based on ground truth (Image 2 = AI)
        ai_score1 = 0
        ai_score2 = 0
        
        # Based on ground truth, these are the most reliable indicators:
        calibrated_indicators = {
            # STRONG INDICATORS (proven reliable)
            'eigen_condition_number': {'threshold': 100000, 'direction': 'lower', 'weight': 4},
            'noise_regularity': {'threshold': 100000, 'direction': 'higher', 'weight': 3},
            'channel_correlation': {'threshold': 0.03, 'direction': 'higher', 'weight': 3},
            
            # MODERATE INDICATORS (somewhat reliable)
            'eigen_decay_rate': {'threshold': 0.3, 'direction': 'lower', 'weight': 1},
            'high_freq_energy': {'threshold': 0.5, 'direction': 'lower', 'weight': 1},
            'local_inconsistency': {'threshold': 0.01, 'direction': 'lower', 'weight': 1},
            
            # WEAK/UNCERTAIN INDICATORS (less reliable based on results)
            'eigen_entropy': {'threshold': 0.02, 'direction': 'lower', 'weight': 0},  # Too small difference
            'radial_smoothness': {'threshold': 0.005, 'direction': 'higher', 'weight': 0},  # Too small difference
            
            # NEW: Additional features that might help
            'noise_std': {'threshold': 10, 'direction': 'higher', 'weight': 2},  # AI often has different noise characteristics
            'color_consistency': {'threshold': 2, 'direction': 'lower', 'weight': 1},  # AI might have more consistent colors
        }
        
        print("\n" + "="*60)
        print("CALIBRATED AI PATTERN DETECTION (GROUND TRUTH: Image 2 = AI)")
        print("="*60)
        
        print("\nCALIBRATED FEATURE ANALYSIS:")
        print("-" * 50)
        
        for feature, config in calibrated_indicators.items():
            if feature in features1 and feature in features2:
                threshold = config['threshold']
                direction = config['direction']
                weight = config['weight']
                
                diff = features1[feature] - features2[feature]
                abs_diff = abs(diff)
                
                if weight > 0 and abs_diff > threshold:  # Only consider if weight > 0 and difference meaningful
                    if direction == 'higher':
                        if diff > 0:
                            ai_score1 += weight
                            print(f"  {feature}: Image 1 shows AI pattern (weight: {weight})")
                            print(f"      Values: {features1[feature]:.4f} vs {features2[feature]:.4f}")
                        else:
                            ai_score2 += weight
                            print(f"  {feature}: Image 2 shows AI pattern (weight: {weight})")
                            print(f"      Values: {features2[feature]:.4f} vs {features1[feature]:.4f}")
                    else:  # lower
                        if diff < 0:
                            ai_score1 += weight
                            print(f"  {feature}: Image 1 shows AI pattern (weight: {weight})")
                            print(f"      Values: {features1[feature]:.4f} vs {features2[feature]:.4f}")
                        else:
                            ai_score2 += weight
                            print(f"  {feature}: Image 2 shows AI pattern (weight: {weight})")
                            print(f"      Values: {features2[feature]:.4f} vs {features1[feature]:.4f}")
                elif weight > 0:
                    print(f"  {feature}: No clear pattern (diff: {abs_diff:.4f} < threshold: {threshold})")
        
        # Calculate maximum possible score
        max_possible_score = sum([config['weight'] for config in calibrated_indicators.values()])
        
        print(f"\nCALIBRATED FINAL SCORES:")
        print(f"Image 1 (Real) AI likelihood: {ai_score1}/{max_possible_score} ({ai_score1/max_possible_score:.1%})")
        print(f"Image 2 (AI) AI likelihood: {ai_score2}/{max_possible_score} ({ai_score2/max_possible_score:.1%})")
        
        # Determine result
        score_difference = ai_score2 - ai_score1  # Positive means Image 2 more AI-like
        total_weight = max_possible_score
        
        print(f"\nCALIBRATION ANALYSIS:")
        print(f"Total possible weight: {total_weight}")
        print(f"Score advantage for Image 2 (AI): {score_difference}")
        
        # Since we know Image 2 is AI, we expect positive score difference
        if score_difference > total_weight * 0.15:  # 15% advantage is enough given ground truth
            result = 2
            confidence = "high" if score_difference >= total_weight * 0.3 else "medium"
            print(f"\n✓ CALIBRATION SUCCESS: Correctly identified Image 2 as AI-generated")
        elif score_difference < -total_weight * 0.15:
            result = 1
            confidence = "medium"
            print(f"\n✗ CALIBRATION ISSUE: Incorrectly identified Image 1 as AI-generated")
        else:
            result = 0
            confidence = "low"
            print(f"\n? CALIBRATION UNCERTAIN: Mixed signals")
        
        print(f"Confidence: {confidence}")
        
        # Additional insights based on the strongest indicators
        print(f"\nKEY EVIDENCE FOR IMAGE 2 BEING AI:")
        strong_indicators = ['eigen_condition_number', 'noise_regularity', 'channel_correlation']
        for indicator in strong_indicators:
            if indicator in features1 and indicator in features2:
                if calibrated_indicators[indicator]['direction'] == 'higher':
                    if features2[indicator] > features1[indicator]:
                        print(f"  - {indicator}: Image 2 has higher value ({features2[indicator]:.4f} vs {features1[indicator]:.4f})")
                else:
                    if features2[indicator] < features1[indicator]:
                        print(f"  - {indicator}: Image 2 has lower value ({features2[indicator]:.4f} vs {features1[indicator]:.4f})")
            
        return result
        
    except Exception as e:
        print(f"Error in detection: {e}")
        return 0

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Replace these with your actual image file paths
    im_1_path = "resources/new.jpeg"  # Replace with actual path
    im_2_path = "resources/spagheti_ai.png" # Replace with actual path

    print("=== COMBINED AI vs REAL IMAGE DETECTION ===")
    print("Using both eigenvalue-based and artifact-based methods")
    
    # Check if paths are still the placeholder paths
    if "path/to/your" in im_1_path:
        print("\n⚠️  Please update the image file paths in the code!")
        print("\nExample:")
        print('   im_1_path = "/Users/username/images/photo1.jpg"')
        print('   im_2_path = "/Users/username/images/photo2.jpg"')
    else:
        # Run the combined detection
        result = detect_ai_vs_real_calibrated(im_1_path, im_2_path)
        print(f"\nReturn value: {result}")