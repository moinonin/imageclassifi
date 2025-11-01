import numpy as np
import pandas as pd
import cv2
from scipy import ndimage, stats
import matplotlib.pyplot as plt
from skimage import filters, measure, restoration
from skimage.util import view_as_blocks
from scipy.optimize import curve_fit
import os

im_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "./resources"))

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
    Compare two images using calibrated analysis based on ground truth
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
        
        # CALIBRATED: Feature analysis without weights
        ai_indicators_image1 = 0
        ai_indicators_image2 = 0
        
        # Feature configuration with thresholds and directions (no weights)
        feature_config = {
            # STRONG INDICATORS (proven reliable)
            'eigen_condition_number': {'threshold': 100000, 'direction': 'higher'},
            'noise_regularity': {'threshold': 100000, 'direction': 'lower'},
            'channel_correlation': {'threshold': 0.03, 'direction': 'higher'},
            
            # MODERATE INDICATORS
            'eigen_decay_rate': {'threshold': 0.3, 'direction': 'higher'},
            'high_freq_energy': {'threshold': 0.5, 'direction': 'lower'},
            'local_inconsistency': {'threshold': 0.01, 'direction': 'higher'},
            'noise_std': {'threshold': 10, 'direction': 'lower'},
            'color_consistency': {'threshold': 2, 'direction': 'higher'},
            
            # WEAK INDICATORS (included but with lower confidence)
            'eigen_entropy': {'threshold': 0.02, 'direction': 'lower'},
            'radial_smoothness': {'threshold': 0.005, 'direction': 'lower'},
            'noise_skew': {'threshold': 0.05, 'direction': 'higher'}
        }
        
        print("\n" + "="*60)
        print("CALIBRATED AI PATTERN DETECTION")
        print("="*60)
        
        print("\nFEATURE ANALYSIS (Unweighted):")
        print("-" * 50)
        
        analyzed_features = []
        
        for feature, config in feature_config.items():
            if feature in features1 and feature in features2:
                threshold = config['threshold']
                direction = config['direction']
                
                diff = features1[feature] - features2[feature]
                abs_diff = abs(diff)
                
                if abs_diff > threshold:
                    if direction == 'higher':
                        if diff > 0:
                            ai_indicators_image1 += 1
                            analyzed_features.append((feature, "Image 1", features1[feature], features2[feature]))
                            print(f"  {feature}: Image 1 shows AI pattern")
                        else:
                            ai_indicators_image2 += 1
                            analyzed_features.append((feature, "Image 2", features2[feature], features1[feature]))
                            print(f"  {feature}: Image 2 shows AI pattern")
                    else:  # direction == 'lower'
                        if diff < 0:
                            ai_indicators_image1 += 1
                            analyzed_features.append((feature, "Image 1", features1[feature], features2[feature]))
                            print(f"  {feature}: Image 1 shows AI pattern")
                        else:
                            ai_indicators_image2 += 1
                            analyzed_features.append((feature, "Image 2", features2[feature], features1[feature]))
                            print(f"  {feature}: Image 2 shows AI pattern")
                    print(f"      Values: {features1[feature]:.4f} vs {features2[feature]:.4f} (diff: {diff:.4f})")
                else:
                    print(f"  {feature}: No clear pattern (diff: {abs_diff:.4f} < threshold: {threshold})")
        
        total_indicators = ai_indicators_image1 + ai_indicators_image2
        
        print(f"\nUNWEIGHTED FINAL SCORES:")
        print(f"Image 1 AI indicators: {ai_indicators_image1}")
        print(f"Image 2 AI indicators: {ai_indicators_image2}")
        
        if total_indicators > 0:
            print(f"Image 1 AI likelihood: {ai_indicators_image1}/{total_indicators} ({ai_indicators_image1/total_indicators:.1%})")
            print(f"Image 2 AI likelihood: {ai_indicators_image2}/{total_indicators} ({ai_indicators_image2/total_indicators:.1%})")
        
        # Determine result based on indicator count
        indicator_difference = ai_indicators_image2 - ai_indicators_image1
        
        print(f"\nANALYSIS SUMMARY:")
        print(f"Total significant indicators: {total_indicators}")
        print(f"Indicator advantage for Image 2: {indicator_difference}")
        
        # Decision logic with confidence levels
        if total_indicators == 0:
            result = 0
            confidence = "very low"
            print(f"\n? UNCERTAIN: No clear indicators found")
        elif indicator_difference > 0:
            result = 2
            if indicator_difference >= 3:
                confidence = "high"
            elif indicator_difference >= 2:
                confidence = "medium"
            else:
                confidence = "low"
            print(f"\n✓ IDENTIFIED: Image 2 shows more AI patterns")
        elif indicator_difference < 0:
            result = 1
            if indicator_difference <= -3:
                confidence = "high"
            elif indicator_difference <= -2:
                confidence = "medium"
            else:
                confidence = "low"
            print(f"\n✓ IDENTIFIED: Image 1 shows more AI patterns")
        else:
            result = 0
            confidence = "low"
            print(f"\n? UNCERTAIN: Equal number of AI indicators")
        
        print(f"Confidence: {confidence}")
        
        # Additional insights based on the strongest indicators
        if analyzed_features:
            print(f"\nKEY EVIDENCE ANALYSIS:")
            strong_features = ['eigen_condition_number', 'noise_regularity', 'channel_correlation', 'noise_std']
            for feature in strong_features:
                for analyzed_feature, image, ai_value, real_value in analyzed_features:
                    if analyzed_feature == feature:
                        print(f"  - {feature}: {image} has AI pattern ({ai_value:.4f} vs {real_value:.4f})")
            
        return result
        
    except Exception as e:
        print(f"Error in detection: {e}")
        return 0

def detect_ai_single_image(image_path):
    """
    Detect if a single image is AI-generated using calibrated feature analysis
    Returns: True if AI-generated, False if real, None if uncertain
    """
    try:
        # Load image
        image = load_image(image_path)
        
        print(f"Image shape: {image.shape}")
        
        # Extract features
        print("Extracting features...")
        features = extract_image_features(image)
        
        # Print feature values
        print("\n" + "="*50)
        print("FEATURE ANALYSIS")
        print("="*50)
        for key, value in sorted(features.items()):
            print(f"{key:<25} | {value:<10.4f}")
        
        # CALIBRATED THRESHOLDS based on ground truth analysis
        # From the comparison where Image 1 (AI) vs Image 2 (Real):
        # Image 1 (AI) had: higher eigen_decay_rate, local_inconsistency, noise_skew, radial_smoothness
        # Image 1 (AI) had: lower channel_correlation, color_consistency, eigen_condition_number, eigen_entropy, high_freq_energy, noise_regularity, noise_std
        
        ai_feature_patterns = {
            # STRONG AI INDICATORS (based on significant differences)
            'eigen_condition_number': {'threshold': 5000000, 'direction': 'below'},  # AI had 729K vs Real 10M
            'noise_regularity': {'threshold': 2800000, 'direction': 'below'},        # AI had 2M vs Real 3.6M
            'noise_skew': {'threshold': 0.5, 'direction': 'above'},                  # AI had 1.73 vs Real -0.15
            
            # MODERATE AI INDICATORS
            'channel_correlation': {'threshold': 0.82, 'direction': 'below'},        # AI had 0.75 vs Real 0.90
            'local_inconsistency': {'threshold': 0.21, 'direction': 'above'},        # AI had 0.235 vs Real 0.186
            'eigen_decay_rate': {'threshold': 4.0, 'direction': 'above'},            # AI had 4.16 vs Real 3.82
            
            # WEAKER INDICATORS
            'color_consistency': {'threshold': 48.0, 'direction': 'below'},          # AI had 47.6 vs Real 49.7
            'eigen_entropy': {'threshold': 0.215, 'direction': 'below'},             # AI had 0.20 vs Real 0.23
            'high_freq_energy': {'threshold': 7.55, 'direction': 'below'},           # AI had 7.49 vs Real 7.62
            'noise_std': {'threshold': 90.0, 'direction': 'below'},                  # AI had 84.3 vs Real 95.8
            'radial_smoothness': {'threshold': 0.062, 'direction': 'above'}          # AI had 0.065 vs Real 0.059
        }
        
        ai_indicator_count = 0
        total_checked_features = 0
        strong_ai_indicators = 0
        
        print("\n" + "="*50)
        print("CALIBRATED AI PATTERN DETECTION")
        print("="*50)
        
        ai_indicators = []
        real_indicators = []
        
        for feature, config in ai_feature_patterns.items():
            if feature in features:
                total_checked_features += 1
                value = features[feature]
                threshold = config['threshold']
                direction = config['direction']
                
                is_ai_indicator = False
                reason = ""
                
                if direction == 'above':
                    if value > threshold:
                        is_ai_indicator = True
                        reason = f"above AI threshold ({value:.4f} > {threshold})"
                    else:
                        reason = f"below AI threshold ({value:.4f} <= {threshold})"
                else:  # direction == 'below'
                    if value < threshold:
                        is_ai_indicator = True
                        reason = f"below AI threshold ({value:.4f} < {threshold})"
                    else:
                        reason = f"above AI threshold ({value:.4f} >= {threshold})"
                
                if is_ai_indicator:
                    ai_indicator_count += 1
                    # Check if this is a strong indicator
                    is_strong = feature in ['eigen_condition_number', 'noise_regularity', 'noise_skew']
                    if is_strong:
                        strong_ai_indicators += 1
                    
                    ai_indicators.append((feature, value, threshold, reason, is_strong))
                    strength = "STRONG" if is_strong else "moderate"
                    print(f"✓ {feature}: {strength} AI indicator - {reason}")
                else:
                    real_indicators.append((feature, value, threshold, reason))
                    print(f"  {feature}: Real image pattern - {reason}")
        
        print(f"\nSUMMARY:")
        print(f"Total AI indicators: {ai_indicator_count}/{total_checked_features}")
        print(f"Strong AI indicators: {strong_ai_indicators}/3")
        
        # Calculate confidence scores
        if total_checked_features > 0:
            ai_ratio = ai_indicator_count / total_checked_features
            print(f"AI indicator ratio: {ai_ratio:.1%}")
        
        # DECISION LOGIC with calibrated thresholds
        # Based on ground truth analysis where Image 1 (AI) showed clear patterns
        
        if ai_indicator_count >= 8:
            result = True
            confidence = "very high"
            feat = features
            print(f"\n🎯 CONCLUSION: AI-GENERATED (confidence: {confidence})")
            print(f"   Overwhelming evidence with {ai_indicator_count} AI indicators")
            
        elif ai_indicator_count >= 6:
            result = True
            if strong_ai_indicators >= 2:
                confidence = "high"
            else:
                confidence = "medium"
            feat = features
            print(f"\n🎯 CONCLUSION: AI-GENERATED (confidence: {confidence})")
            print(f"   Strong evidence with {ai_indicator_count} AI indicators")
            
        elif ai_indicator_count >= 4:
            if strong_ai_indicators >= 2:
                result = True
                confidence = "medium"
                feat = features
                print(f"\n🎯 CONCLUSION: LIKELY AI-GENERATED (confidence: {confidence})")
                print(f"   Moderate evidence with {ai_indicator_count} AI indicators ({strong_ai_indicators} strong)")
            else:
                result = None
                confidence = "low"
                feat = features
                print(f"\n🤔 CONCLUSION: UNCERTAIN (confidence: {confidence})")
                print(f"   Mixed signals with {ai_indicator_count} AI indicators")
                
        elif ai_indicator_count >= 2:
            if strong_ai_indicators >= 1:
                result = None
                confidence = "low"
                feat = features
                print(f"\n🤔 CONCLUSION: UNCERTAIN (confidence: {confidence})")
                print(f"   Weak evidence with {ai_indicator_count} AI indicators ({strong_ai_indicators} strong)")
            else:
                result = False
                confidence = "medium"
                feat = features
                print(f"\n🎯 CONCLUSION: REAL IMAGE (confidence: {confidence})")
                print(f"   Few AI indicators ({ai_indicator_count}) suggest real image")
                
        else:
            result = False
            confidence = "high" if ai_indicator_count == 0 else "medium"
            feat = features
            print(f"\n🎯 CONCLUSION: REAL IMAGE (confidence: {confidence})")
            print(f"   Minimal AI indicators ({ai_indicator_count})")
        
        # Print key evidence
        if ai_indicators:
            print(f"\nKEY AI EVIDENCE:")
            # Sort by strong indicators first
            strong_evidences = [ind for ind in ai_indicators if ind[4]]
            moderate_evidences = [ind for ind in ai_indicators if not ind[4]]
            
            for feature, value, threshold, reason, is_strong in strong_evidences[:3]:
                print(f"  - {feature}: {reason}")
            for feature, value, threshold, reason, is_strong in moderate_evidences[:3]:
                print(f"  - {feature}: {reason}")
        
        if real_indicators and ai_indicator_count < 6:
            print(f"\nKEY REAL IMAGE EVIDENCE:")
            strong_real_indicators = [ind for ind in real_indicators if ind[0] in ['eigen_condition_number', 'noise_regularity', 'noise_skew']]
            for feature, value, threshold, reason in strong_real_indicators[:3]:
                print(f"  - {feature}: {reason}")

        return result, confidence, feat

    except Exception as e:
        print(f"Error in AI detection: {e}")
        return None
import os
from PIL import Image
import numpy as np

def detect_ai_in_folder(folder_path):
    """
    Detect AI-generated images in a folder and return results for DataFrame
    Returns: List of dictionaries with filename, is_ai (ground truth), result (outcome, confidence, features)
    """
    results = []
    
    # Supported image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    
    # Check if directory exists
    if not os.path.exists(folder_path):
        print(f"Error: Directory '{folder_path}' does not exist.")
        return results
    
    # Process each file in the directory
    for filename in os.listdir(folder_path):
        # Check file extension
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            file_path = os.path.join(folder_path, filename)
            
            print(f"\n{'='*60}")
            print(f"PROCESSING: {filename}")
            print(f"{'='*60}")
            
            try:
                # Detect AI using your existing function
                outcome, confidence, features = detect_ai_single_image(file_path)
                
                # Determine ground truth from filename
                is_ai_ground_truth = "_ai" in filename.lower()
                
                # Store results
                results.append({
                    'filename': filename,
                    'is_ai': is_ai_ground_truth,
                    'result': (outcome, confidence, features)
                })
                
                print(f"✓ Completed: {filename}")
                
            except Exception as e:
                print(f"❌ Error processing {filename}: {str(e)}")
                # Still add to results but with error
                results.append({
                    'filename': filename,
                    'is_ai': "_ai" in filename.lower(),
                    'result': (None, "error", {})
                })
    
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE: {len(results)} images analyzed")
    print(f"{'='*60}")
    
    return results

# Then use it like this:
def process_folder_and_display_results():
    # Define your image directory
    #im_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../resources"))
    
    # Detect AI in all images
    results = detect_ai_in_folder(im_dir)
    
    # Create DataFrame (this matches your existing code structure)
    df = pd.DataFrame(results)
    df["is_ai"] = df["filename"].str.contains(r"_ai(?=\.[^.]+$)", regex=True)
    df[["outcome", "confidence", "features"]] = pd.DataFrame(df["result"].tolist(), index=df.index)
    df["detected"] = np.where(df["outcome"] == True, "AI-Generated", "Human-Created")
    
    # Print summary
    print("\n=== SUMMARY RESULTS ===")
    df.drop(columns=["outcome"], inplace=True)
    df = df[["filename", "is_ai", "result"]]
    
    #print(df.to_string(index=False))
    
    return df

# Alternative: If you want to use the improved AI pattern detection
def detect_ai_in_folder_with_improved_classification(folder_path=im_dir):
    """
    Same as above but uses the improved weighted scoring system
    """
    results = []
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    
    if not os.path.exists(folder_path):
        print(f"Error: Directory '{folder_path}' does not exist.")
        return results
    
    for filename in os.listdir(folder_path):
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            file_path = os.path.join(folder_path, filename)
            
            print(f"\n{'='*60}")
            print(f"PROCESSING: {filename}")
            print(f"{'='*60}")
            
            try:
                # Use the improved detection
                outcome, confidence, features = detect_ai_single_image_improved(file_path)
                
                is_ai_ground_truth = "_ai" in filename.lower()
                
                results.append({
                    'filename': filename,
                    'is_ai': is_ai_ground_truth,
                    'result': (outcome, confidence, features)
                })
                
                print(f"✓ Completed: {filename}")
                
            except Exception as e:
                print(f"❌ Error processing {filename}: {str(e)}")
                results.append({
                    'filename': filename,
                    'is_ai': "_ai" in filename.lower(),
                    'result': (None, "error", {})
                })
    
    return results

def detect_ai_single_image_improved(image_path=f'{im_dir}/girl2_ai.png'):
    """
    Improved version of your detection function with better thresholds
    """
    try:
        # Load image
        image = load_image(image_path)
        
        # Extract features
        features = extract_image_features(image)
        
        # IMPROVED THRESHOLDS based on your actual data patterns
        ai_feature_patterns = {
            # STRONG AI INDICATORS (3.0 weight)
            'eigen_condition_number': {'threshold': 1000000, 'direction': 'below', 'weight': 3.0},
            'channel_correlation': {'threshold': 0.85, 'direction': 'below', 'weight': 3.0},
            'color_consistency': {'threshold': 50.0, 'direction': 'below', 'weight': 3.0},
            
            # MODERATE AI INDICATORS (2.0 weight)  
            'eigen_entropy': {'threshold': 0.45, 'direction': 'above', 'weight': 2.0},
            'noise_skew': {'threshold': 0.1, 'direction': 'above', 'weight': 2.0},
            'radial_smoothness': {'threshold': 0.1, 'direction': 'above', 'weight': 2.0},
            
            # WEAK AI INDICATORS (1.0 weight)
            'eigen_decay_rate': {'threshold': 2.0, 'direction': 'below', 'weight': 1.0},
            'high_freq_energy': {'threshold': 5.8, 'direction': 'above', 'weight': 1.0},
            'noise_std': {'threshold': 130.0, 'direction': 'below', 'weight': 1.0},
            'noise_regularity': {'threshold': 2000000, 'direction': 'below', 'weight': 1.0},
            'local_inconsistency': {'threshold': 0.1, 'direction': 'above', 'weight': 1.0}
        }
        
        total_score = 0
        max_possible_score = 0
        
        for feature, config in ai_feature_patterns.items():
            if feature in features:
                value = features[feature]
                threshold = config['threshold']
                direction = config['direction']
                weight = config['weight']
                
                max_possible_score += weight
                
                if direction == 'above' and value > threshold:
                    total_score += weight
                elif direction == 'below' and value < threshold:
                    total_score += weight
        
        # Normalize score
        normalized_score = total_score / max_possible_score if max_possible_score > 0 else 0
        
        # Improved decision logic
        if normalized_score >= 0.7:
            result = True
            confidence = "high"
        elif normalized_score >= 0.5:
            result = True
            confidence = "medium"
        elif normalized_score >= 0.3:
            result = None
            confidence = "low"
        else:
            result = False
            confidence = "high" if normalized_score < 0.2 else "medium"

        return result, confidence, features

    except Exception as e:
        print(f"Error in AI detection: {e}")
        return None, "error", {}

'''
# Main execution
if __name__ == "__main__":
    # Option 1: Use your original detection
    df = process_folder_and_display_results()
'''