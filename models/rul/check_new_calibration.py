import sys
sys.path.insert(0, 'models/rul')
from calibrate_rul import load_calibration_params, BUCKET_EDGES

params = load_calibration_params()
print("Calibration params loaded successfully")
print("Saved calibration's bucket_edges:", params['bucket_edges'])
print("Code's current BUCKET_EDGES:", BUCKET_EDGES)
print("Do they match?", BUCKET_EDGES == params['bucket_edges'])

print("\nPer-bucket calibration values (from saved file):")
for b, edges in enumerate(params['bucket_edges']):
    info = params['buckets'][b]
    print(f"  bucket {edges}: n={info['n']}  bias={info['mean_residual']:.2f}  band={info['conformal_width']:.2f}")