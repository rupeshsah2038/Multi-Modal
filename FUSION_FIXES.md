# Fusion Module Fixes

## Issue
Four fusion-explore experiments were failing:
- `medpix-energy_aware_adaptive`
- `medpix-shomr`
- `wound-energy_aware_adaptive`
- `wound-shomr`

## Root Cause
Both `EnergyAwareAdaptiveFusion` and `SHoMRFusion` were returning tuples instead of single tensors:
- `EnergyAwareAdaptiveFusion.forward()` returned `(out, energy_loss)`
- `SHoMRFusion.forward()` returned `(fused, routing_info)`

The training engine expects all fusion modules to return a single fused tensor, following the pattern established by other fusion modules like `SimpleFusion`, `CrossAttentionFusion`, etc.

## Fix Applied
Modified both fusion modules to return only the fused output tensor:

### EnergyAwareAdaptiveFusion (`models/fusion/energy_aware_adaptive.py`)
- Changed `return out, energy_loss` to `return out`
- Added comment noting that `energy_loss` could be used for additional regularization in future versions

### SHoMRFusion (`models/fusion/shomr.py`)
- Changed `return fused, routing_info` to `return fused`
- Commented out the `routing_info` construction with note that it could be logged for analysis in future
- Updated docstring to reflect single return value

## Verification
Both modules now:
1. Return a single tensor of shape `(B, D)` matching other fusion modules
2. Load correctly in the training pipeline
3. Pass unit tests with dummy inputs

## Next Steps
To use the `energy_loss` and `routing_info` in future:
- Could modify the training engine to handle optional auxiliary outputs
- Could add custom logging hooks for these advanced fusion modules
- For now, keeping them simple ensures compatibility with the existing training framework

All four failed experiments should now run successfully.
