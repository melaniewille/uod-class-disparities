from ultralytics.utils.callbacks.wb import _log_plots
import wandb as wb

def on_val_start(validator):
    """Log validation start event to WandB if not already running."""
    if not wb.run:
        # Initialize WandB if not already running
        wb.init(
            project="Ultralytics-Validation" if not hasattr(validator, "args") or not validator.args.project else 
                   str(validator.args.project).replace("/", "-"),
            name="validation" if not hasattr(validator, "args") or not validator.args.name else 
                 str(validator.args.name).replace("/", "-"),
            config=vars(validator.args) if hasattr(validator, "args") else {},
        )


def on_val_end(validator):
    """Log final validation metrics and plots to WandB."""
    # Extract metrics from validator object safely
    val_metrics = {}
    
    try:
        # Try to log basic metrics that should be available
        if hasattr(validator, "metrics"):
            # Try to log mAP metrics safely
            try:
                if hasattr(validator.metrics, "box"):
                    if hasattr(validator.metrics.box, "map50"):
                        val_metrics["val/mAP50"] = float(validator.metrics.box.map50)
                    if hasattr(validator.metrics.box, "map"):
                        val_metrics["val/mAP"] = float(validator.metrics.box.map)
            except Exception as e:
                print(f"Warning: Failed to log mAP metrics: {e}")
            
            # Try to log speed metrics safely
            try:
                if hasattr(validator.metrics, "speed"):
                    for attr_name in ["preprocess", "inference", "postprocess", "total"]:
                        if hasattr(validator.metrics.speed, attr_name):
                            try:
                                val_value = getattr(validator.metrics.speed, attr_name)
                                if isinstance(val_value, (int, float)):
                                    val_metrics[f"val/speed_{attr_name}"] = val_value
                            except:
                                pass
            except Exception as e:
                print(f"Warning: Failed to log speed metrics: {e}")
        
        # Log the extracted metrics
        if val_metrics:
            wb.run.log(val_metrics, step=getattr(validator, "epoch", 0) + 1)
        
        # Try to log validation plots safely
        try:
            if hasattr(validator, "plots"):
                _log_plots(validator.plots, step=getattr(validator, "epoch", 0) + 1)
        except Exception as e:
            print(f"Warning: Failed to log validation plots: {e}")
        
    except Exception as e:
        print(f"Warning: Error in on_val_end: {e}")
