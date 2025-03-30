from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner

class MedNeXtPlanner(ExperimentPlanner):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def plan_experiment(self, *args, **kwargs):
        # Call the parent method to get the standard nnUNet planning
        plans = super().plan_experiment(*args, **kwargs)
        
        # Modify the patch size and batch size in the plan
        plans['configurations']['3d_fullres_mednext'] = {
                    'inherits_from': '3d_fullres',
                    'patch_size': (128, 128, 128),
                    'batch_size': 2,
                }
        plans['configurations']['2d_mednext'] = {
                    'inherits_from': '2d',
                    'patch_size': (512, 512),
                    'batch_size': 8,
                }

        self.save_plans(plans)
        return plans
    

class MedNeXtPlannerDDP(ExperimentPlanner):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def plan_experiment(self, *args, **kwargs):
        # Call the parent method to get the standard nnUNet planning
        plans = super().plan_experiment(*args, **kwargs)
        
        # Modify the patch size and batch size in the plan for DDP training.
        # Designed for 8x A100 GPUs
        plans['configurations']['3d_fullres_mednext_ddp_16'] = {
                    'inherits_from': '3d_fullres',
                    'patch_size': (128, 128, 128),
                    'batch_size': 16,
                }
        
        # Designed for 4x A100 GPUs
        plans['configurations']['3d_fullres_mednext_ddp_16'] = {
                    'inherits_from': '3d_fullres',
                    'patch_size': (128, 128, 128),
                    'batch_size': 8,
                }
        
        # Designed for 2x A100 GPUs
        plans['configurations']['3d_fullres_mednext_ddp_16'] = {
                    'inherits_from': '3d_fullres',
                    'patch_size': (128, 128, 128),
                    'batch_size': 4,
                }

        self.save_plans(plans)
        return plans