---
language:
- en
---

## 🛠 Training Details (RoboTwin 2.0 Simulation)

This model is a pi0.5 checkpoint fine-tuned on the **RoboTwin 2.0** simulation environment.

### Model Pedigree

* **Base Model:** Official released pi0.5 base model.
* **Framework Conversion:** The weights were converted from the original **JAX** implementation to **PyTorch** for this training pipeline.

### Training Data & Setting

The model was trained under a **multi-task** setting involving **50 distinct tasks**, blending both "Clean" and "Randomized" simulation environments:

| Data Setting | Episodes per Task | Total Episodes |
| --- | --- | --- |
| **Clean Setting** | 50 | 2,500 |
| **Randomized Setting** | 500 | 25,000 |
| **Total** | **550** | **27,500** |

### Training Configuration

```
    TrainConfig(
        name="pi05_base_finetune_on_robotwin_clean_randomized_joint_training",
        project_name="pi05_finetune",
        exp_name="robotwin_clean_randomized_joint_training",
        model=pi0_config.Pi0Config(
            pi05=True, 
            action_horizon=32,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("./pi05_base/params"),
        pytorch_weight_path="./pi05_base_torch",
        lr_schedule=_optimizer.ConstantScheduleWithWarmup, # we defined this scheduler ourselves
        #optimizer=_optimizer.AdamW,
        data=LeRobotAlohaDataConfig(
            repo_id= "clean_randomized_joint_training",
            data_dir= HF_LEROBOT_HOME / "robotwin" ,
            multi_task=True, # we use the MultiLeRobotDataset and therefore modified the official repo slightly to better start the train
            base_config=DataConfig(
                prompt_from_task=True,
                random_prompt_from_task=True, # we add this option for the training setting of RoboTwin(every episode of the same task randomly selects instruction for training from the same bunch of instructions)
            ),
            assets=AssetsConfig(
                assets_dir="./assets/pi05_base_finetune_on_robotwin_clean_randomized_joint_training",
                asset_id="robotwin_clean_randomized_joint_training",
            ),
            adapt_to_pi=True,
            default_prompt=DEFAULT_PROMPT,
            use_delta_joint_actions=True,
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "high_image",
                                "cam_left_wrist": "left_wrist_image",
                                "cam_right_wrist": "right_wrist_image",
                            },
                            "state": "state",
                            "actions": "actions",
                            "prompt": "prompt",
                        }
                    ),
                ]
            ),
            action_sequence_keys=("actions",)
        ),
        seed=42,
        batch_size=128,
        num_workers=16,
        num_train_steps=1000000,
        log_interval=100,
        val_interval=1000,
        save_interval=5000,
        keep_period=5000,
        resume=True,
        wandb_enabled=True,
    ),
```