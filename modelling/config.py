import torch

CONFIG = {"N_EPOCHS" : 20,
          "SEED" : 7457769,
          "LR" : 1e-3 / 3,
          "BATCH_SIZE" : 128,
          "DS_VERSION" : "v4",
          "MODEL_DEVICE": "cpu",
          "TRAIN_SIZE_FRACTION": 1.0,
          "loss_criterion": torch.nn.L1Loss()}

print("=========TRAINING CONFIG=========")
print(CONFIG)
