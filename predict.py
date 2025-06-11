import os
import torch
import pandas as pd
import numpy as np
from model.model import ResNet3D
from utils.dataloader import mri_dset
from transforms.load_transform import load_transforms
from sklearn.metrics import mean_absolute_error
from monai.data import ThreadDataLoader
import argparse

# Argumenty
parser = argparse.ArgumentParser(description='Brain age prediction using pretrained ResNet3D ensemble')
parser.add_argument('--input-csv', required=True, help='CSV with paths and ages')
parser.add_argument('--model-dir', required=True, help='Directory with saved model .pth files')
parser.add_argument('--output-csv', default='predictions.csv', help='Where to save predictions')
parser.add_argument('--batch-size', default=16, type=int, help='Batch size for inference')
args = parser.parse_args()

# Konfiguracja
cfg = {
    'img_dim': [160, 192, 160],
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
}

# Wczytanie danych
print(f'Loading data from {args.input_csv}')
df = pd.read_csv(args.input_csv, usecols=['uid', 'path_registered', 'age_at_scan', 'partition'])
transforms = load_transforms(cfg, random_chance=0)


# Dataset i DataLoader tylko z testem
dset_test = mri_dset(df, partition='test', is_training=False, input_transform=transforms)
loader_test = ThreadDataLoader(dset_test, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Inicjalizacja modeli
print(f'Loading models from {args.model_dir}')
model_names = [f'ResNet3D_3x_{i}' for i in range(5)]
models = {}

# Próbka do inicjalizacji modelu
sample_img, _, _ = dset_test[0]

for name in model_names:
    model = ResNet3D(sample_img.shape, width_f=3)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, f'{name}.pth'), map_location=cfg['device']))
    model = model.to(cfg['device']).eval()
    models[name] = model

# Predykcja
all_results = []
print('Starting predictions...')

with torch.no_grad():
    for imgs, ages, uids in loader_test:
        imgs = imgs.to(cfg['device'])
        preds_ensemble = []

        for name, model in models.items():
            preds, _ = model(imgs)
            preds_ensemble.append(preds.squeeze().cpu().numpy())

        preds_ensemble = np.stack(preds_ensemble, axis=0)
        mean_preds = np.mean(preds_ensemble, axis=0)

        for uid, age, pred in zip(uids, ages, mean_preds):
            all_results.append({'uid': uid, 'age_at_scan': age.item(), 'predicted_age': pred})

# Zapis wyników
print('Saving predictions...')
df_out = pd.DataFrame(all_results)
df_out.to_csv(args.output_csv, index=False)
print(f'Predictions saved to {args.output_csv}')

# MAE
mae = mean_absolute_error(df_out['age_at_scan'], df_out['predicted_age'])
print(f'MAE (ensemble): {mae:.2f}')
