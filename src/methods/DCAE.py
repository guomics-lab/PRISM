import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from models import DCAE

class DCAEImputer(pl.LightningModule):
    def __init__(self, full_data_tensor, full_mask_tensor, 
                ae_dim=64, mask_predictor_hidden_dim=128, 
                lambda_mask=0, num_encoder_blocks=2, num_decoder_blocks=2,
                dilation=2, learning_rate=2e-5, batch_size=64):
        super().__init__()
        self.full_data_tensor = full_data_tensor
        self.full_mask_tensor = full_mask_tensor

        self.save_hyperparameters(ignore=['full_data_tensor', 'full_mask_tensor'])

        feature_dim = self.full_data_tensor.size(1)
        
        self.model = DCAE(input_feature_dim=feature_dim, 
                            ae_dim=ae_dim, 
                            mask_predictor_hidden_dim=mask_predictor_hidden_dim,
                            num_encoder_blocks=num_encoder_blocks,
                            num_decoder_blocks=num_decoder_blocks,
                            dilation=dilation)
        
        self.batch_size = batch_size    
        self.learning_rate = learning_rate
        self.lambda_mask = lambda_mask
   

    def forward(self, x):
        return self.model(x)
    
    def train_dataloader(self):
        train_dataset = torch.utils.data.TensorDataset(
            self.full_data_tensor * self.full_mask_tensor,  
            self.full_data_tensor, 
            self.full_mask_tensor
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=20,
            persistent_workers=True
        )
    
    def training_step(self, batch, batch_idx):
        input_batch, target_batch, mask_batch = batch  
        loss_points = (target_batch != 0)
        x_tilde, predicted_mask_logits = self.model(input_batch)
        recon_loss = F.mse_loss(
                x_tilde[loss_points],
                target_batch[loss_points]
            )

        if self.lambda_mask > 0:
            true_mask = (target_batch != 0).float()
            mask_loss = F.binary_cross_entropy_with_logits(
                predicted_mask_logits, 
                true_mask
            )
        else:
            mask_loss = torch.tensor(0.0, device=self.device)
            
        total_loss = recon_loss + self.lambda_mask * mask_loss 

        self.log("train/loss", total_loss, on_epoch=True, prog_bar=True)
        self.log("train/recon_loss", recon_loss, on_epoch=True) 
        if self.lambda_mask > 0:
            self.log("train/mask_loss", mask_loss, on_epoch=True) 
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)        
        return optimizer


    def get_imputed_data(self):
        self.eval()
        with torch.no_grad():
            input_data = self.full_data_tensor * self.full_mask_tensor
            x_tilde, _ = self.model(input_data)
            imputed_data = self.full_data_tensor.clone()

            unobserved_mask = (self.full_mask_tensor == 0)
            imputed_data[unobserved_mask] = x_tilde[unobserved_mask]
            
            return imputed_data