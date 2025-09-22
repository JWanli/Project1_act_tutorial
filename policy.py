import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        # 新增：重建损失配置
        self.recon_loss = args_override.get('recon_loss', 'l1')  # 'l1' | 'huber'
        self.huber_beta = args_override.get('huber_beta', 0.1)
        self.l1_weight = args_override.get('l1_weight', 1.0)
        print(f'KL Weight {self.kl_weight}, Recon Loss {self.recon_loss}, '
              f'Huber beta {self.huber_beta}, Recon Weight {self.l1_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

            loss_dict = dict()
            # 新增：可切换的重建损失
            if self.recon_loss == 'l1':
                all_rec = F.l1_loss(actions, a_hat, reduction='none')
            elif self.recon_loss in ('huber', 'smooth_l1'):
                all_rec = F.smooth_l1_loss(actions, a_hat, reduction='none', beta=self.huber_beta)
            else:
                raise ValueError(f'Unsupported recon_loss {self.recon_loss}')

            rec = (all_rec * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['recon'] = rec
            if self.recon_loss == 'l1':
                loss_dict['l1'] = rec
            else:
                loss_dict['huber'] = rec

            loss_dict['kl'] = total_kld[0]
            # 新增：加入重建损失权重
            loss_dict['loss'] = self.l1_weight * loss_dict['recon'] + self.kl_weight * loss_dict['kl']
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    # mu, logvar: (bs, latent_dim)
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()) # (bs, latent_dim)
    total_kld = klds.sum(1).mean(0, True) 
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
