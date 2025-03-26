import torch
import itertools
from torch import nn
from torch.nn import functional as F
import lightning as L
from torch import optim
from ..models import CycleCmKAN
from ..utils.image_pool import ImagePool
from cm_kan.core import Logger
from ..metrics import (
    PSNR,
    SSIM,
    DeltaE,
)


class UnsupervisedPipeline(L.LightningModule):
    def __init__(self,
        model: CycleCmKAN,
        optimiser: str = 'adam',
        lr: float = 1e-3,
        weight_decay: float = 0,
        pretrained: bool = False,
        pretrained_model: str = None
    ) -> None:
        super(UnsupervisedPipeline, self).__init__()

        self.model = model
        self.fake_pool_a = ImagePool()
        self.fake_pool_b = ImagePool()
        self.lm = 10.0
        self.optimizer_type = optimiser
        self.lr = lr
        self.weight_decay = weight_decay
        self.mae_loss = nn.L1Loss(reduction='mean')
        self.ssim_loss = SSIM(data_range=(0, 1))
        self.de_metric = DeltaE()
        self.ssim_metric = SSIM(data_range=(0, 1))
        self.psnr_metric = PSNR(data_range=(0, 1))
        self.pretrained = pretrained
        if pretrained:
            self.automatic_optimization = False
            self.pretrained_model = pretrained_model
        
        self.save_hyperparameters(ignore=['model'])

    def _identity_loss(self, predictions, targets):
        mae_loss = self.mae_loss(predictions, targets)
        return mae_loss

    def _cycle_loss(self, predictions, targets):
        mae_loss = self.mae_loss(predictions, targets)
        ssim_loss = self.ssim_loss(predictions, targets)
        loss = mae_loss + (1 - ssim_loss) * 0.15
        return loss

    def _disc_loss(self, predictions, label):
        """
            According to the CycleGan paper, label for
            real is one and fake is zero.
        """
        if label.lower() == 'real':
            target = torch.ones_like(predictions)
        else:
            target = torch.zeros_like(predictions)
        
        return F.mse_loss(predictions, target)
    
    @staticmethod
    def _set_requires_grad(nets, requires_grad = False):

        """
        Set requies_grad=False for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """

        if not isinstance(nets, list): nets = [nets]
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def setup(self, stage: str) -> None:
        '''
        Initialize model weights before training
        '''
        if stage == 'fit' or stage is None:
            for m in self.model.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

            if self.pretrained:
                pipeline = UnsupervisedPipeline.load_from_checkpoint(
                    self.pretrained_model,
                    model=self.model,
                    optimiser=self.optimizer_type,
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    pretrained=False
                )
                self.model.gen_ab = pipeline.model.gen_ab
                self.model.gen_ba = pipeline.model.gen_ba
                del pipeline
                Logger.info(f'Initialized model weights {self.pretrained_model}.')
        
        Logger.info('Initialized model weights with [bold green]Unsupervised[/bold green] pipeline.')
        if self.pretrained:
            Logger.info('Model is in [bold green]CycleGAN training[/bold green] mode.')
        else:
            Logger.info('Model is in [bold green]Generator pre-training[/bold green] mode.')

    def configure_optimizers(self):
        if not self.pretrained:
            if self.optimizer_type == 'adam':
                optimizer = optim.Adam(itertools.chain(self.model.gen_ab.parameters(), self.model.gen_ba.parameters()),
                            lr=self.lr, weight_decay=self.weight_decay)
            else:
                raise ValueError(f'unsupported optimizer_type: {self.optimizer_type}')
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=500, T_mult=1, eta_min=1e-5
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        else:
            if self.optimizer_type == 'adam':
                optG = optim.Adam(
                    itertools.chain(self.model.gen_ab.parameters(), self.model.gen_ba.parameters()),
                    lr=2e-4, betas=(0.5, 0.999)
                )

                optD = optim.Adam(
                    itertools.chain(self.model.dis_a.parameters(), self.model.dis_b.parameters()),
                    lr=2e-4, betas=(0.5, 0.999)
                )
            else:
                raise ValueError(f'unsupported optimizer_type: {self.optimizer_type}')
            gamma = lambda epoch: 1 - max(0, epoch + 1 - 100) / 101
            schG = optim.lr_scheduler.LambdaLR(optG, lr_lambda=gamma)
            schD = optim.lr_scheduler.LambdaLR(optD, lr_lambda=gamma)
            return [optG, optD], [schG, schD]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred = self.model.gen_ab(x)
        return pred
    
    def generator_training_step(self, imgA, imgB):        
        """cycle images - using only generator nets"""
        fakeB = self.model.gen_ab(imgA)
        cycledA = self.model.gen_ba(fakeB)
        
        fakeA = self.model.gen_ba(imgB)
        cycledB = self.model.gen_ab(fakeA)
        
        sameB = self.model.gen_ab(imgB)
        sameA = self.model.gen_ba(imgA)
        
        # generator gen_ab must fool discrim dis_b so label is real = 1
        predFakeB = self.model.dis_b(fakeB)
        mseGenB = self._disc_loss(predFakeB, 'real')
        
        # generator gen_ba must fool discrim dis_a so label is real
        predFakeA = self.model.dis_a(fakeA)
        mseGenA = self._disc_loss(predFakeA, 'real')
        
        # compute extra losses
        identityLoss = self._identity_loss(sameA, imgA) + self._identity_loss(sameB, imgB)
        
        # compute cycleLosses
        cycleLoss = self._cycle_loss(cycledA, imgA) + self._cycle_loss(cycledB, imgB)
        
        # gather all losses
        extraLoss = cycleLoss + 0.5 * identityLoss
        gen_loss = mseGenA + mseGenB + self.lm * extraLoss
        self.log('gen_loss', gen_loss.item(), prog_bar=True, logger=True)
        
        # store detached generated images
        self.fakeA = fakeA.detach()
        self.fakeB = fakeB.detach()
        
        return gen_loss
    
    def discriminator_training_step(self, imgA, imgB):
        """Update Discriminator"""        
        fakeA = self.fake_pool_a.query(self.fakeA)
        fakeB = self.fake_pool_b.query(self.fakeB)
        
        # dis_a checks for domain A photos
        predRealA = self.model.dis_a(imgA)
        mseRealA = self._disc_loss(predRealA, 'real')
        
        predFakeA = self.model.dis_a(fakeA)
        mseFakeA = self._disc_loss(predFakeA, 'fake')
        
        # dis_b checks for domain B photos
        predRealB = self.model.dis_b(imgB)
        mseRealB = self._disc_loss(predRealB, 'real')
        
        predFakeB = self.model.dis_b(fakeB)
        mseFakeB = self._disc_loss(predFakeB, 'fake')
        
        # gather all losses
        dis_loss = 0.5 * (mseFakeA + mseRealA + mseFakeB + mseRealB)
        self.log('dis_loss', dis_loss.item(), prog_bar=True, logger=True)
        return dis_loss
    
    def generator_pretaining_step(self, imgA, imgB):
        reco_b = self.model.gen_ab(imgA)
        reco_a = self.model.gen_ba(imgB)
        loss = self._cycle_loss(reco_b, imgB) + self._cycle_loss(reco_a, imgA)
        self.log('pretrain_loss', loss.item(), prog_bar=True, logger=True)
        return loss


    def training_step(self, batch, batch_idx):
        img_a, img_b = batch

        if not self.pretrained:
            loss = self.generator_pretaining_step(img_a, img_b)
            return {'loss': loss}
        else:
            opt_gen, opt_disc = self.optimizers()
            sch_gen, sch_disc = self.lr_schedulers()
            
            # train generator
            self.toggle_optimizer(opt_gen)
            self._set_requires_grad([self.model.dis_a, self.model.dis_b], requires_grad=False)
            gen_loss = self.generator_training_step(img_a, img_b)
            opt_gen.zero_grad()
            self.manual_backward(gen_loss)
            opt_gen.step()
            self.untoggle_optimizer(opt_gen)
            
            # train discriminator
            self.toggle_optimizer(opt_disc)
            self._set_requires_grad([self.model.dis_a, self.model.dis_b], requires_grad=True)
            disc_loss = self.discriminator_training_step(img_a, img_b)
            opt_disc.zero_grad()
            self.manual_backward(disc_loss)
            opt_disc.step()
            self.untoggle_optimizer(opt_disc)

            if self.trainer.is_last_batch:
                sch_gen.step()
                sch_disc.step()
            return
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self(inputs)
        mae_loss = self.mae_loss(predictions, targets)
        psnr_metric = self.psnr_metric(predictions, targets)
        ssim_metric = self.ssim_metric(predictions, targets)
        de_metric = self.de_metric(predictions, targets)
        
        self.log('val_psnr', psnr_metric, prog_bar=True, logger=True)
        self.log('val_ssim', ssim_metric, prog_bar=True, logger=True)
        self.log('val_de', de_metric, prog_bar=True, logger=True)
        self.log('val_loss', mae_loss, prog_bar=True, logger=True)
        return {'loss': mae_loss}
    
    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self(inputs)
        mae_loss = self.mae_loss(predictions, targets)
        panr_metric = self.psnr_metric(predictions, targets)
        ssim_metric = self.ssim_metric(predictions, targets)
        de_metric = self.de_metric(predictions, targets)
        
        self.log('test_panr', panr_metric, prog_bar=True, logger=True)
        self.log('test_ssim', ssim_metric, prog_bar=True, logger=True)
        self.log('test_de', de_metric, prog_bar=True, logger=True)
        self.log('test_loss', mae_loss, prog_bar=True, logger=True)
        return {'loss': mae_loss}
    
    def predict_step(self, batch, batch_idx, dataloader_idx):
        inputs, _ = batch
        output = self(inputs)
        return output
