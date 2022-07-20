from models.generator import TSCNet
from models import discriminator
import os
from data import dataloader
import torch.nn.functional as F
from utils import *
import logging
from torchinfo import summary
import argparse
import subprocess
from google.cloud import storage

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=120, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--log_interval", type=int, default=500)
parser.add_argument("--decay_epoch", type=int, default=30, help="epoch from which to start lr decay")
parser.add_argument("--init_lr", type=float, default=5e-4, help="initial learning rate")
parser.add_argument("--cut_len", type=int, default=16000*2, help="cut length, default is 2 seconds in denoise "
                                                                 "and dereverberation")
parser.add_argument("--data_dir", type=str, default='dir to VCTK-DEMAND dataset',
                    help="directory of VCTK+DEMAND dataset")
parser.add_argument("--save_model_dir", type=str, default='./saved_model',
                    help="directory of saved model")
parser.add_argument("--storage_bucket", type=str, default='',
                    help="Google Cloud Storage bucket for saving checkpoints")
parser.add_argument("--remote_save_model_dir", type=str, default='./saved_model',
                    help="directory within Google Cloud Storage bucket for saving checkpoints")
parser.add_argument("--loss_weights", type=list, default=[0.1, 0.9, 0.2, 0.05],
                    help="weights of RI components, magnitude, time loss, and Metric Disc")
args = parser.parse_args()
logging.basicConfig(level=logging.INFO)


class Trainer:
    def __init__(self, train_ds, test_ds):
        self.n_fft = 400
        self.hop = 100
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.model = TSCNet(num_channel=64, num_features=self.n_fft // 2 + 1).cuda()
        summary(self.model, [(1, 2, args.cut_len//self.hop+1, int(self.n_fft/2)+1)])
        self.discriminator = discriminator.Discriminator(ndf=16).cuda()
        summary(self.discriminator, [(1, 1, int(self.n_fft/2)+1, args.cut_len//self.hop+1),
                                     (1, 1, int(self.n_fft/2)+1, args.cut_len//self.hop+1)])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.init_lr)
        self.optimizer_disc = torch.optim.AdamW(self.discriminator.parameters(), lr=2*args.init_lr)


    def train_step(self, batch):
        clean = batch[0].cuda()
        noisy = batch[1].cuda()
        one_labels = torch.ones(args.batch_size).cuda()

        # Normalization
        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy ** 2.0), dim=-1))
        noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
        noisy, clean = torch.transpose(noisy * c, 0, 1), torch.transpose(clean * c, 0, 1)

        self.optimizer.zero_grad()
        noisy_spec = torch.stft(noisy, self.n_fft, self.hop, window=torch.hamming_window(self.n_fft).cuda(),
                                onesided=True)
        clean_spec = torch.stft(clean, self.n_fft, self.hop, window=torch.hamming_window(self.n_fft).cuda(),
                                onesided=True)
        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
        clean_spec = power_compress(clean_spec)
        clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
        clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)

        est_real, est_imag = self.model(noisy_spec)
        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
        est_mag = torch.sqrt(est_real**2 + est_imag**2)
        clean_mag = torch.sqrt(clean_real**2 + clean_imag**2)

        predict_fake_metric = self.discriminator(clean_mag, est_mag)
        gen_loss_GAN = F.mse_loss(predict_fake_metric.flatten(), one_labels.float())

        loss_mag = F.mse_loss(est_mag, clean_mag)
        loss_ri = F.mse_loss(est_real, clean_real) + F.mse_loss(est_imag, clean_imag)

        est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
        est_audio = torch.istft(est_spec_uncompress, self.n_fft, self.hop,
                                window=torch.hamming_window(self.n_fft).cuda(), onesided=True)

        time_loss = torch.mean(torch.abs(est_audio - clean))
        length = est_audio.size(-1)
        loss = args.loss_weights[0] * loss_ri + args.loss_weights[1] * loss_mag + args.loss_weights[2] * time_loss \
               + args.loss_weights[3] * gen_loss_GAN
        loss.backward()
        self.optimizer.step()

        est_audio_list = list(est_audio.detach().cpu().numpy())
        clean_audio_list = list(clean.cpu().numpy()[:, :length])
        pesq_score = discriminator.batch_pesq(clean_audio_list, est_audio_list)

        # The calculation of PESQ can be None due to silent part
        if pesq_score is not None:
            self.optimizer_disc.zero_grad()
            predict_enhance_metric = self.discriminator(clean_mag, est_mag.detach())
            predict_max_metric = self.discriminator(clean_mag, clean_mag)
            discrim_loss_metric = F.mse_loss(predict_max_metric.flatten(), one_labels) + \
                                  F.mse_loss(predict_enhance_metric.flatten(), pesq_score)
            discrim_loss_metric.backward()
            self.optimizer_disc.step()
        else:
            discrim_loss_metric = torch.tensor([0.])

        return loss.item(), discrim_loss_metric.item()

    @torch.no_grad()
    def test_step(self, batch):
        clean = batch[0].cuda()
        noisy = batch[1].cuda()
        one_labels = torch.ones(args.batch_size).cuda()

        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy ** 2.0), dim=-1))
        noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
        noisy, clean = torch.transpose(noisy * c, 0, 1), torch.transpose(clean * c, 0, 1)

        noisy_spec = torch.stft(noisy, self.n_fft, self.hop, window=torch.hamming_window(self.n_fft).cuda(),
                                onesided=True)
        clean_spec = torch.stft(clean, self.n_fft, self.hop, window=torch.hamming_window(self.n_fft).cuda(),
                                onesided=True)
        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
        clean_spec = power_compress(clean_spec)
        clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
        clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)

        est_real, est_imag = self.model(noisy_spec)
        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
        est_mag = torch.sqrt(est_real ** 2 + est_imag ** 2)
        clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2)

        predict_fake_metric = self.discriminator(clean_mag, est_mag)
        gen_loss_GAN = F.mse_loss(predict_fake_metric.flatten(), one_labels.float())

        loss_mag = F.mse_loss(est_mag, clean_mag)
        loss_ri = F.mse_loss(est_real, clean_real) + F.mse_loss(est_imag, clean_imag)

        est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
        est_audio = torch.istft(est_spec_uncompress, self.n_fft, self.hop,
                                window=torch.hamming_window(self.n_fft).cuda(), onesided=True)

        time_loss = torch.mean(torch.abs(est_audio - clean))
        length = est_audio.size(-1)
        loss = args.loss_weights[0] * loss_ri + args.loss_weights[1] * loss_mag + args.loss_weights[2] * time_loss \
               + args.loss_weights[3] * gen_loss_GAN

        est_audio_list = list(est_audio.detach().cpu().numpy())
        clean_audio_list = list(clean.cpu().numpy()[:, :length])
        pesq_score = discriminator.batch_pesq(clean_audio_list, est_audio_list)
        if pesq_score is not None:
            predict_enhance_metric = self.discriminator(clean_mag, est_mag.detach())
            predict_max_metric = self.discriminator(clean_mag, clean_mag)
            discrim_loss_metric = F.mse_loss(predict_max_metric.flatten(), one_labels) + \
                                  F.mse_loss(predict_enhance_metric.flatten(), pesq_score)
        else:
            discrim_loss_metric = torch.tensor([0.])

        return loss.item(), discrim_loss_metric.item()

    def test(self):
        self.model.eval()
        self.discriminator.eval()
        gen_loss_total = 0.
        disc_loss_total = 0.
        for idx, batch in enumerate(self.test_ds):
            step = idx + 1
            loss, disc_loss = self.test_step(batch)
            gen_loss_total += loss
            disc_loss_total += disc_loss
        gen_loss_avg = gen_loss_total / step
        disc_loss_avg = disc_loss_total / step

        template = 'Generator loss: {}, Discriminator loss: {}'
        logging.info(
            template.format(gen_loss_avg, disc_loss_avg))

        return gen_loss_avg

    def train(self):
        scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.decay_epoch, gamma=0.5)
        scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizer_disc, step_size=args.decay_epoch, gamma=0.5)

        # Load saved checkpoint
        initial_epoch = 0
        checkpoint_filenames = []
        for filename in os.listdir(args.save_model_dir):
            if filename.startswith('CMGAN_epoch'):
                checkpoint_filenames.append(filename)

        if len(checkpoint_filenames) > 0:
            checkpoint_filenames.sort()
            checkpoint_path = os.path.join(args.save_model_dir, checkpoint_filenames[-1])
            checkpoint = torch.load(checkpoint_path)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.optimizer_disc.load_state_dict(checkpoint['optimizer_disc_state_dict'])
            scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
            scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
            initial_epoch = checkpoint['epoch'] + 1

        for epoch in range(initial_epoch, args.epochs):
            epoch_start = time.process_time()
            self.model.train()
            self.discriminator.train()

            interval_start = time.process_time()
            for idx, batch in enumerate(self.train_ds):
                step = idx + 1
                loss, disc_loss = self.train_step(batch)
                template = 'Epoch {}, Step {}, loss: {}, disc_loss: {}'
                if (step % args.log_interval) == 0:
                    logging.info(template.format(epoch, step, loss, disc_loss))
                    interval = time.process_time() - interval_start
                    logging.info(f'Completed interval in {interval} seconds')
                    interval_start = time.process_time()
            gen_loss = self.test()
            filename = os.path.join('CMGAN_epoch_' + str(epoch) + '_' + str(gen_loss)[:5])
            path = os.path.join(args.save_model_dir, filename)
            if not os.path.exists(args.save_model_dir):
                os.makedirs(args.save_model_dir)

            scheduler_G.step()
            scheduler_D.step()

            epoch_duration = time.process_time() - epoch_start
            logging.info(f'Completed epoch {epoch} in {epoch_duration} seconds')

            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'discriminator_state_dict': self.discriminator.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'optimizer_disc_state_dict': self.optimizer_disc.state_dict(),
                'scheduler_G_state_dict': scheduler_G.state_dict(),
                'scheduler_D_state_dict': scheduler_D.state_dict(),
                'gen_loss': gen_loss,
                'disc_loss': disc_loss},
            path)

            if args.storage_bucket != "" and args.remote_save_model_dir != "":
                remote_storage_path = os.path.join(args.remote_save_model_dir, filename)
                upload_blob(args.storage_bucket, path, remote_storage_path)
                # Assumes user is already authenticated


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )


def main():
    print(args)
    available_gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print(available_gpus)
    train_ds, test_ds = dataloader.load_data(args.data_dir, args.batch_size, 2, args.cut_len)
    trainer = Trainer(train_ds, test_ds)
    trainer.train()


if __name__ == '__main__':
    main()
