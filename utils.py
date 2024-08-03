import math
import time
import torch

from EfficientLayers import CustomConv2d


class DropSchedular:
    def __init__(self, model, drop_mode, percentage, min_percentage, total_epoch, interleave=False, warmup=0, by_epoch=True, T=10):
        # priority: warmup > interleave > drop_mode

        self.model = model
        self.drop_mode = drop_mode
        self.percentage = percentage
        self.min_percentage = min_percentage
        self.total_epoch = total_epoch

        self.warmup = warmup
        self.by_epoch = by_epoch

        # only for by_epoch and interleave
        self.last_time = True
        self.interleave = interleave

        # only for by_iteration
        self.T = T

    def get_percentage(self, epoch):
        # epoch starts from 0 to total_epoch - 1
        if epoch < self.warmup:
            return 1.0
        
        if self.interleave:
            if self.last_time:
                self.last_time = False
                return 1.0

        self.last_time = True
        if self.drop_mode == 'constant':
            return self.percentage
        elif self.drop_mode == 'linear':
            # from percentage to min_percentage in total_epoch
            return max(self.min_percentage, self.percentage - (self.percentage - self.min_percentage) * epoch / (self.total_epoch - 1))
        elif self.drop_mode == 'cosine':
            return self.min_percentage + 0.5 * (self.percentage - self.min_percentage) * (1 + math.cos(epoch * 3.14159 / (self.total_epoch - 1)))
        elif self.drop_mode == 'bar':
            return self.percentage if epoch < self.total_epoch // 2 else self.min_percentage
        else:
            raise ValueError(f"Drop mode {self.drop_mode} not recognized")
        
    def get_percentage_by_iteration(self, epoch, iteration):
        if epoch < self.warmup:
            return 1.0
        
        if self.drop_mode == 'linear':
            # from percentage to min_percentage in each T period
            return max(self.min_percentage, self.percentage - (self.percentage - self.min_percentage) * (iteration % self.T) / self.T)
            # return max(self.min_percentage, self.percentage - (self.percentage - self.min_percentage) * iteration % (self.T - 1) / (self.T - 1))
        elif self.drop_mode == 'cosine':
            return self.min_percentage + 0.5 * (self.percentage - self.min_percentage) * (1 + math.cos(iteration * 3.14159 / self.T))
        elif self.drop_mode == 'bar':
            return self.percentage if (iteration % self.T) < (self.T // 2) else self.min_percentage
        else:
            raise ValueError(f"Drop mode {self.drop_mode} not recognized")
    
    def step(self, epoch, iteration):
        if self.by_epoch:
            if iteration != 0:
                return None
            percentage = self.get_percentage(epoch)
        else:
            percentage = self.get_percentage_by_iteration(epoch, iteration)
        
        # traverse all the layers in the model
        for layer in self.model.modules():
            if isinstance(layer, CustomConv2d):
                layer.percentage = percentage

        return percentage


def train(model, task, device, train_loader, dropschedular, optimizer, criterion, epoch, writer, model_name):
    model.train()
    data_time = 0
    forward_time = 0
    backprop_time = 0

    # context = model.warmup_scope if mode == 'efficient' and epoch < warmup else nullcontext
    
    for batch_idx, (data, target) in enumerate(train_loader):

        cur_percentage = dropschedular.step(epoch, batch_idx)
        if dropschedular.by_epoch:
            if batch_idx == 0:
                writer.add_scalar('percentage / epoch', cur_percentage, epoch)
        else:
            writer.add_scalar('percentage / iteration', cur_percentage, epoch * len(train_loader) + batch_idx)

        s = time.time()
        data, target = data.to(device), target.to(device)
        if model_name == 'mlp':
            data = data.view(data.size(0), -1)
        data_time += (time.time() - s)

        optimizer.zero_grad()

        s = time.time()
        # with context(f'Warmup'):
        #     output = model(data)

        output = model(data)
        if task != 'CelebA':
            loss = criterion(output, target)
        else:
            loss = criterion(output, target.type_as(output))
        forward_time += (time.time() - s)

        s = time.time()
        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        #     loss.backward()
        loss.backward()
        optimizer.step()
        prev_backprop_time = backprop_time
        backprop_time += (time.time() - s)
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

            # record training loss every iteration
            writer.add_scalar('training loss / iteration', loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('backprop time / iteration', backprop_time - prev_backprop_time, epoch * len(train_loader) + batch_idx)
    
    print(f'Time taken for data transfer: {data_time:.2f} seconds')
    print(f'Time taken for forward pass: {forward_time:.2f} seconds')
    print(f'Time taken for backpropagation: {backprop_time:.2f} seconds')

    writer.add_scalar('backprop time / epoch', backprop_time, epoch)

    # print(prof.key_averages().table(sort_by="cpu_time_total"))


def test(model, task, device, test_loader, criterion, epoch, writer, dataset='Test'):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if 'MLP' in model.__class__.__name__:
                data = data.view(data.size(0), -1)

            output = model(data)
            if task != 'CelebA':
                test_loss += criterion(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
            else:
                test_loss += criterion(output, target.type_as(output)).item()
                pred = torch.sigmoid(output) > 0.5
                correct += (pred == target).sum().item() / target.size(1)
    
    test_loss /= len(test_loader.dataset)

    if dataset == 'Test':
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)\n')
        writer.add_scalar('test accuracy', 100. * correct / len(test_loader.dataset), epoch)
    elif dataset == 'Validation':
        print(f'\nValidation set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)\n')
        writer.add_scalar('validation accuracy', 100. * correct / len(test_loader.dataset), epoch)

    return 100. * correct / len(test_loader.dataset)


def train_ddpm(model, device, train_loader, dropschedular, optimizer, epoch, writer):
    model.train()
    data_time = 0
    forward_time = 0
    backprop_time = 0

    # context = model.warmup_scope if mode == 'efficient' and epoch < warmup else nullcontext

    for batch_idx, (data, _) in enumerate(train_loader):

        cur_percentage = dropschedular.step(epoch, batch_idx)
        if dropschedular.by_epoch:
            if batch_idx == 0:
                writer.add_scalar('percentage / epoch', cur_percentage, epoch)
        else:
            writer.add_scalar('percentage / iteration', cur_percentage, epoch * len(train_loader) + batch_idx)
        
        s = time.time()
        data = data.to(device)
        data_time += (time.time() - s)

        optimizer.zero_grad()

        s = time.time()
        # with context(f'Warmup'):
        loss = model.forward_loss(data)
        forward_time += (time.time() - s)

        s = time.time()
        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        #     loss.backward()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        optimizer.step()
        cur_backprop = (time.time() - s)
        backprop_time += cur_backprop
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            writer.add_scalar('training loss', loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('backprop time / iteration', cur_backprop, epoch * len(train_loader) + batch_idx)
    
    print(f'Time taken for data transfer: {data_time:.2f} seconds')
    print(f'Time taken for forward pass: {forward_time:.2f} seconds')
    print(f'Time taken for backpropagation: {backprop_time:.2f} seconds')

    writer.add_scalar('backprop time / epoch', backprop_time, epoch)


def test_ddpm(model, device, test_loader, epoch, writer, dataset='Test'):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)

            loss = model.forward_loss(data)
            test_loss += loss.item()  # sum up batch loss

    test_loss /= len(test_loader.dataset)

    if dataset == 'Test':
        print(f'\nTest set: Average loss: {test_loss:.4f}\n')
        writer.add_scalar('Average test reconstruction loss', test_loss, epoch)
        
        samples = model.sample(64)
        writer.add_images('samples', samples, epoch)
    elif dataset == 'Validation':
        print(f'\nValidation set: Average loss: {test_loss:.4f}\n')
        writer.add_scalar('Average val reconstruction loss', test_loss, epoch)

    return test_loss


def train_gan(latent_dim, generator, discriminator, device, train_loader, dropschedular_G, dropschedular_D, optimizer_G, optimizer_D, epoch, adversarial_loss, writer):
    generator.train()
    discriminator.train()

    data_time = 0
    forward_time = 0
    backprop_time = 0

    # context = model.warmup_scope if mode == 'efficient' and epoch < warmup else nullcontext
    
    for batch_idx, (data, _) in enumerate(train_loader):
        cur_percentage_G = dropschedular_G.step(epoch, batch_idx)
        if dropschedular_G.by_epoch and batch_idx == 0:
            writer.add_scalar('percentage_G / epoch', cur_percentage_G, epoch)
        else:
            writer.add_scalar('percentage_G / iteration', cur_percentage_G, epoch * len(train_loader) + batch_idx)
        
        if dropschedular_D is not None:
            cur_percentage_D = dropschedular_D.step(epoch, batch_idx)
            if dropschedular_D.by_epoch and batch_idx == 0:
                writer.add_scalar('percentage_D / epoch', cur_percentage_D, epoch)
            else:
                writer.add_scalar('percentage_D / iteration', cur_percentage_D, epoch * len(train_loader) + batch_idx)

        cur_backprop = 0

        s = time.time()
        real_imgs = data.to(device)
        data_time += (time.time() - s)

        valid = torch.ones(real_imgs.size(0), 1, device=device)
        fake = torch.zeros(real_imgs.size(0), 1, device=device)

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = torch.randn(data.size(0), latent_dim, device=device)
        
        s = time.time()
        # Generate a batch of images
        gen_imgs = generator(z)
        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        forward_time += (time.time() - s)

        s = time.time()
        g_loss.backward()
        optimizer_G.step()
        cur_backprop += (time.time() - s)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        s = time.time()
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)

        d_loss = (real_loss + fake_loss) / 2
        forward_time += (time.time() - s)

        s = time.time()
        d_loss.backward()
        optimizer_D.step()
        cur_backprop += (time.time() - s)

        backprop_time += cur_backprop
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tG Loss: {g_loss.item():.6f}\tD Loss: {d_loss.item():.6f}')
            writer.add_scalar('training G loss', g_loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('training D loss', d_loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('backprop time / iteration', cur_backprop, epoch * len(train_loader) + batch_idx)
    
    print(f'Time taken for data transfer: {data_time:.2f} seconds')
    print(f'Time taken for forward pass: {forward_time:.2f} seconds')
    print(f'Time taken for backpropagation: {backprop_time:.2f} seconds')

    writer.add_scalar('backprop time / epoch', backprop_time, epoch)


def test_gan(latent_dim, generator, discriminator, device, test_loader, epoch, adversarial_loss, writer, dataset='Test'):
    generator.eval()
    discriminator.eval()

    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)

            # generator loss
            z = torch.randn(data.size(0), latent_dim, device=device)
            gen_imgs = generator(z)
            disc_out = discriminator(gen_imgs)
            valid = torch.ones(disc_out.size(), device=device)
            g_loss = adversarial_loss(disc_out, valid)
            
            test_loss += g_loss.item()  # sum up batch loss

    test_loss /= len(test_loader.dataset)

    if dataset == 'Test':
        print(f'\nTest set: Average loss: {test_loss:.4f}\n')
        writer.add_scalar('Average test reconstruction loss', test_loss, epoch)
        
        samples = generator(torch.randn(64, latent_dim, device=device))
        writer.add_images('samples', samples, epoch)
    elif dataset == 'Validation':
        print(f'\nValidation set: Average loss: {test_loss:.4f}\n')
        writer.add_scalar('Average val reconstruction loss', test_loss, epoch)

    return test_loss


def save_checkpoint(model, optimizer, epoch, save_dir, model_name='latest_checkpoint'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, f'{save_dir}/{model_name}.pth')

def load_checkpoint(model, optimizer, save_dir, model_name='latest_checkpoint'):
    checkpoint = torch.load(f'{save_dir}/{model_name}.pth')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch