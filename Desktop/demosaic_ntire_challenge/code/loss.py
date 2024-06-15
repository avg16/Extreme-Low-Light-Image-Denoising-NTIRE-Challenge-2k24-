def make_gauss_kernel(window_size, sigma):
    gauss=torch.tensor([torch.exp(-(x-window_size//2)**2/(2*sigma**2)) for x in range(window_size)], dtype=torch.float32)
    gauss=gauss/gauss.sum()
    return gauss

def make_window(window_size, channels):
    one_d=make_gauss_kernel(window_size, 1.5).unsqueeze(1)
    two_d=one_d.mm(one_d.t()).float().unsqueeze(0).unsqueeze(0)
    window=two_d.expand(channels, 1, window_size, window_size).contiguous()
    return window

def calc_ssim(img1, img2, window_size=11, avg=True):
    _, chan, _, _=img1.size()
    window=make_window(window_size, chan).to(img1.device)
    mu1=F.conv2d(img1, window, padding=window_size // 2, groups=chan)
    mu2=F.conv2d(img2, window, padding=window_size // 2, groups=chan)
    mu1_sq=mu1.pow(2)
    mu2_sq=mu2.pow(2)
    mu1_mu2=mu1*mu2
    sigma1_sq=F.conv2d(img1*img1,window,padding=window_size//2, groups=chan)-mu1_sq
    sigma2_sq=F.conv2d(img2*img2,window,padding=window_size//2, groups=chan)-mu2_sq
    sigma12=F.conv2d(img1*img2,window,padding=window_size//2, groups=chan)-mu1_mu2
    C1=0.01**2
    C2=0.03**2
    ssim_map=((2*mu1_mu2+C1)*(2*sigma12+C2))/((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
    if avg:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, avg=True):
        super(SSIMLoss, self).__init__()
        self.window_size=window_size
        self.avg=avg

    def forward(self,img1,img2):
        return 1-calc_ssim(img1, img2, self.window_size, self.avg)

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.ssim_loss=SSIMLoss()
        self.l1_loss=nn.L1Loss()

    def forward(self, output,target):
        l1_loss=self.l1_loss(output,target)
        ssim_loss=self.ssim_loss(output,target)
        grad_loss=self.grad_loss(output,target)
        total_loss=0.1*ssim_loss+l1_loss+grad_loss

        return total_loss

    def grad_loss(self, output, target):
        output_grad_x=torch.abs(output[:,:,:,:-1]-output[:,:,:,1:])
        output_grad_y=torch.abs(output[:,:,:-1,:]-output[:,:,1:,:])
        target_grad_x=torch.abs(target[:,:,:,:-1]-target[:,:,:,1:])
        target_grad_y=torch.abs(target[:,:,:-1,:]-target[:,:,1:,:])
        grad_loss_x=F.l1_loss(output_grad_x, target_grad_x)
        grad_loss_y=F.l1_loss(output_grad_y, target_grad_y)

        return grad_loss_x+grad_loss_y