#scpa 
class SCPA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SCPA,self).__init__()
        self.branch1_conv1=nn.Conv2d(in_channels, out_channels,1)
        self.branch1_conv2=nn.Conv2d(out_channels, out_channels,3, padding=1)
        self.sigmoid=nn.Sigmoid()
        self.branch1_conv3=nn.Conv2d(out_channels,out_channels,1)
        self.branch1_conv4=nn.Conv2d(out_channels,out_channels,3,padding=1)

        self.branch2_conv1=nn.Conv2d(in_channels,out_channels, 1)
        self.branch2_conv2=nn.Conv2d(out_channels,out_channels,3 ,padding=1)
        
        self.final_conv=nn.Conv2d(out_channels,out_channels, 1)
        
    def forward(self,x):
        #branch1
        branch1=self.branch1_conv1(x)
        branch1a=self.branch1_conv3(branch1)
        branch1b=self.branch1_conv2(branch1)
        sigmoid_1a=self.sigmoid(branch1a)
        branch1_final1=(branch1b)*(sigmoid_1a)
        branch1_final2=self.branch1_conv4(branch1_final1)
        
        #branch2
        branch2=self.branch2_conv1(x)
        branch2_final=self.branch2_conv2(branch2)
        
        combine_branch12=branch1_final2+branch2_final
        seventh_output=self.final_conv(combine_branch12)
        
        SCPA_output=seventh_output+x
        return SCPA_output
    
#convolution block
class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,padding=1,use_batchnorm=True,activation=nn.ReLU):
        super(ConvBlock, self).__init__()
        self.use_batchnorm=use_batchnorm
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,padding, bias=not use_batchnorm)
        self.bn=nn.BatchNorm2d(out_channels)
        self.activation=activation()
        
    def forward(self, x):
        batch_size=x.size(0)
        input_channels=x.size(1)
        height,width=x.shape[-2:]

        if input_channels!=self.conv.in_channels:
            x=x.view(batch_size,self.conv.in_channels,height,width)

        x=self.conv(x)
        if self.use_batchnorm:
            x=self.bn(x)
        x=self.activation(x)
        return x
    
#inverse residual block
class inverted_residual_block(nn.Module):
    def __init__(self,in_channels,out_channels,expansion_factor,stride):
        super(inverted_residual_block,self).__init__()
        self.stride=stride
        hidden_dmi=in_channels*expansion_factor
        self.use_residual=self.stride==1 and in_channels==out_channels

        layers=[]
        if expansion_factor!= 1:
            layers.append(nn.Conv2d(in_channels,hidden_dmi,kernel_size=1,bias=False))
            layers.append(nn.BatchNorm2d(hidden_dmi))
            layers.append(nn.ReLU6(inplace=True))

        layers.append(nn.Conv2d(hidden_dmi,hidden_dmi,kernel_size=3,stride=1,padding=1,groups=hidden_dmi,bias=False))
        layers.append(nn.BatchNorm2d(hidden_dmi))
        layers.append(nn.ReLU6(inplace=True))

        layers.append(nn.Conv2d(hidden_dmi, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv=nn.Sequential(*layers)

    def forward(self,x):
        if self.use_residual:
            return x+self.conv(x)
        else:
            return self.conv(x)
        
#coord conv block
class coord_conv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(coord_conv,self).__init__()
        self.conv=nn.Conv2d(in_channels+2,out_channels,3,padding=1)
    def forward(self,x):
        batch_size,_,height,width=x.size()
        device=x.device 
        xx=torch.arange(width,device=device).repeat(height, 1)
        yy=torch.arange(height,device=device).view(-1,1).repeat(1, width)
        xx=xx.float()/(width-1)
        yy=yy.float()/(height-1)
        xx=xx.repeat(batch_size,1,1).unsqueeze(1)
        yy=yy.repeat(batch_size,1,1).unsqueeze(1)

        x=torch.cat([x,xx,yy], dim=1)
        x=self.conv(x)
        return x

'''class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = 1 if kernel_size == 3 else 3
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out=torch.mean(x, dim=1, keepdim=True)
        max_out,_=torch.max(x, dim=1, keepdim=True)
        x=torch.cat([avg_out, max_out], dim=1)
        x=self.conv(x)
        return x*self.sigmoid(x)'''
class AttentionBlock(nn.Module):
    def __init__(self,in_channels):
        super(AttentionBlock,self).__init__()
        self.conv1=nn.Conv2d(in_channels,1,kernel_size=1)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        att_weights=self.conv1(x)
        att_weights=self.sigmoid(att_weights)
        output=x*att_weights
        return output
    
#residual attention block
class residual_block(nn.Module):
    def __init__(self, in_channels):
        super(residual_block,self).__init__()
        self.conv1=nn.Conv2d(in_channels,in_channels,3,padding=1)
        self.bn1=nn.BatchNorm2d(in_channels)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(in_channels,in_channels,3,padding=1)
        self.bn2=nn.BatchNorm2d(in_channels)
    def forward(self,x):
        residual=x
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.conv2(out)
        out=self.bn2(out)
        return x
    
class dense_block(nn.Module):
    def __init__(self, in_channels, growth_rate=32, num_layers=4):
        super(dense_block, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Conv2d(in_channels + i*(growth_rate), growth_rate, kernel_size=3, padding=1))
            self.layers.append(nn.BatchNorm2d(growth_rate))
            self.layers.append(nn.ReLU(inplace=True))

    def forward(self, x):
        features=[x]
        for i in range(0, len(self.layers), 3):
            out=self.layers[i](torch.cat(features, 1))
            out=self.layers[i+1](out)
            out=self.layers[i+2](out)
            features.append(out)
        return torch.cat(features, 1)

class spatial_attention(nn.Module):
    def __init__(self,kernel_size=3):
        super(spatial_attention,self).__init__()
        assert kernel_size in (3,7), "nigga"
        if kernel_size==3:
            padding=1
        else:
            padding=3
        self.conv=nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        avg_out=torch.mean(x, dim=1, keepdim=True)
        max_out,_=torch.max(x, dim=1, keepdim=True)
        x=torch.cat([avg_out, max_out], dim=1)
        x=self.conv(x)
        return x * self.sigmoid(x)

class residual_dense_attention(nn.Module):
    def __init__(self,in_channels, growth_rate=32,num_layers=4,reduction=16,kernel_size=7):
        super(residual_dense_attention,self).__init__()
        self.residualblock=residual_block(in_channels)
        self.denseblock=dense_block(in_channels,growth_rate,num_layers)
        self.attentionblock=spatial_attention(kernel_size)
    
    def forward(self,x):
        residual_out=self.residualblock(x)
        dense_out=self.denseblock(residual_out)
        attention_out=self.attentionblock(dense_out)
        return attention_out+x
    
#upsampling & downsampling
class downsampling_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(downsampling_block, self).__init__()
        self.conv=nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x=self.conv(x)
        x=self.bn(x)
        x=self.relu(x)
        x=self.pool(x)
        return x

class upsampling_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upsampling_block, self).__init__()
        self.upconv=nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv=nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)

    def forward(self, x):
        x=self.upconv(x)
        x=self.conv(x)
        x=self.bn(x)
        x=self.relu(x)
        return x
    
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        def up_conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.ReLU(inplace=True)
            )

        self.enc1=conv_block(in_channels, 64)
        self.enc2=conv_block(64, 128)
        self.enc3=conv_block(128, 256)
        self.enc4=conv_block(256, 512)
        
        self.bottleneck=conv_block(512, 1024)

        self.upconv4=up_conv_block(1024, 512)
        self.dec4=conv_block(1024, 512)
        
        self.upconv3=up_conv_block(512, 256)
        self.dec3=conv_block(512, 256)
        
        self.upconv2=up_conv_block(256, 128)
        self.dec2=conv_block(256, 128)
        
        self.upconv1=up_conv_block(128, 64)
        self.dec1=conv_block(128, 64)
        
        self.conv_final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1=self.enc1(x)
        enc2=self.enc2(F.max_pool2d(enc1, kernel_size=2, stride=2))
        enc3=self.enc3(F.max_pool2d(enc2, kernel_size=2, stride=2))
        enc4=self.enc4(F.max_pool2d(enc3, kernel_size=2, stride=2))
        bottleneck=self.bottleneck(F.max_pool2d(enc4, kernel_size=2, stride=2))
        dec4=self.upconv4(bottleneck)
        dec4=torch.cat((dec4,enc4),dim=1)
        dec4=self.dec4(dec4)
        
        dec3=self.upconv3(dec4)
        dec3=torch.cat((dec3,enc3),dim=1)
        dec3=self.dec3(dec3)
        
        dec2=self.upconv2(dec3)
        dec2=torch.cat((dec2,enc2),dim=1)
        dec2=self.dec2(dec2)
        
        dec1=self.upconv1(dec2)
        dec1=torch.cat((dec1,enc1),dim=1)
        dec1=self.dec1(dec1)
        
        return self.conv_final(dec1)
    
#model_architecture
class final_imagelab_model(nn.Module):
    def __init__(self,in_channels):
        super(final_imagelab_model,self).__init__()
        self.coord=coord_conv(3,5)
        self.scpa1=SCPA(5,5)
        self.scpa2=SCPA(5,5)
        self.scpa3=SCPA(5,5)
        self.scpa4=SCPA(5,5) 
        self.scpa5=SCPA(5,5)
        self.conv_coordvala=nn.Conv2d(5,3,kernel_size=3,padding=1)
        self.coord_relu=nn.ReLU(inplace=True)
        self.conv_final=nn.Conv2d(3,3,kernel_size=3,padding=1)
        self.final_relu=nn.ReLU(inplace=True)
        #irb
        self.irb1=inverted_residual_block(3,3,6,1)
        self.irb2=inverted_residual_block(3,3,6,1)
        self.irb3=inverted_residual_block(3,3,6,1)
        self.irb4=inverted_residual_block(3,3,6,1)
        
        self.conv_denvala1=nn.Conv2d(3,3, kernel_size=3, padding=1)
        self.denvala1_relu=nn.ReLU(inplace=True)
        self.conv_denvala2=nn.Conv2d(3,3, kernel_size=3, padding=1)
        self.denvala2_relu=nn.ReLU(inplace=True)
        self.attention=AttentionBlock(3)
        self.rda1=residual_dense_attention(3,growth_rate=32,num_layers=4,reduction=16,kernel_size=7)
        self.rda2=residual_dense_attention(3,growth_rate=32,num_layers=4,reduction=16,kernel_size=7)
        self.rda3=residual_dense_attention(64,growth_rate=32,num_layers=4,reduction=16,kernel_size=7)
        self.rda4=residual_dense_attention(64,growth_rate=32,num_layers=4,reduction=16,kernel_size=7)
        self.rda5=residual_dense_attention(128,growth_rate=32,num_layers=4,reduction=16,kernel_size=7)
        self.rda6=residual_dense_attention(128,growth_rate=32,num_layers=4,reduction=16,kernel_size=7)
        self.rda7=residual_dense_attention(64,growth_rate=32,num_layers=4,reduction=16,kernel_size=7)
        self.rda8=residual_dense_attention(64,growth_rate=32,num_layers=4,reduction=16,kernel_size=7)
        self.upsample1=upsampling_block(128,64)
        self.upsample2=upsampling_block(64,3)
        self.downsample1=downsampling_block(3,64)
        self.downsample2=downsampling_block(64,128)
        
    def forward(self,x):
            #branch1(denoiser)
            branch1=self.conv_denvala1(x)
            branch1=self.denvala1_relu(branch1)
            branch1=self.irb1(branch1)
            branch1=self.irb2(branch1)
            branch1=self.irb3(branch1)
            branch1=self.irb4(branch1)
            branch1=self.attention(branch1)
            branch1=self.conv_denvala2(branch1)
            branch1_output=self.denvala2_relu(branch1)
            
            #branch2(coord,scpa,encoder,decoder)
            branch2=self.coord(x)
            branch2=self.scpa1(branch2)
            branch2=self.scpa2(branch2)
            branch2=self.scpa3(branch2)
            branch2=self.scpa4(branch2)
            branch2=self.scpa5(branch2)
            branch2=self.conv_coordvala(branch2)
            branch2=self.coord_relu(branch2)
            branch2_sub=(branch2)+x
            #encoder
            branch2_sub=self.rda1(branch2_sub)
            branch2_sub=self.rda2(branch2_sub)
            branch2_sub_1=self.downsample1(branch2_sub)
            branch2_subb=self.rda3(branch2_sub_1)
            branch2_subb=self.rda4(branch2_subb)
            branch2_sub_2=self.downsample2(branch2_subb)
            #decoder
            branch2_subb=self.rda5(branch2_sub_2)
            branch2_subb=self.rda6(branch2_subb)
            branch2_subb=self.upsample1(branch2_subb)
            branch2_subb=self.rda7(branch2_subb + branch2_sub_1)
            branch2_subb=self.rda8(branch2_subb)
            branch2_subb=self.upsample2(branch2_subb)
            #conv
            branch2_subb=self.conv_final(branch2_subb)
            branch2_output=self.final_relu(branch2_subb)
            return (branch1_output)+(branch2_output)