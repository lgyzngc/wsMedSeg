import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from util.util import prob_2_entropy,to_3dim,to_4dim
import numpy as np
import torch.nn.functional as F


class AttentionGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A',  'G_B', 'cycle_B', 'idt_B']

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A', 'o1_b', 'o2_b', 'a1_b', 'a2_b', 'i1_b']
        visual_names_B = ['real_B', 'fake_A', 'rec_B', 'o1_a', 'o2_a', 'a1_a', 'a2_a', 'i1_a']

        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        if self.opt.saveDisk:   
            self.visual_names = ['real_A', 'fake_B', 'rec_A', 'a1_b', 'a2_b']
            
        else:
            self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'our', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, 'our', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_content_A = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_content_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                                    opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain,
                                                    self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_C_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_C_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.real_C_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.real_C_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    # def entropy_loss(v):
    #     """
    #         Entropy loss for probabilistic prediction vectors
    #         input: batch_size x channels x h x w
    #         output: batch_size x 1 x h x w
    #     """
    #     assert v.dim() == 4
    #     n, c, h, w = v.size()
    #     return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c))

    # def prob_2_entropy(prob):
    #     """ convert probabilistic prediction maps to weighted self-information maps
    #     """
    #     n, c, h, w = prob.size()
    #     #entropy = -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)
    #     entropy = -torch.mul(prob, torch.log2(prob + 1e-30)) / (n * h * w)
    #     return entropy

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
       
        self.fake_B, self.o1_b, self.o2_b, \
        self.a1_b, self.a2_b, \
        self.i1_b,\
        self.fake_c_b= self.netG_A(self.real_A)  # G_A(A)  
        self.rec_A, _, _, \
        _, _, \
        _,_= self.netG_B(self.fake_B)   # G_B(G_A(A))


        self.fake_A, self.o1_a, self.o2_a, \
        self.a1_a, self.a2_a,\
        self.i1_a,\
        self.fake_c_a = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B, _, _,  \
        _, _,\
        _,_ = self.netG_A(self.fake_A)   # G_A(G_B(B))


      
        self.idt_A, _, _, \
        self.idt_A_att1, self.idt_A_att2, \
        self.idt_A_cont, _ = self.netG_A(self.real_B)

        self.idt_B, _, _, \
        self.idt_B_att1, self.idt_B_att2, \
        self.idt_B_cont, _ = self.netG_B(self.real_A)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_basicA(self, netD, real_a, real_b, fake_a, fake_b):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """

        # Reala
        pred_real_a = netD(real_a)
        loss_D_real_a = self.criterionGAN(pred_real_a, 0)
        # Real
        pred_real_b = netD(real_b)
        loss_D_real_b = self.criterionGAN(pred_real_b, 1)
        # Fake
        pred_fake_a = netD(fake_a.detach())
        loss_D_fake_a = self.criterionGAN(pred_fake_a, 2)
        # Fake
        pred_fake_b = netD(fake_b.detach())
        loss_D_fake_b = self.criterionGAN(pred_fake_b, 3)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real_a + loss_D_real_b + loss_D_fake_a + loss_D_fake_b)
        loss_D.backward()
        return loss_D

    def backward_D_basicB(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, 0)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, 2)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_A = self.backward_D_basicA(self.netD_A, self.real_A, self.real_B, fake_A, fake_B)

    def backward_D_content_A(self):
        fake_C_B = self.fake_C_B_pool.query(self.fake_c_b)
        real_C_B = self.real_C_B_pool.query(self.o2_b)
        self.loss_D_content_A = self.backward_D_basic(self.netD_content_A, real_C_B, fake_C_B)


    def backward_D_content_B(self):
        fake_C_A = self.fake_C_A_pool.query(self.fake_c_a)
        real_C_A = self.real_C_A_pool.query(self.o2_a)
        self.loss_D_content_B = self.backward_D_basic(self.netD_content_B, real_C_A, fake_C_A)



    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity   #0.5
        lambda_A = self.opt.lambda_A     #10
        lambda_B = self.opt.lambda_B     #10
        ones = torch.ones(self.real_A.size()).cuda()
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A, _, _,  \
            self.idt_A_att1, self.idt_A_att2, \
            self.idt_A_cont,_ = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B, _, _, \
            self.idt_B_att1, self.idt_B_att2, \
            self.idt_B_cont,_ = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), 1)*4
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_A(self.fake_A), 0)*4

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B


        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + \
                      self.loss_idt_A + self.loss_idt_B 
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        # self.backward_D_B()      # calculate graidents for D_B
        # self.backward_D_content_A()
        # self.backward_D_content_B()
        self.optimizer_D.step()  # update D_A and D_B's weights
