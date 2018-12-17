import torch
from torch.autograd import Variable
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn as nn
import torch.nn.parallel
import pdb

from vgg16 import Vgg16
import utils
import os

class CycleGANModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        # self.loss_names = ['D_A', 'G_A', 'identity', 'idt_A', 'D_B', 'G_B', 'style', 'idt_B', 'D_P','MSE_B', 'identity_B' ]
        self.loss_names = ['D_A', 'G_A', 'identity', 'D_B', 'G_B', 'style', 'D_P', 'MSE_B', 'identity_B']
		#self.loss_names = ['D_A', 'G_A', 'identity', 'idt_A', 'D_B', 'G_B', 'style', 'idt_B', 'D_P','MSE_B', 'identity_B']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_Y', 'f', 'rec_Y', 'real_D']
        visual_names_B = ['real_X', 'g', 'rec_X', 'real_C']
#        if self.isTrain and self.opt.lambda_identity > 0.0:
#            visual_names_A.append('idt_A')
#            visual_names_B.append('idt_B')

        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        #self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
        #                                opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_A = networks.define_G(6, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
       
#        self.netG_A = nn.DataParallel(self.netG_A, device_ids = [0, 1, 2]).cuda()
       
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
       
#        self.netG_B = nn.DataParallel(self.netG_B, device_ids = [0, 1, 2]).cuda()

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,                             
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
#            self.netD_A = nn.DataParallel(self.netD_A, device_ids = [0, 1, 2]).cuda()

            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

            self.netD_P = networks.define_D(6, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
          
#            self.netD_B = nn.DataParallel(self.netD_B, device_ids = [0, 1, 2]).cuda()

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionMSE = torch.nn.MSELoss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)
        self.print_networks(opt.verbose)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        # input_A = input['A' if AtoB else 'B']
        # input_B = input['B' if AtoB else 'A']

        #here let A = y, B = x
		
        input_Y = input['A']
        input_X = input['B']
        input_XY = torch.cat((input_X, input_Y), 1)
        input_C = input['C']
        input_D = input['D']
        # if len(self.gpu_ids) > 0:
        #     input_A = input_A.cuda(self.gpu_ids[0], async=True)
        #     input_B = input_B.cuda(self.gpu_ids[0], async=True)

        if len(self.gpu_ids) > 0:
            input_XY = input_XY.cuda(async=True)
            input_Y = input_Y.cuda(async=True)
            input_X = input_X.cuda(async=True)
            input_C = input_C.cuda(async=True)
            input_D = input_D.cuda(async=True)

        # self.input_A = input_A
        # self.input_B = input_B

        self.input_XY = input_XY
        self.input_Y = input_Y
        self.input_X = input_X
        self.input_C = input_C
        self.input_D = input_D

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        # self.real_A = Variable(self.input_A)
        # self.real_B = Variable(self.input_B)

        self.real_XY = Variable(self.input_XY)
        self.real_Y = Variable(self.input_Y)
        self.real_X = Variable(self.input_X)
        self.real_C = Variable(self.input_C)
        self.real_D = Variable(self.input_D)

#     def test(self):
# #        self.real_A = Variable(self.input_A, volatile=True)
#  #       self.fake_B = self.netG_A(self.real_A)
#    #     self.rec_A = self.netG_B(self.fake_B)
#
#         self.real_XY = Variable(self.input_XY, volatile=True)
#         self.g = self.netG_A(self.real_XY)
#         self.rec_X = self.netG_B(self.g)
#         self.real_X = Variable(self.input_X)
#
#      #   self.real_B = Variable(self.input_B, volatile=True)
#     #    self.fake_A = self.netG_B(self.real_B)
#       #  self.rec_B = self.netG_A(self.fake_A)
#
#         self.real_Y = Variable(self.input_Y, volatile=True)
#         self.f = self.netG_B(self.real_Y)
#
#         self.fg = torch.cat((self.f, self.g), 1)
#         self.rec_Y = self.netG_A(self.fg)


    def test(self):
        #        self.real_A = Variable(self.input_A, volatile=True)
        #       self.fake_B = self.netG_A(self.real_A)
        #     self.rec_A = self.netG_B(self.fake_B)
        self.real_C = Variable(self.input_C)
        self.real_D = Variable(self.input_D)
        self.real_XY = Variable(self.input_XY, volatile=True)
        self.g = self.netG_A(self.real_XY)
        self.rec_X = self.netG_B(self.g)
        self.real_X = Variable(self.input_X)

        #   self.real_B = Variable(self.input_B, volatile=True)
        #    self.fake_A = self.netG_B(self.real_B)
        #  self.rec_B = self.netG_A(self.fake_A)

        self.real_Y = Variable(self.input_Y, volatile=True)
        self.f = self.netG_B(self.real_Y)

        self.fg = torch.cat((self.f, self.g), 1)
        self.rec_Y = self.netG_A(self.fg)

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        # fake_B = self.fake_B_pool.query(self.fake_B)
        # self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        g = self.fake_B_pool.query(self.g)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_Y, g)

    def backward_D_B(self):
        # fake_A = self.fake_A_pool.query(self.fake_A)
        # self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        f = self.fake_A_pool.query(self.f)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_X, f)

    def backward_D_P(self):
        # fake_A = self.fake_A_pool.query(self.fake_A)
        # self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        g = self.g
        self.yw = torch.cat((self.real_Y, self.real_C), 1)
        self.yg = torch.cat((self.real_Y, self.g), 1)
        self.loss_D_P = self.backward_D_basic(self.netD_P, self.yw, self.yg)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B


        # Identity loss
        # if lambda_idt > 0:
        #     # G_A should be identity if real_B is fed.
        #     self.idt_A = self.netG_A(self.real_B)
        #     self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
        #     # G_B should be identity if real_A is fed.
        #     self.idt_B = self.netG_B(self.real_A)
        #     self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        # else:
        #     self.loss_idt_A = 0
        #     self.loss_idt_B = 0
        self.loss_idt_A = 0
        self.loss_idt_B = 0
        # GAN loss D_A(G_A(A))
        # self.fake_B = self.netG_A(self.real_A)
        # self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)

        #self.g = self.netG_A(self.real_XY)
        self.g = self.netG_A(self.real_XY)
        self.loss_G_A = self.criterionGAN(self.netD_A(self.g), True)

        # GAN loss D_B(G_B(B))
        # self.fake_A = self.netG_B(self.real_B)
        # self.loss_G_B = self.criterionGAN(self.netD_B(self .fake_A), True)
        self.f = self.netG_B(self.real_Y)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.f), True)

        # Forward cycle loss
        # self.rec_A = self.netG_B(self.fake_B)
        # self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A

        #identity loss
        self.rec_X = self.netG_B(self.g)
      #  self.real_X.cuda()
        self.loss_identity = self.criterionCycle(self.rec_X, self.real_X) * lambda_A

        # # Backward cycle loss
        # self.rec_B = self.netG_A(self.fake_A)
        # self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        #style loss
        self.fg = torch.cat((self.f, self.g), 1)
        self.rec_Y = self.netG_A(self.fg)
  #      self.real_Y.cuda()
        self.loss_style = self.criterionCycle(self.rec_Y, self.real_Y) * lambda_B

        #pdb.set_trace()
        #MSE loss
        self.loss_MSE_B = (self.criterionMSE(self.f, self.real_D))

        #A identity loss, compare f and D
        vgg = Vgg16()
        utils.init_vgg16(self.opt.vgg_model_dir)
        vgg.load_state_dict(torch.load(os.path.join(self.opt.vgg_model_dir, "vgg16.weight")))
        if len(self.gpu_ids) > 0:
            vgg = vgg.cuda()

        real_D_copy = Variable(self.real_D.data.clone(), volatile=True)

        #vgg returns a list of four value
        #pdb.set_trace()
        features_f = vgg(self.f)
        features_D = vgg(real_D_copy)

        #pdb.set_trace()
        f_D_c = Variable(features_D[1].data, requires_grad=False)
        self.loss_identity_B = self.criterionMSE(features_f[1], f_D_c)

        # combined loss
        #self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        # self.loss_G =  5 * self.loss_G_A +  self.loss_G_B + self.loss_identity +  self.loss_style + self.loss_idt_A + self.loss_MSE_B
        # self.loss_G = 5 * self.loss_G_A + self.loss_G_B + self.loss_identity + self.loss_style + \
        #               self.loss_idt_A + self.loss_idt_B + self.loss_identity_B
        self.loss_G = 5 * self.loss_G_A + self.loss_G_B + self.loss_identity + self.loss_style + \
                      self.loss_idt_A + self.loss_idt_B + self.loss_identity_B + self.loss_MSE_B
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        #pdb.set_trace()
        self.backward_G()
        self.optimizer_G.step()
        # D_A and D_B
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.backward_D_P()
        self.optimizer_D.step()

        # forward
        self.forward()
        # G_A and G_B
       # self.optimizer_G.zero_grad()
       # pdb.set_trace()
        #self.backward_G()
        #self.optimizer_G.step()
        # D_A and D_B
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.backward_D_P()
        self.optimizer_D.step()




