import torch
from gen_model.network.Adver_network import ReverseLayerF, Discriminator
from gen_model.network.Common_network import linear_classifier
from gen_model.network.Feature_extraction_network import CNN_Feature_Extraction_Network
from gen_model.network.Denoise_network import DenoiseModel
from gen_model.network.Noise_distinct_network import NoiseEncoder, NoiseDistinct_reparameterize
import torch.nn.functional as F
from torch import nn, mean


class DiffusionModel(nn.Module):
    def __init__(self, n_t, beta_1, beta_t, device, conv1_in_channels, conv1_out_channels, conv2_out_channels,
                 conv_kernel_size_num, pool_kernel_size_num,
                 denoise_hidden_size, in_features_size,
                 num_class, discriminator_dis_hidden,
                 ReverseLayer_latent_domain_alpha,
                 Alpha, Beta, Delta):
        super(DiffusionModel, self).__init__()

        self.device = device

        # Feature_Extraction
        self.featurizer = CNN_Feature_Extraction_Network(conv1_in_channels,
                                                         conv1_out_channels,
                                                         conv2_out_channels,
                                                         conv_kernel_size_num,
                                                         pool_kernel_size_num, self.device)

        # Noise distinct
        self.noise_distinct_in_features = in_features_size
        self.noise_encoder = NoiseEncoder(self.noise_distinct_in_features, self.device)
        self.noise_distinct_reparameterize = NoiseDistinct_reparameterize(self.device)

        self.num_class = num_class
        self.diffuse_classify_st = linear_classifier(self.noise_distinct_in_features, self.num_class * 2, self.device)
        self.diffuse_domains = linear_classifier(self.noise_distinct_in_features, 2, self.device)

        # Denoise
        self.denoise_hidden_size = denoise_hidden_size
        self.n_t = n_t
        self.denoise_model = DenoiseModel(self.n_t, self.denoise_hidden_size, self.noise_distinct_in_features, self.device)

        self.discriminator_dis_hidden = discriminator_dis_hidden
        self.denoise_discriminator_classify_st = Discriminator(self.noise_distinct_in_features,
                                                               self.discriminator_dis_hidden, 2 * self.num_class, self.device)
        self.denoise_discriminator_classify_d = Discriminator(self.noise_distinct_in_features,
                                                              self.discriminator_dis_hidden, 2, self.device)
        self.denoise_classify_source = linear_classifier(self.noise_distinct_in_features, self.num_class, self.device)

        # Parameters
        self.beta_1 = beta_1
        self.beta_t = beta_t
        self.betas = torch.linspace(self.beta_1, self.beta_t, self.n_t).to(self.device)
        self.alphas = 1 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, axis=0).to(self.device)

        self.r_alphas_bar = torch.sqrt(self.alphas_bar).to(self.device)
        self.r_1m_alphas_bar = torch.sqrt(1 - self.alphas_bar).to(self.device)

        self.inv_r_alphas = torch.pow(self.alphas, -0.5).to(self.device)
        self.pre_noise_terms = self.betas / self.r_1m_alphas_bar
        self.sigmas = torch.pow(self.betas, 0.5).to(self.device)

        self.ReverseLayer_latent_domain_alpha = ReverseLayer_latent_domain_alpha

    def diffuse(self, x, t, eps):
        t = t - 1
        diffused = self.r_alphas_bar[t] * x + self.r_1m_alphas_bar[t] * eps
        return eps, diffused

    def forward(self, ST_x, S_x_index):
        raw_feature_x, one_dimension_x = self.featurizer(ST_x)
        mean, var = self.noise_encoder(one_dimension_x)
        noise = self.noise_distinct_reparameterize(mean, var)

        add_noise_classify_st = self.diffuse_classify_st(noise)
        add_noise_classify_d = self.diffuse_domains(noise)

        t = torch.randint(1, self.n_t + 1, (len(ST_x),)).unsqueeze(1).to(self.device)
        eps, diffused = self.diffuse(one_dimension_x, t, noise)

        pred_eps = self.denoise_model(diffused, t)

        disc_classes = ReverseLayerF.apply(pred_eps, self.ReverseLayer_latent_domain_alpha)
        denoise_classify_st_reverse = self.denoise_discriminator_classify_st(disc_classes)

        disc_domains = ReverseLayerF.apply(pred_eps, self.ReverseLayer_latent_domain_alpha)
        denoise_classify_d_reverse = self.denoise_discriminator_classify_d(disc_domains)

        denoise_classify_s = self.denoise_classify_source(pred_eps[S_x_index])

        return add_noise_classify_st, add_noise_classify_d, eps, pred_eps, denoise_classify_st_reverse, denoise_classify_d_reverse, denoise_classify_s

    def update(self, ST_data, S_x_index, opt, num_round):
        all_x = ST_data[0].float().to(self.device)
        all_c = ST_data[1].long().to(self.device)
        all_d = ST_data[2].long().to(self.device)

        s_c = all_c[S_x_index]

        add_noise_classify_st, add_noise_classify_d, eps, pred_eps, denoise_classify_st_reverse, denoise_classify_d_reverse, denoise_classify_s = self.forward(
            all_x, S_x_index)

        NOISE_CLASS_ST_L = F.cross_entropy(add_noise_classify_st, all_c)
        NOISE_CLASS_D_L = F.cross_entropy(add_noise_classify_d, all_d)

        RECON_L = mean((eps - pred_eps) ** 2)

        DENOISE_CLASS_ST_R_L = F.cross_entropy(denoise_classify_st_reverse, all_c)
        DENOISE_CLASS_D_R_L = F.cross_entropy(denoise_classify_d_reverse, all_d)
        DENOISE_CLASS_R_L = DENOISE_CLASS_ST_R_L + DENOISE_CLASS_D_R_L

        DENOISE_CLASS_S_L = F.cross_entropy(denoise_classify_s, s_c)

        NOISE_CLASS_L = NOISE_CLASS_ST_L + NOISE_CLASS_D_L + DENOISE_CLASS_S_L

        if (num_round // 10) % 2 == 0:
            loss = RECON_L + NOISE_CLASS_L
        else:
            loss = RECON_L + DENOISE_CLASS_R_L

        opt.zero_grad()
        loss.backward()
        opt.step()

        return {'total': loss.item(), 'NOISE_RECON_L': RECON_L.item(), 'NOISE_CLASS_L': NOISE_CLASS_L.item(),
                'DENOISE_CLASS_R_L': DENOISE_CLASS_R_L.item(), 'DENOISE_CLASS_S_L': DENOISE_CLASS_S_L.item()}

    def predict(self, x):
        raw_feature_x, one_dimension_x = self.featurizer(x)
        mean, var = self.noise_encoder(one_dimension_x)
        noise = self.noise_distinct_reparameterize(mean, var)

        t = torch.randint(1, self.n_t + 1, (len(x),)).unsqueeze(1).to(self.device)
        eps, diffused = self.diffuse(one_dimension_x, t, noise)

        pred_eps = self.denoise_model(diffused, t)

        denoise_classify_c = self.denoise_classify_source(pred_eps)

        return denoise_classify_c
