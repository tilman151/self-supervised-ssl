import torch


class DataHparamsMixin:
    def add_data_hparams(self, data):
        self.hparams.update(data.hparams)


class LoadEncoderMixin:
    def load_encoder(self, checkpoint_path, load_disc=False):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        encoder_state = self._extract_state_dict(checkpoint, model="encoder")
        self.encoder.load_state_dict(encoder_state)
        if load_disc:
            disc_state = self._extract_state_dict(checkpoint, model="domain_disc")
            self.domain_disc.load_state_dict(disc_state)

        self.hparams["pretrained_checkpoint"] = checkpoint_path
        self.encoder.norm_outputs = True

    def _extract_state_dict(self, checkpoint, model):
        encoder_state = {
            n.replace(model + ".", ""): weight
            for n, weight in checkpoint["state_dict"].items()
            if n.startswith(model)
        }

        return encoder_state

    def load_from_rbm(self, rbm):
        first_conv = self.encoder.layers[0]
        first_bn = self.encoder.layers[1]
        first_conv.weight.data = rbm.interaction.module.weight.data
        first_bn.bias.data = rbm.hidden.bias.data.squeeze()
